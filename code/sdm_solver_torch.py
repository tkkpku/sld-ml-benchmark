"""GPU-batched first-order SDM solver (PyTorch, CUDA) for the SLD-ML benchmark.

Mathematically identical to code/sdm_solver.py (fixed signs, fixed product
order, linear-in-delay interval flow), but evaluates all grid points of one
parameter case in parallel on the GPU:

    rho[i, j] = spectral radius of D(n_j, a_p_i, m)

The transition matrix is assembled interval-by-interval with batched 8x8
matrix exponentials (block-exponential identity) and batched (4+2m)^2 dense
matrices. Chunking over depth rows keeps GPU memory well below 8 GB even for
m=160 (dim=324).

Usage (inside WSL2 with the yolo_env PyTorch environment):
    python code/sdm_solver_torch.py --case 0 --m 80 --out results/_archive_benchmark_v1/cpu_gpu_check.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch


def milling_params(aD: float = 0.5, zeta: float = 0.011, fn: float = 922.0):
    """Same defaults as MillingParams in sdm_solver.py (2-DOF benchmark)."""
    return {
        "N": 2,
        "fn": fn,
        "zeta": zeta,
        "mt": 0.03993,
        "Kt": 6.0e8,
        "Kn": 2.0e8,
        "ad": -1,          # down milling
        "aD": aD,
        "wn": 2.0 * np.pi * fn,
        "r": 2.0e8 / 6.0e8,
    }


def direction_coeffs_batch(phi: torch.Tensor, p: dict) -> torch.Tensor:
    """Batch 2x2 directional matrix A(phi) for one tooth.

    phi: (...,) tensor of tooth angles. Returns (..., 2, 2).
    """
    if p["ad"] < 0:                       # down milling
        phi_st = 0.0
        phi_ex = float(np.arccos(1.0 - 2.0 * p["aD"]))
    else:
        phi_st = float(np.arccos(2.0 * p["aD"] - 1.0))
        phi_ex = np.pi
    g = ((phi >= phi_st) & (phi <= phi_ex)).to(torch.float64)
    s2 = torch.sin(2.0 * phi)
    c2 = torch.cos(2.0 * phi)
    r = p["r"]
    a = torch.stack([
        torch.stack([-s2 - r * (1.0 - c2), -(1.0 + c2) + r * s2], dim=-1),
        torch.stack([(1.0 - c2) - r * s2, s2 - r * (1.0 + c2)], dim=-1),
    ], dim=-2) / 2.0
    return g[..., None, None] * a


def direction_matrix_batch(t: torch.Tensor, n_rpm: torch.Tensor,
                           p: dict) -> torch.Tensor:
    """A(t) summed over teeth. t, n_rpm: (1, W) -> (1, W, 2, 2)."""
    omega_s = 2.0 * np.pi * n_rpm / 60.0
    pitch = 2.0 * np.pi / p["N"]
    A = torch.zeros(t.shape + (2, 2), dtype=torch.float64, device=t.device)
    for j in range(p["N"]):
        phi = torch.remainder(omega_s * t + j * pitch, 2.0 * np.pi)
        A = A + direction_coeffs_batch(phi, p)
    return A


def _interval_flow_linear_batch(L: torch.Tensor, R: torch.Tensor,
                                dt: torch.Tensor):
    """Batched block-exponential interval flow.

    L: (B, 4, 4), R: (B, 4, 2), dt: (B,) -> Phi, Psi0, Psi1 (B, 4, 4/2).
    """
    B = L.shape[0]
    M8 = torch.zeros((B, 8, 8), dtype=torch.float64, device=L.device)
    M8[:, :4, :4] = L
    M8[:, :4, 4:6] = R
    M8[:, 4:6, 6:8] = torch.eye(2, dtype=torch.float64, device=L.device)
    E = torch.linalg.matrix_exp(M8 * dt[:, None, None])
    Phi = E[:, :4, :4].clone()
    Psi = E[:, :4, 4:6].clone()
    Psi0 = E[:, :4, 6:8].clone() / dt[:, None, None]
    Psi1 = Psi - Psi0
    return Phi, Psi0, Psi1


@torch.no_grad()
def build_transition_batch_fast(n_rpms: np.ndarray, a_ps: np.ndarray, p: dict,
                                m: int, device: str = "cuda",
                                chunk_rows: int = 128) -> np.ndarray:
    """Build all one-period transition matrices with one batched expm call.

    Returns D array of shape (len(a_ps)*len(n_rpms), 4+2m, 4+2m) in float32.
    Depth rows are processed in chunks to bound GPU memory for large m.
    """
    W = len(n_rpms)
    H = len(a_ps)
    dim = 4 + 2 * m
    n_t = torch.tensor(n_rpms, dtype=torch.float64, device=device)[None, :]
    ap_t = torch.tensor(a_ps, dtype=torch.float64, device=device)
    tau_col = 60.0 / (p["N"] * n_rpms)
    dt_w = torch.tensor(tau_col / m, dtype=torch.float64, device=device)
    wn = p["wn"]
    eye2 = torch.eye(2, dtype=torch.float64, device=device)
    eye2f = torch.eye(2, dtype=torch.float32, device=device)
    parts = []
    for h0 in range(0, H, chunk_rows):
        h1 = min(h0 + chunk_rows, H)
        Bh = h1 - h0
        B = Bh * W
        ap_b = ap_t[h0:h1]
        kappa = (ap_b * p["Kt"] / p["mt"])[:, None]   # (Bh, 1)
        kap = kappa.expand(Bh, W).reshape(B)
        dt_all = dt_w.repeat(Bh)                      # (B,)
        M8 = torch.zeros((m, B, 8, 8), dtype=torch.float64, device=device)
        for i in range(m):
            t_mid = (i + 0.5) * dt_w[None, :]
            A = direction_matrix_batch(t_mid, n_t, p)[0]   # (W, 2, 2)
            A = A[None, :, :, :].expand(Bh, W, 2, 2).reshape(B, 2, 2)
            L = torch.zeros((B, 4, 4), dtype=torch.float64, device=device)
            L[:, 0, 2] = 1.0
            L[:, 1, 3] = 1.0
            L[:, 2:4, :2] = -wn * wn * eye2 + kap[:, None, None] * A
            L[:, 2:4, 2:] = -2.0 * p["zeta"] * wn * eye2
            R = torch.zeros((B, 4, 2), dtype=torch.float64, device=device)
            R[:, 2:, :] = -kap[:, None, None] * A
            M8[i, :, :4, :4] = L
            M8[i, :, :4, 4:6] = R
            M8[i, :, 4:6, 6:8] = eye2
            M8[i] = M8[i] * dt_all[:, None, None]
        E = torch.linalg.matrix_exp(M8.reshape(m * B, 8, 8))
        E = E.reshape(m, B, 8, 8)
        Phi = E[:, :, :4, :4].to(torch.float32)
        Psi0 = (E[:, :, :4, 6:8] / dt_all[None, :, None, None]).to(torch.float32)
        Psi1 = (E[:, :, :4, 4:6]
                - E[:, :, :4, 6:8] / dt_all[None, :, None, None]
                ).to(torch.float32)
        D = torch.eye(dim, dtype=torch.float32, device=device)
        D = D.expand(B, dim, dim).clone()
        for i in range(m - 1, -1, -1):
            Di = torch.zeros((B, dim, dim), dtype=torch.float32, device=device)
            Di[:, :4, :4] = Phi[i]
            if m == 1:
                Di[:, :4, :2] += Psi1[i]
                Di[:, :4, 4:6] = Psi0[i]
            else:
                Di[:, :4, 4 + 2 * (m - 2):4 + 2 * (m - 1)] = Psi1[i]
                Di[:, :4, 4 + 2 * (m - 1):4 + 2 * m] = Psi0[i]
            Di[:, 4:6, :2] = eye2f
            for k in range(1, m):
                Di[:, 4 + 2 * k:6 + 2 * k,
                   4 + 2 * (k - 1):6 + 2 * (k - 1)] = eye2f
            D = D @ Di
        parts.append(D.cpu().numpy())
        torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def spectral_radius_median_batch(D: torch.Tensor, iters: int = 500,
                                 tail: int = 250) -> torch.Tensor:
    """Robust spectral-radius estimate for non-normal SDM transition matrices.

    D: (B, n, n). Power iteration on a real matrix with a complex-conjugate
    dominant pair produces a norm sequence that oscillates around the true
    radius instead of converging monotonically. The median of the tail
    norm sequence estimates |lambda_max| within ~1e-4..6e-3 (validated
    against full eigendecomposition); points near the stability boundary
    are corrected exactly by the caller.
    """
    B, n, _ = D.shape
    g = torch.Generator(device=D.device).manual_seed(0)
    v = torch.randn(B, n, 1, device=D.device, generator=g,
                    dtype=D.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
    seq = torch.zeros(iters, B, device=D.device, dtype=D.dtype)
    for k in range(iters):
        w = D @ v
        nrm = w.norm(dim=1, keepdim=True).clamp_min(1e-300)
        seq[k] = nrm.reshape(-1)
        v = w / nrm
    tail_seq = seq[iters - tail:]
    return tail_seq.median(dim=0).values


@torch.no_grad()
def correct_boundary_batch(D: torch.Tensor, rho_est: torch.Tensor,
                           band: float = 0.03) -> torch.Tensor:
    """Exact spectral radius for matrices whose estimate falls in a band
    around the stability boundary |rho-1| < band (exact eigendecomposition,
    only for the few boundary points)."""
    B, n, _ = D.shape
    mask = (rho_est - 1.0).abs() < band
    idx = torch.nonzero(mask).squeeze(-1)
    if idx.numel() == 0:
        return rho_est
    Dsel = D[idx].double()
    vals = torch.linalg.eigvals(Dsel)
    rho_exact = vals.abs().amax(dim=-1).to(rho_est.dtype)
    out = rho_est.clone()
    out[idx] = rho_exact
    return out


@torch.no_grad()
def spectral_radius_median_long(D: torch.Tensor, steps: int = 2000,
                                tail: int = 1000,
                                return_iqr: bool = False):
    """Robust spectral-radius estimate for non-normal SDM matrices.

    D: (B, n, n). For a real matrix whose dominant spectrum is a
    complex-conjugate pair, the single-step norm sequence ||D v_k||
    oscillates around the true radius instead of converging; the median of
    a long tail of this sequence converges to the spectral radius
    (validated: tail-1000 median error ~4e-4 at rho=1.66 after 2000 steps).
    Points near the boundary are corrected exactly by the caller.
    """
    B, n, _ = D.shape
    g = torch.Generator(device=D.device).manual_seed(0)
    v = torch.randn(B, n, 1, device=D.device, generator=g,
                    dtype=D.dtype)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
    seq = torch.zeros(tail, B, device=D.device, dtype=D.dtype)
    for k in range(steps):
        w = D @ v
        nrm = w.norm(dim=1, keepdim=True).clamp_min(1e-300)
        if k >= steps - tail:
            seq[k - (steps - tail)] = nrm.reshape(-1)
        v = w / nrm
    med = seq.median(dim=0).values
    if return_iqr:
        q75, q25 = torch.quantile(seq, 0.75, dim=0), torch.quantile(seq, 0.25, dim=0)
        return med, (q75 - q25)
    return med


@torch.no_grad()
def floquet_radii_matrixfree(n_rpms: np.ndarray, a_ps: np.ndarray, p: dict,
                             m: int, device: str = "cuda",
                             iters: int = 300, tol: float = 1e-7,
                             chunk_rows: int = 64) -> np.ndarray:
    """Matrix-free power iteration: apply D to vectors without building D.

    Cost per iteration is O(m * B * const) instead of O(B * dim^3), so the
    full 128x80 grid becomes cheap on GPU.
    """
    W = len(n_rpms)
    H = len(a_ps)
    n_t = torch.tensor(n_rpms, dtype=torch.float64, device=device)[None, :]
    ap_t = torch.tensor(a_ps, dtype=torch.float64, device=device)
    dim = 4 + 2 * m
    tau_col = 60.0 / (p["N"] * n_rpms)
    eye2 = torch.eye(2, dtype=torch.float64, device=device)
    out = np.empty((H, W))
    g = torch.Generator(device=device).manual_seed(0)
    dt_w = torch.tensor(tau_col / m, dtype=torch.float64, device=device)

    # Precompute the per-interval directional matrices for all speeds.
    A_list = []
    for i in range(m - 1, -1, -1):
        t_mid = (i + 0.5) * dt_w[None, :]
        A_list.append(direction_matrix_batch(t_mid, n_t, p)[0])
    A_list = torch.stack(A_list)                     # (m, W, 2, 2)

    wn = p["wn"]
    for h0 in range(0, H, chunk_rows):
        h1 = min(h0 + chunk_rows, H)
        Bh = h1 - h0
        ap_b = ap_t[h0:h1]
        kappa = (ap_b * p["Kt"] / p["mt"])[:, None]     # (Bh, 1)
        B = Bh * W
        rho = torch.zeros(B, device=device)
        v = torch.randn(B, dim, device=device, generator=g)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-30)
        for _ in range(iters):
            u = v[:, :4].clone()                          # (B,4)
            hist = v[:, 4:].reshape(B, m, 2)              # (B, m, 2)
            for i in range(m - 1, -1, -1):
                A = A_list[m - 1 - i]                     # (W, 2, 2)
                A = A[None, :, :, :].expand(Bh, W, 2, 2).reshape(B, 2, 2)
                kap = kappa.expand(Bh, W).reshape(B)
                L = torch.zeros((B, 4, 4), device=device)
                L[:, 0, 2] = 1.0
                L[:, 1, 3] = 1.0
                L[:, 2:4, :2] = -wn * wn * eye2 + kap[:, None, None] * A
                L[:, 2:4, 2:] = -2.0 * p["zeta"] * wn * eye2
                R = torch.zeros((B, 4, 2), device=device)
                R[:, 2:, :] = -kap[:, None, None] * A
                dt = dt_w[None, :].expand(Bh, W).reshape(B)
                Phi, Psi0, Psi1 = _interval_flow_linear_batch(L, R, dt)
                q_im = hist[:, m - 1, :]                  # q_{i-m}
                q_im1 = (hist[:, m - 2, :] if m >= 2 else u[:, :2])
                unew = (Phi @ u[:, :, None]).squeeze(-1) \
                    + (Psi1 @ q_im1[:, :, None]).squeeze(-1) \
                    + (Psi0 @ q_im[:, :, None]).squeeze(-1)
                new_hist = torch.empty_like(hist)
                new_hist[:, 0, :] = u[:, :2]
                if m > 1:
                    new_hist[:, 1:, :] = hist[:, :-1, :]
                hist = new_hist
                u = unew
            w = torch.cat([u, hist.reshape(B, m * 2)], dim=1)
            nrm = w.norm(dim=1).clamp_min(1e-300)
            rho_new = nrm
            if torch.allclose(rho_new, rho, rtol=tol, atol=1e-12):
                rho = rho_new
                break
            rho = rho_new
            v = w / nrm[:, None]
        out[h0:h1, :] = rho.reshape(Bh, W).cpu().numpy()
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def floquet_radii_batch(n_rpms: np.ndarray, a_ps: np.ndarray, p: dict,
                        m: int, device: str = "cuda",
                        chunk_rows: int = 16, use_power: bool = True,
                        power_iters: int = 150,
                        use_numpy_eig: bool = False,
                        nproc: int = 16) -> np.ndarray:
    """Return rho array of shape (len(a_ps), len(n_rpms))."""
    W = len(n_rpms)
    H = len(a_ps)
    n_t = torch.tensor(n_rpms, dtype=torch.float64, device=device)[None, :]
    ap_t = torch.tensor(a_ps, dtype=torch.float64, device=device)
    dim = 4 + 2 * m
    out = np.empty((H, W))
    tau_col = 60.0 / (p["N"] * n_rpms)          # (W,)
    for h0 in range(0, H, chunk_rows):
        h1 = min(h0 + chunk_rows, H)
        ap_b = ap_t[h0:h1]                        # (Bh,)
        kappa = ap_b * p["Kt"] / p["mt"]          # (Bh,)
        Bh = h1 - h0
        # batch dimension B = Bh * W
        # Build L/R for every (depth, speed) pair.
        # omega_s depends on speed; tau/dt depends on speed; A depends on t.
        D = torch.eye(dim, dtype=torch.float32, device=device)
        D = D.expand(Bh, W, dim, dim).clone()
        dt_mat = torch.zeros((Bh, W), dtype=torch.float64, device=device)
        dt_mat[:] = torch.tensor(tau_col / m, dtype=torch.float64,
                                 device=device)[None, :]
        eye2_f32 = torch.eye(2, dtype=torch.float32, device=device)
        eye2 = torch.eye(2, dtype=torch.float64, device=device)
        for i in range(m - 1, -1, -1):
            t_mid = (i + 0.5) * dt_mat[0:1, :]    # (1, W)
            A = direction_matrix_batch(t_mid, n_t, p)   # (1, W, 2, 2)
            A = A.expand(Bh, W, 2, 2).contiguous()
            wn = p["wn"]
            L = torch.zeros((Bh, W, 4, 4), dtype=torch.float64, device=device)
            L[..., 0, 2] = 1.0
            L[..., 1, 3] = 1.0
            L[..., 2:4, :2] = -wn * wn * eye2 + kappa[:, None, None, None] * A
            L[..., 2:4, 2:] = -2.0 * p["zeta"] * wn * eye2
            R = torch.zeros((Bh, W, 4, 2), dtype=torch.float64, device=device)
            R[..., 2:, :] = -kappa[:, None, None, None] * A
            Lf = L.reshape(Bh * W, 4, 4)
            Rf = R.reshape(Bh * W, 4, 2)
            dtf = dt_mat.reshape(Bh * W)
            Phi, Psi0, Psi1 = _interval_flow_linear_batch(Lf, Rf, dtf)
            Phi = Phi.reshape(Bh, W, 4, 4).to(torch.float32)
            Psi0 = Psi0.reshape(Bh, W, 4, 2).to(torch.float32)
            Psi1 = Psi1.reshape(Bh, W, 4, 2).to(torch.float32)
            Di = torch.zeros((Bh, W, dim, dim), dtype=torch.float64,
                             device=device).to(torch.float32)
            Di[..., :4, :4] = Phi
            if m == 1:
                Di[..., :4, :2] += Psi1
                Di[..., :4, 4:6] = Psi0
            else:
                Di[..., :4, 4 + 2 * (m - 2):4 + 2 * (m - 1)] = Psi1
                Di[..., :4, 4 + 2 * (m - 1):4 + 2 * m] = Psi0
            Di[..., 4:6, :2] = eye2_f32
            for k in range(1, m):
                Di[..., 4 + 2 * k:6 + 2 * k,
                   4 + 2 * (k - 1):6 + 2 * (k - 1)] = eye2_f32
            D = D @ Di
        Df = D.reshape(Bh * W, dim, dim)
        if use_power:
            rho_est = spectral_radius_median_long(
                Df, steps=power_iters, tail=min(1000, power_iters // 2)
                ).reshape(Bh, W)
            # correct boundary points exactly (they decide the mask)
            rho_c = correct_boundary_batch(
                Df, rho_est.reshape(Bh * W), band=0.03).reshape(Bh, W)
            rho_b = rho_c.cpu().numpy()
        elif use_numpy_eig:
            Dn = Df.cpu().numpy()
            with Pool(nproc) as pool:
                vals = pool.map(
                    _numpy_rho, [Dn[b] for b in range(Dn.shape[0])],
                    chunksize=32)
            rho_b = np.array(vals).reshape(Bh, W)
        else:
            vals = torch.linalg.eigvals(Df)
            rho_b = vals.abs().amax(dim=-1).reshape(Bh, W).cpu().numpy()
        out[h0:h1, :] = rho_b
        torch.cuda.empty_cache()
    return out


def _numpy_rho(D: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(D))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--m", type=int, default=80)
    ap.add_argument("--grid", type=str, default="fine",
                    choices=["fine", "c32", "c16"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--full-eig", action="store_true")
    ap.add_argument("--power-iters", type=int, default=150)
    ap.add_argument("--matrixfree", action="store_true")
    ap.add_argument("--numpy-eig", action="store_true")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--dump-D", action="store_true",
                    help="save transition matrices (D) for CPU exact eigvals")
    ap.add_argument("--aD", type=float, default=None)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--fn", type=float, default=None)
    args = ap.parse_args()

    if args.aD is not None:
        case = {"aD": args.aD, "zeta": args.zeta, "fn": args.fn}
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_unet_v2 import ALL_CASES
        case = ALL_CASES[args.case]
    p = milling_params(case["aD"], case["zeta"], case["fn"])
    if args.grid == "fine":
        n_rpms = np.linspace(4000.0, 16000.0, 80)
        a_ps = np.linspace(0.05e-3, 1.5e-3, 128)
    elif args.grid == "c32":
        n_rpms = np.linspace(4000.0, 16000.0, 20)
        a_ps = np.linspace(0.05e-3, 1.5e-3, 32)
    else:
        n_rpms = np.linspace(4000.0, 16000.0, 10)
        a_ps = np.linspace(0.05e-3, 1.5e-3, 16)
    t0 = time.perf_counter()
    if args.dump_D:
        chunk = 32 if args.m >= 160 else 128
        D = build_transition_batch_fast(n_rpms, a_ps, p, args.m,
                                        device=args.device,
                                        chunk_rows=chunk)
        dt = time.perf_counter() - t0
        print(f"case {args.case} grid={args.grid} m={args.m}: "
              f"D {D.shape} in {dt:.1f}s", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                        exist_ok=True)
            np.savez(args.out, D=D, n_rpms=n_rpms, a_p_mm=a_ps * 1e3,
                     case=case, m=args.m, elapsed_s=dt)
            print("saved", args.out)
        return
    elif args.matrixfree:
        rho = floquet_radii_matrixfree(n_rpms, a_ps, p, args.m,
                                       device=args.device,
                                       iters=args.power_iters)
    else:
        rho = floquet_radii_batch(n_rpms, a_ps, p, args.m, device=args.device,
                                  use_power=not args.full_eig,
                                  power_iters=args.power_iters,
                                  use_numpy_eig=args.numpy_eig,
                                  nproc=args.nproc)
    dt = time.perf_counter() - t0
    print(f"case {args.case} {case} grid={args.grid} m={args.m}: "
          f"{rho.shape} in {dt:.1f}s", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, rho=rho, n_rpms=n_rpms, a_p_mm=a_ps * 1e3,
                 case=case, m=args.m, elapsed_s=dt)
        print("saved", args.out)


if __name__ == "__main__":
    main()
