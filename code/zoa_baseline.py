"""Zero-order analytical (ZOA) stability boundary for the 2-DOF milling model.

The averaged characteristic equation is
    det( I - a_p K_t (1 - e^{-i w tau}) A0 G(i w) ) = 0,
with G(i w) = 1/(k - m w^2 + i c w) per DOF and tau = 60/(N n_rpm).

A0*G(i w) has complex-conjugate eigenvalue pairs, so the classical
single-DOF tangent formula does not apply.  For a fixed spindle speed n,
the boundary depth is found by scanning the chatter frequency w for the
roots of
    Im[ (1 - e^{-i w tau}) lambda_j(w) ] = 0,
with Re[ (1 - e^{-i w tau}) lambda_j(w) ] > 0, and taking
    a_lim = 1 / ( K_t Re[ (1 - e^{-i w tau}) lambda_j(w) ] ).
Because G(i w) = frf(w) I, lambda_j(w) = frf(w) mu_j with the constant
eigenvalues mu_j of A0; each candidate root is refined by bisection.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_solver import MillingParams, direction_coeffs


def average_direction_matrix(p: MillingParams, n_phi: int = 400) -> np.ndarray:
    """A0 = (N/2pi) int_0^{2pi} A_single(phi) dphi (zero-order average)."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    acc = np.zeros((2, 2))
    for ph in phi:
        acc += direction_coeffs(ph, p)
    return p.N * acc * (2.0 * np.pi / n_phi) / (2.0 * np.pi)


def frf(w: float, p: MillingParams) -> complex:
    k = p.mt * p.wn ** 2
    c = 2.0 * p.zeta * p.wn * p.mt
    return 1.0 / (k - p.mt * w * w + 1j * c * w)


def zoa_limit_depth(n_rpm: float, p: MillingParams, n_w: int = 8000,
                    w_lo: float | None = None, w_hi: float | None = None,
                    bisect_iters: int = 80) -> float | None:
    """Smallest positive ZOA a_lim at spindle speed n_rpm (exact roots)."""
    A0 = average_direction_matrix(p)
    mu = np.linalg.eigvals(A0)
    tau = 60.0 / (p.N * n_rpm)
    w_lo = 0.15 * p.wn if w_lo is None else w_lo
    w_hi = 2.0 * p.wn if w_hi is None else w_hi
    ws = np.linspace(w_lo, w_hi, n_w)
    eiw = np.exp(-1j * ws * tau)
    frfs = np.array([frf(w, p) for w in ws], dtype=complex)
    best = None
    for muj in mu:
        z = (1.0 - eiw) * frfs * muj
        re, im = z.real, z.imag
        s = np.sign(im)
        cross = np.where((s[:-1] != 0) & (s[1:] != 0) & (s[:-1] != s[1:]))[0]
        for i in cross:
            # root exists with Re < 0 only if one endpoint has Re < 0
            if re[i] <= 0.0 and re[i + 1] <= 0.0:
                continue
            wl, wr = ws[i], ws[i + 1]
            sl = s[i]
            for _ in range(bisect_iters):
                wm = 0.5 * (wl + wr)
                zm = (1.0 - np.exp(-1j * wm * tau)) * frf(wm, p) * muj
                if sl * np.sign(zm.imag) < 0:
                    wr = wm
                else:
                    wl = wm
            zm = (1.0 - np.exp(-1j * wr * tau)) * frf(wr, p) * muj
            if zm.real > 0.0:
                alim = 1.0 / (p.Kt * zm.real)
                if best is None or alim < best[0]:
                    best = (alim, wr, zm)
    return best[0] if best is not None else None


def zoa_curve(p: MillingParams, n_rpms: np.ndarray | None = None,
              n_w: int = 8000) -> list[tuple[float, float]]:
    """ZOA lobe points (n_rpm, a_lim) on the requested speed grid."""
    if n_rpms is None:
        n_rpms = np.linspace(4000.0, 16000.0, 120)
    pts = []
    for n in n_rpms:
        lim = zoa_limit_depth(n, p, n_w=n_w)
        if lim is not None:
            pts.append((float(n), lim))
    return pts


def zoa_sld(n_rpms: np.ndarray, a_ps: np.ndarray, p: MillingParams) -> np.ndarray:
    """Binary SLD (1 stable) on the fine grid by the exact ZOA roots."""
    out = np.empty((len(a_ps), len(n_rpms)), dtype=np.float32)
    for j, n in enumerate(n_rpms):
        lim = zoa_limit_depth(n, p)
        if lim is None:
            out[:, j] = 1.0
        else:
            out[:, j] = (a_ps < lim).astype(np.float32)
    return out


if __name__ == "__main__":
    p = MillingParams()
    for n in (5216.0, 5000.0, 8000.0, 12000.0):
        lim = zoa_limit_depth(n, p)
        print(f"n={n:.0f} rpm: ZOA a_lim = {lim * 1e3:.4f} mm")
    # quick comparison with fine SDM on the main case (data must be regenerated)
    fine_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "cases", "case_0_fine.npy")
    if os.path.exists(fine_path):
        fine = np.load(fine_path)
        n_rpms = np.linspace(4000.0, 16000.0, 80)
        a_ps = np.linspace(0.05e-3, 1.5e-3, 128)
        zoa = zoa_sld(n_rpms, a_ps, p)
        from unet_surrogate import metrics
        m = metrics((fine < 1).astype(np.float32), zoa)
        print("ZOA vs fine SDM: F1=%.4f prec=%.4f rec=%.4f" %
              (m["f1"], m["precision"], m["recall"]))
