"""Wall-clock timings for the benchmark evaluation protocol (finding 4).

Measures per-case, per-method timings on the 10 test cases:
  - ZOA analytic boundary (128x80 grid)
  - bilinear16 / bilinear32 / nearest16 upsampling
  - U-Net reference inference (pure-NumPy implementation of the identical
    58,481-parameter architecture; the release predictions were produced by
    the same architecture in PyTorch, whose GPU inference is faster)

All timings are single-threaded CPU (BLAS threads pinned to 1) on the same
machine as the paper's other timings. Writes
benchmark/results/benchmark_timings.json. Reference costs for the fine SDM
label generation are cited from results/timings.json (76.5 s/map, 16
workers) and the benchmark generation log (~26 s/case, GPU assembly + CPU
LAPACK).

Run: python code/benchmark_timings.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from sdm_solver import MillingParams  # noqa: E402
from zoa_baseline import zoa_sld  # noqa: E402
from unet_surrogate import UNet  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "benchmark_timings.json")


def upsample(rho: np.ndarray, shape: tuple, order: int) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(rho, (shape[0] / rho.shape[0], shape[1] / rho.shape[1]),
                order=order)


def bilinear_resize_af0(x: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Bilinear resize with align_corners=False (torch convention)."""
    hin, win = x.shape
    hout, wout = out_shape
    sy = hin / hout
    sx = win / wout
    ys = (np.arange(hout) + 0.5) * sy - 0.5
    xs = (np.arange(wout) + 0.5) * sx - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, hin - 1)
    y1 = np.clip(y0 + 1, 0, hin - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, win - 1)
    x1 = np.clip(x0 + 1, 0, win - 1)
    wy = (ys - np.floor(ys))[:, None]
    wx = (xs - np.floor(xs))[None, :]
    out = (x[y0][:, x0] * (1 - wy) * (1 - wx)
           + x[y0][:, x1] * (1 - wy) * wx
           + x[y1][:, x0] * wy * (1 - wx)
           + x[y1][:, x1] * wy * wx)
    return out


def make_input_np(rho_c16: np.ndarray, case: dict) -> np.ndarray:
    field = bilinear_resize_af0((rho_c16 - 1.0).astype(np.float32),
                                (128, 80))
    pn = np.array([case["aD"], case["zeta"] * 100.0, case["fn"] / 1000.0],
                  dtype=np.float32)
    ch = np.full((3, 128, 80), pn[:, None, None], dtype=np.float32)
    return np.concatenate([field[None].astype(np.float32), ch], axis=0)


def median_time(fn, repeats: int):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    meta = load_meta()
    test_idx = list(split_indices(meta)["test"])
    net = UNet(in_ch=4, seed=0)
    w = np.load(os.path.join(ROOT, "benchmark", "results",
                             "unet_s0_both.npz"))
    for k in w.files:
        net.p[k] = w[k]

    rows = []
    for idx in test_idx:
        case = meta["cases"][idx]
        p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
        fine = load_case(idx, "fine", 80)
        c16 = load_case(idx, "c16", 20)["rho"]
        c32 = load_case(idx, "c32", 40)["rho"]
        n_fine = fine["n_rpms"]
        a_fine = fine["a_p_mm"] * 1e-3
        x = make_input_np(c16, case)

        t_zoa = median_time(lambda: zoa_sld(n_fine, a_fine, p), 2)
        t_b16 = median_time(
            lambda: upsample(c16, (128, 80), 1), 5)
        t_b32 = median_time(
            lambda: upsample(c32, (128, 80), 1), 5)
        t_n16 = median_time(
            lambda: upsample(c16, (128, 80), 0), 5)
        t_unet = median_time(lambda: net.forward(x[None]), 3)
        rows.append({
            "idx": idx,
            "case": case,
            "wall_clock_s": {
                "zoa": t_zoa,
                "bilinear16": t_b16,
                "bilinear32": t_b32,
                "nearest16": t_n16,
                "unet_both_s0": t_unet,
            },
        })
        print(f"case {idx:03d}: zoa {t_zoa:.3f}s unet {t_unet:.3f}s "
              f"b16 {t_b16*1e3:.1f}ms b32 {t_b32*1e3:.1f}ms "
              f"n16 {t_n16*1e3:.1f}ms")

    methods = ("zoa", "bilinear16", "bilinear32", "nearest16",
               "unet_both_s0")
    summary = {
        m: {
            "mean_s": float(np.mean([r["wall_clock_s"][m] for r in rows])),
            "median_s": float(np.median(
                [r["wall_clock_s"][m] for r in rows])),
            "min_s": float(np.min([r["wall_clock_s"][m] for r in rows])),
            "max_s": float(np.max([r["wall_clock_s"][m] for r in rows])),
        } for m in methods
    }
    result = {
        "description": "per-case wall-clock timings, 10 benchmark test "
                       "cases, single-thread CPU (Ryzen 9 7845HX, Windows, "
                       "BLAS threads=1); U-Net is the pure-NumPy "
                       "implementation of the release architecture "
                       "(PyTorch GPU inference is faster)",
        "n_repeats": {"zoa": 2, "bilinear16": 5, "bilinear32": 5,
                      "nearest16": 5, "unet_both_s0": 3},
        "summary": summary,
        "reference_costs": {
            "fine_sdm_m80_128x80_cpu_16workers_s":
                76.51158099999884,
            "coarse_sdm_c16_m20_cpu_16workers_s": 7.619981800002279,
            "source": "results/timings.json (21-case study, same machine)",
            "benchmark_label_generation_s_per_case_fine_m80": 26.0,
            "note": "GPU assembly + CPU LAPACK eigvals; see "
                    "docs/2026-08-13_阶段成果_基准数据生成.md",
        },
        "rows": rows,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("\nsummary (s/map):")
    for m in methods:
        s = summary[m]
        print(f"  {m:<14} mean {s['mean_s']:.4f}  "
              f"median {s['median_s']:.4f}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
