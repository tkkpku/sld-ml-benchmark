"""Full 90-case evaluation: U-Net (3 seeds) + interpolation baselines vs ZOA.

Direction-2 first pass: compare the learned surrogate and the classical
baselines on every benchmark case, grouped by parameter bands and by
explicit failure-zone definitions (high immersion, low damping, low natural
frequency), to locate any region where the U-Net beats the analytic ZOA
boundary.

U-Net inference uses the pure-NumPy implementation of the release
architecture with the release weights (unet_s*_both.npz), the same input
construction as training (bilinear, align_corners=False) and threshold 0.5.
ZOA per-case scores are read from zoa_region_scan.json (same conventions).

Run: python code/benchmark_full_eval.py
Output: benchmark/results/full_eval.json
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
from unet_surrogate import UNet, metrics  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "full_eval.json")


def upsample(rho: np.ndarray, shape: tuple, order: int) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(rho, (shape[0] / rho.shape[0], shape[1] / rho.shape[1]),
                order=order)


def bilinear_resize_af0(x: np.ndarray, out_shape: tuple) -> np.ndarray:
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    from scipy.special import expit
    return expit(x).astype(np.float32)


def band_stats(rows, key, bands, methods):
    out = []
    for lo, hi, label in bands:
        rs = [r for r in rows if lo <= r["case"][key] < hi]
        out.append({
            "band": label,
            "n": len(rs),
            **{m: float(np.mean([r[m]["f1"] for r in rs]))
               for m in methods},
            **{f"{m}_wins_over_zoa": int(sum(
                r[m]["f1"] > r["zoa"]["f1"] for r in rs))
               for m in methods if m != "zoa"},
        })
    return out


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    split_of = {i: name for name, idxs in split_indices(meta).items()
                for i in idxs}
    zscan = json.load(open(os.path.join(
        ROOT, "benchmark", "results", "zoa_region_scan.json"),
        encoding="utf-8"))
    zoa_by_idx = {r["idx"]: r for r in zscan["rows"]}

    nets = {}
    for s in range(3):
        net = UNet(in_ch=4, seed=s)
        w = np.load(os.path.join(ROOT, "benchmark", "results",
                                 f"unet_s{s}_both.npz"))
        for k in w.files:
            net.p[k] = w[k]
        nets[s] = net

    rows = []
    for idx, case in enumerate(meta["cases"]):
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        c16 = load_case(idx, "c16", 20)["rho"]
        c32 = load_case(idx, "c32", 40)["rho"]
        x = make_input_np(c16, case)

        row = {"idx": idx, "case": case, "split": split_of[idx],
               "stable_frac": float(y.mean())}
        for name, pred in (
            ("bilinear16", (upsample(c16, y.shape, 1) < 1.0).astype(np.float32)),
            ("bilinear32", (upsample(c32, y.shape, 1) < 1.0).astype(np.float32)),
            ("nearest16", (upsample(c16, y.shape, 0) < 1.0).astype(np.float32)),
        ):
            m = metrics(y, pred)
            fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
            row[name] = {"f1": m["f1"], "false_stable": float(fs)}
        f1s, fss = [], []
        for s in range(3):
            logits = nets[s].forward(x[None])
            pred = (sigmoid(logits) > 0.5).astype(np.float32)[0]
            m = metrics(y, pred)
            fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
            f1s.append(m["f1"])
            fss.append(float(fs))
        row["unet_both"] = {"f1": float(np.mean(f1s)),
                            "f1_std": float(np.std(f1s)),
                            "false_stable": float(np.mean(fss))}
        row["zoa"] = {"f1": zoa_by_idx[idx]["f1"],
                      "false_stable": zoa_by_idx[idx]["false_stable"]}
        rows.append(row)

    methods = ("unet_both", "bilinear16", "bilinear32", "nearest16")
    result = {
        "description": "full 90-case head-to-head: U-Net (3 seeds, t=0.5, "
                       "numpy inference, release weights) and interpolation "
                       "baselines vs the ZOA analytic boundary "
                       "(zoa_region_scan.json)",
        "bands": {
            "aD": band_stats(rows, "aD", (
                (0.0, 0.15, "low<0.15"),
                (0.15, 0.5, "mid 0.15-0.5"),
                (0.5, 1.01, "high>=0.5"),
            ), methods),
            "zeta": band_stats(rows, "zeta", (
                (0.0, 0.008, "low<0.008"),
                (0.008, 0.02, "mid 0.008-0.02"),
                (0.02, 0.04, "high>=0.02"),
            ), methods),
            "fn": band_stats(rows, "fn", (
                (650.0, 850.0, "low<850"),
                (850.0, 1000.0, "mid 850-1000"),
                (1000.0, 1200.0, "high>=1000"),
            ), methods),
        },
        "failure_zones": {
            zone: {
                "n": len(rs),
                **{m: float(np.mean([r[m]["f1"] for r in rs]))
                   for m in methods + ("zoa",)},
                "unet_wins_over_zoa": int(sum(
                    r["unet_both"]["f1"] > r["zoa"]["f1"] for r in rs)),
                "bilinear32_wins_over_zoa": int(sum(
                    r["bilinear32"]["f1"] > r["zoa"]["f1"] for r in rs)),
                "cases": [r["idx"] for r in rs],
            }
            for zone, rs in (
                ("high_immersion_low_damping",
                 [r for r in rows if r["case"]["aD"] >= 0.5
                  and r["case"]["zeta"] <= 0.008]),
                ("low_fn_le800",
                 [r for r in rows if r["case"]["fn"] <= 800.0]),
                ("high_immersion_low_fn",
                 [r for r in rows if r["case"]["aD"] >= 0.5
                  and r["case"]["fn"] <= 800.0]),
                ("two_weak_factors",
                 [r for r in rows if sum((
                     r["case"]["aD"] >= 0.5,
                     r["case"]["zeta"] <= 0.008,
                     r["case"]["fn"] <= 800.0)) >= 2]),
                ("zoa_f1_below_0p93",
                 [r for r in rows if r["zoa"]["f1"] < 0.93]),
            )
        },
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print(f"elapsed {result['elapsed_s']:.1f}s")
    print("\nfailure zones (mean F1):")
    for zone, v in result["failure_zones"].items():
        print(f"  {zone:<32} n={v['n']:>2} "
              f"unet={v['unet_both']:.3f} b32={v['bilinear32']:.3f} "
              f"zoa={v['zoa']:.3f}  unet>zoa={v['unet_wins_over_zoa']}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
