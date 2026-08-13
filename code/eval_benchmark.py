"""SLD-ML Benchmark v1 standard evaluation protocol.

Computes per-case and summary metrics on the 10 test cases for:
  - bilinear / nearest upsampling of c16 and c32 rho maps
  - ZOA analytic boundary
  - U-Net (read from benchmark/results/unet_test_metrics.json)
Metrics: F1, precision, recall, false-stable fraction, boundary distance.
Also compares fine m80 vs fine m160 labels on the test+val cases that have
both (label uncertainty report).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices
from sdm_solver import MillingParams
from unet_surrogate import metrics, mean_boundary_distance
from zoa_baseline import zoa_sld

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results")


def upsample(rho: np.ndarray, shape: tuple, order: int) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(rho, (shape[0] / rho.shape[0], shape[1] / rho.shape[1]),
                order=order)


def eval_baselines() -> dict:
    meta = load_meta()
    test_idx = split_indices(meta)["test"]
    rows = {}
    for idx in test_idx:
        case = meta["cases"][idx]
        p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        c16 = load_case(idx, "c16", 20)["rho"]
        c32 = load_case(idx, "c32", 40)["rho"]
        n_fine = fine["n_rpms"]
        a_fine = fine["a_p_mm"] * 1e-3
        r = {}
        b16 = (upsample(c16, y.shape, 1) < 1.0).astype(np.float32)
        b32 = (upsample(c32, y.shape, 1) < 1.0).astype(np.float32)
        n16 = (upsample(c16, y.shape, 0) < 1.0).astype(np.float32)
        zoa = zoa_sld(n_fine, a_fine, p).astype(np.float32)
        for name, pred in (("bilinear16", b16), ("bilinear32", b32),
                           ("nearest16", n16), ("zoa", zoa)):
            m = metrics(y, pred)
            fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
            r[name] = {"f1": m["f1"], "precision": m["precision"],
                       "recall": m["recall"], "false_stable": float(fs),
                       "boundary_dist": float(mean_boundary_distance(y, pred))}
        rows[idx] = {"case": case, "baselines": r}
        print(f"case {idx} {case}: " +
              " ".join(f"{k}={v['f1']:.3f}" for k, v in r.items()))
    return {"protocol": "SLD-ML Benchmark v1",
            "split": "test", "cases": rows}


def label_uncertainty() -> dict:
    """m80 vs m160 stable-mask disagreement on test+val (where m160 exists)."""
    meta = load_meta()
    out = {}
    for split in ("test", "val"):
        for idx in split_indices(meta)[split]:
            path = os.path.join(ROOT, "benchmark", "data", "rho",
                                f"case_{idx:03d}_fine_m160.npz")
            if not os.path.exists(path):
                continue
            m80 = load_case(idx, "fine", 80)["rho"] < 1.0
            m160 = load_case(idx, "fine", 160)["rho"] < 1.0
            out[idx] = {"split": split,
                        "mask_disagreement": float(np.mean(m80 != m160))}
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    baselines = eval_baselines()
    lu = label_uncertainty()
    result = {"baselines": baselines, "label_uncertainty": lu,
              "elapsed_s": time.perf_counter() - t0}
    with open(os.path.join(OUT, "benchmark_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=float)
    # summary table
    names = ("bilinear16", "bilinear32", "nearest16", "zoa")
    print("\nsummary (mean over test cases):")
    for name in names:
        vals = [r["baselines"][name] for r in baselines["cases"].values()]
        print(f"  {name:12s} F1={np.mean([v['f1'] for v in vals]):.4f} "
              f"fs={np.mean([v['false_stable'] for v in vals]):.4f}")
    print("\nlabel m80 vs m160 mask disagreement (test+val):")
    d = [v["mask_disagreement"] for v in lu.values()]
    if d:
        print(f"  n={len(d)} mean={np.mean(d):.4f} max={np.max(d):.4f}")
    print("saved", os.path.join(OUT, "benchmark_eval.json"))


if __name__ == "__main__":
    main()
