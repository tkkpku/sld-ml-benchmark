"""Cheapest analytic baseline: coarse 16x10 ZOA + bilinear upsampling.

Direction-8 control requested by the fifth review: condition-channel gains
must be compared against the trivial non-learned use of the same analytic
prior (compute ZOA on the 16x10 grid and bilinearly upsample it). This is a
Windows-runnable, single-process script.

Run: python code/benchmark_coarse_zoa_baseline.py
Output: benchmark/results/coarse_zoa_baseline.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from sdm_solver import MillingParams  # noqa: E402
from unet_surrogate import metrics  # noqa: E402
from zoa_baseline import zoa_sld  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results",
                   "coarse_zoa_baseline.json")


def coarse_zoa(case: dict) -> np.ndarray:
    p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
    n = np.linspace(4000.0, 16000.0, 10)
    a = np.linspace(0.05e-3, 1.5e-3, 16)
    return zoa_sld(n, a, p).astype(np.float32)


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    test_idx = list(split_indices(meta)["test"])
    rows = []
    for idx in test_idx:
        case = meta["cases"][idx]
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        zc = coarse_zoa(case)
        pred = (zoom(zc, (128 / 16, 80 / 10), order=1) >= 0.5).astype(
            np.float32)
        m = metrics(y, pred)
        fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        rows.append({"idx": idx, "case": case,
                     "f1": m["f1"], "false_stable": float(fs)})

    def mean(key, sel=None):
        rs = rows if sel is None else [r for r in rows if sel(r)]
        return float(np.mean([r[key] for r in rs]))

    nontriv = lambda r: r["case"] not in (  # noqa: E731
        {"aD": 0.05, "zeta": 0.011, "fn": 922.0},
        {"aD": 0.05, "zeta": 0.015, "fn": 850.0},
        {"aD": 0.05, "zeta": 0.02, "fn": 1000.0})
    zone = lambda r: sum((r["case"]["aD"] >= 0.5,  # noqa: E731
                          r["case"]["zeta"] <= 0.008,
                          r["case"]["fn"] <= 800.0)) >= 2
    result = {
        "description": "coarse 16x10 ZOA boundary bilinearly upsampled to "
                       "128x80 and thresholded at 0.5 (no learning)",
        "all_10": {"f1": mean("f1"), "false_stable": mean("false_stable")},
        "non_trivial_7": {"f1": mean("f1", nontriv),
                          "false_stable": mean("false_stable", nontriv)},
        "failure_zone_5": {"f1": mean("f1", zone),
                           "false_stable": mean("false_stable", zone)},
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    a = result["all_10"]
    n = result["non_trivial_7"]
    z = result["failure_zone_5"]
    print(f"all-10 F1 {a['f1']:.4f} fs {a['false_stable']:.4f}; "
          f"non-trivial-7 F1 {n['f1']:.4f} fs {n['false_stable']:.4f}; "
          f"failure-zone-5 F1 {z['f1']:.4f} fs {z['false_stable']:.4f}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
