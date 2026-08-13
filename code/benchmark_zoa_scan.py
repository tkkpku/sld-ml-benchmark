"""Repository-side ZOA failure-region scan over all 90 benchmark cases.

Same conventions as the standard evaluation protocol (eval_benchmark.py):
128x80 fine grid, rho<1 labels, zoa_sld, false_stable = fp/(tp+fp).
Runs all 90 cases in parallel (8 workers, ~50 s) and writes per-case rows
plus parameter-band statistics to benchmark/results/zoa_region_scan.json.

This is the primary evidence for the Section 7 claim that the analytic ZOA
boundary is weakest for high immersion / low damping / low natural
frequency, and strongest for low immersion - the opposite of the earlier
speculation that low immersion is a ZOA-failure region.

Run: python code/benchmark_zoa_scan.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from sdm_solver import MillingParams  # noqa: E402
from unet_surrogate import metrics  # noqa: E402
from zoa_baseline import zoa_sld  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "zoa_region_scan.json")


def eval_one(args) -> dict:
    idx, case = args
    p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
    fine = load_case(idx, "fine", 80)
    y = (fine["rho"] < 1.0).astype(np.float32)
    n_fine = fine["n_rpms"]
    a_fine = fine["a_p_mm"] * 1e-3
    zoa = zoa_sld(n_fine, a_fine, p).astype(np.float32)
    m = metrics(y, zoa)
    fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
    return {
        "idx": idx,
        "case": case,
        "f1": float(m["f1"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "false_stable": float(fs),
        "stable_frac": float(y.mean()),
    }


def band_stats(rows, key, bands) -> list[dict]:
    out = []
    for lo, hi, label in bands:
        rs = [r for r in rows if lo <= r["case"][key] < hi]
        out.append({
            "band": label,
            "n": len(rs),
            "mean_f1": float(np.mean([r["f1"] for r in rs])),
            "min_f1": float(np.min([r["f1"] for r in rs])),
            "mean_false_stable": float(
                np.mean([r["false_stable"] for r in rs])),
            "max_false_stable": float(
                np.max([r["false_stable"] for r in rs])),
        })
    return out


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    cases = meta["cases"]
    split_map = split_indices(meta)
    split_of = {}
    for name, idxs in split_map.items():
        for i in idxs:
            split_of[i] = name

    with mp.Pool(8) as pool:
        rows = pool.map(eval_one, list(enumerate(cases)))

    for r in rows:
        r["split"] = split_of[r["idx"]]
    rows.sort(key=lambda r: r["f1"])

    result = {
        "description": "ZOA analytic boundary vs fine m80 labels over all "
                       "90 benchmark cases (eval_benchmark.py conventions)",
        "n_cases": len(rows),
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
        "overall": {
            "mean_f1": float(np.mean([r["f1"] for r in rows])),
            "min_f1": float(np.min([r["f1"] for r in rows])),
            "mean_false_stable": float(
                np.mean([r["false_stable"] for r in rows])),
        },
        "by_split": {
            name: {
                "n": len([r for r in rows if r["split"] == name]),
                "mean_f1": float(np.mean(
                    [r["f1"] for r in rows if r["split"] == name])),
            } for name in ("train", "val", "test")
        },
        "bands": {
            "aD": band_stats(rows, "aD", (
                (0.0, 0.15, "low<0.15"),
                (0.15, 0.5, "mid 0.15-0.5"),
                (0.5, 1.01, "high>=0.5"),
            )),
            "zeta": band_stats(rows, "zeta", (
                (0.0, 0.008, "low<0.008"),
                (0.008, 0.02, "mid 0.008-0.02"),
                (0.02, 0.04, "high>=0.02"),
            )),
            "fn": band_stats(rows, "fn", (
                (650.0, 850.0, "low<850"),
                (850.0, 1000.0, "mid 850-1000"),
                (1000.0, 1200.0, "high>=1000"),
            )),
        },
        "worst_15": rows[:15],
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"elapsed {result['elapsed_s']:.1f}s, {len(rows)} cases")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
