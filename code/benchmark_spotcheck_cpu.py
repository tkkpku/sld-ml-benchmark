"""Label spot-check: stored benchmark labels vs the audited CPU solver.

For a deterministic set of cases and grid points, recompute the spectral
radius with the audited first-order SDM solver (code/sdm_solver.py, m=80)
and compare against the stored labels in benchmark/data/rho. This is the
primary evidence for the paper's claim that the GPU assembly pipeline is
numerically identical to the audited CPU solver.

Run (Windows): python code/benchmark_spotcheck_cpu.py
Output: benchmark/results/label_spotcheck.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from sdm_solver import MillingParams, floquet_radius  # noqa: E402

# deterministic spread: train anchors, all three original held-out cases,
# additional extrapolation cases, and the last case
CASE_IDS = [0, 6, 7, 8, 20, 21, 36, 37, 89]
# (row, col) in the 128x80 grid: corners, center and two interior points
POINTS = [(0, 0), (32, 20), (64, 40), (96, 60), (127, 79)]


def main() -> int:
    meta = json.load(open(os.path.join(ROOT, "benchmark", "meta.json"),
                          encoding="utf-8"))
    t0 = time.perf_counter()
    rows = []
    max_abs = 0.0
    n_checked = 0
    for idx in CASE_IDS:
        case = meta["cases"][idx]
        p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
        path = os.path.join(ROOT, "benchmark", "data", "rho",
                            f"case_{idx:03d}_fine_m80.npz")
        d = np.load(path, allow_pickle=True)
        rho_label = d["rho"]
        n_rpms = d["n_rpms"]
        a_p_mm = d["a_p_mm"]
        for (r, c) in POINTS:
            n_rpm = float(n_rpms[c])
            a_p = float(a_p_mm[r]) * 1e-3  # mm -> m (solver SI convention)
            rho_cpu = floquet_radius(n_rpm, a_p, p, m=80)
            diff = abs(rho_cpu - float(rho_label[r * 80 + c]))
            max_abs = max(max_abs, diff)
            n_checked += 1
            rows.append({
                "case_idx": idx,
                "case": case,
                "row": r,
                "col": c,
                "n_rpm": n_rpm,
                "a_p_mm": a_p,
                "rho_label": float(rho_label[r * 80 + c]),
                "rho_cpu": rho_cpu,
                "abs_diff": diff,
            })
        print(f"case {idx:03d} done ({time.perf_counter()-t0:.1f}s)")
    result = {
        "description": "stored fine m80 labels vs audited CPU SDM solver "
                       "(m=80, floquet_radius), deterministic grid points",
        "cases": CASE_IDS,
        "points_per_case": POINTS,
        "n_cases": len(CASE_IDS),
        "n_points": n_checked,
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.mean([r["abs_diff"] for r in rows])),
        "p99_abs_diff": float(np.percentile(
            [r["abs_diff"] for r in rows], 99)),
        "p50_abs_diff": float(np.median(
            [r["abs_diff"] for r in rows])),
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    out = os.path.join(ROOT, "benchmark", "results",
                       "label_spotcheck.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"max |rho_label - rho_cpu| = {max_abs:.3e} "
          f"({n_checked} points, {len(CASE_IDS)} cases)")
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
