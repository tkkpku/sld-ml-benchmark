"""90-case bilinear32 same-safety control by stable-region erosion.

Sixth-review finding 1: Section 8.8 compared the conditioned surrogate
against the 21-case-era interpolation numbers (F1 0.23-0.25). This script
reproduces the reviewer's on-the-spot control on the 90-case benchmark's 10
test cases: bilinear32 (c32 rho bilinearly upsampled, rho<1), conservative
operation = binary erosion of the predicted stable region by k pixels;
budget rule = smallest k whose 10-case mean false-stable fraction is at
most the budget.

Run: python code/benchmark_b32_safety_shift.py
Output: benchmark/results/b32_safety_shift.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from unet_surrogate import metrics  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "b32_safety_shift.json")
BUDGETS = (0.08, 0.045, 0.02, 0.011, 0.01)


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    test_idx = list(split_indices(meta)["test"])
    masks = {}
    for idx in test_idx:
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        c32 = load_case(idx, "c32", 40)["rho"]
        from scipy.ndimage import zoom
        b32 = (zoom(c32, (128 / 32, 80 / 20), order=1) < 1.0).astype(
            np.float32)
        masks[idx] = (y, b32)

    shifts = {}
    for k in range(0, 11):
        rows = []
        for idx, (y, b32) in masks.items():
            pred = b32
            if k > 0:
                pred = ndimage.binary_erosion(b32.astype(np.uint8),
                                              iterations=k).astype(
                    np.float32)
            m = metrics(y, pred)
            fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
            rows.append({"idx": idx, "f1": m["f1"], "false_stable": float(fs)})
        shifts[str(k)] = {
            "mean_f1": float(np.mean([r["f1"] for r in rows])),
            "mean_false_stable": float(np.mean(
                [r["false_stable"] for r in rows])),
            "worst_false_stable": float(np.max(
                [r["false_stable"] for r in rows])),
            "per_case": rows,
        }

    chosen = {}
    for b in BUDGETS:
        k = next((kk for kk in range(11)
                  if shifts[str(kk)]["mean_false_stable"] <= b), None)
        chosen[str(b)] = {"shift": k}
        if k is not None:
            chosen[str(b)].update(shifts[str(k)])

    result = {
        "description": "bilinear32 stable-region erosion to same-safety "
                       "budgets on the 10 benchmark test cases "
                       "(sixth-review control)",
        "budgets": list(BUDGETS),
        "shifts": shifts,
        "chosen": chosen,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for b in BUDGETS:
        c = chosen[str(b)]
        if c["shift"] is None:
            print(f"budget {b}: no shift within 0..10")
        else:
            print(f"budget {b}: shift={c['shift']} "
                  f"F1={c['mean_f1']:.4f} fs={c['mean_false_stable']:.4f} "
                  f"worst={c['worst_false_stable']:.4f}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
