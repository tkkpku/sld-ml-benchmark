"""Stratified evaluation: all-10 vs non-trivial-7 and difficulty layers.

Difficulty layers by stable-pixel fraction of the fine m80 label:
  trivial  >= 0.99   (almost the whole map stable)
  medium    0.5-0.99
  hard     < 0.5

For every method (U-Net 3-seed mean, bilinear16/32, nearest16, ZOA) reports
mean F1 and mean false-stable over: all 10 test cases, the 7 non-trivial
cases, and each layer. Writes benchmark/results/stratified_eval.json.

This directly answers the fourth review's finding 2: the headline F1 must
not depend on trivial cases, and future submissions cannot inflate scores
with them.

Run: python code/benchmark_stratified.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "stratified_eval.json")

TRIVIAL = 0.99


def mean_rows(rows, method: str) -> dict:
    f1 = np.mean([r[method]["f1"] for r in rows])
    fs = np.mean([r[method]["false_stable"] for r in rows])
    return {"n": len(rows), "f1": float(f1), "false_stable": float(fs)}


def main() -> int:
    meta = load_meta()
    test_idx = split_indices(meta)["test"]
    ev = json.load(open(os.path.join(
        ROOT, "benchmark", "results", "benchmark_eval.json"),
        encoding="utf-8"))
    utm = json.load(open(os.path.join(
        ROOT, "benchmark", "results", "unet_test_metrics.json"),
        encoding="utf-8"))

    rows = []
    test_idx = list(test_idx)
    for idx in test_idx:
        case = meta["cases"][idx]
        fine = load_case(idx, "fine", 80)
        stable_frac = float((fine["rho"] < 1.0).mean())
        per = ev["unet"]["per_case"][str(idx)]
        # U-Net 3-seed mean fs from per-seed tp/fp
        fs_seeds = []
        pos = test_idx.index(idx)
        for s in range(3):
            pc = utm[f"both_s{s}"]["c16_t0.5"]["per_case"]
            p = pc[pos]  # per_case order follows test_idx order
            fs_seeds.append(p["fp"] / (p["tp"] + p["fp"])
                            if (p["tp"] + p["fp"]) else 0.0)
        b = ev["baselines"]["cases"][str(idx)]["baselines"]
        rows.append({
            "idx": idx,
            "case": case,
            "stable_frac": stable_frac,
            "layer": ("trivial" if stable_frac >= TRIVIAL else
                      ("medium" if stable_frac >= 0.5 else "hard")),
            "unet_both": {
                "f1": per["f1_3seed_mean"],
                "false_stable": float(np.mean(fs_seeds)),
            },
            "bilinear16": {"f1": b["bilinear16"]["f1"],
                           "false_stable": b["bilinear16"]["false_stable"]},
            "bilinear32": {"f1": b["bilinear32"]["f1"],
                           "false_stable": b["bilinear32"]["false_stable"]},
            "nearest16": {"f1": b["nearest16"]["f1"],
                          "false_stable": b["nearest16"]["false_stable"]},
            "zoa": {"f1": b["zoa"]["f1"],
                    "false_stable": b["zoa"]["false_stable"]},
        })

    methods = ("unet_both", "bilinear16", "bilinear32", "nearest16", "zoa")
    summary = {
        "trivial_threshold": TRIVIAL,
        "all_10": {m: mean_rows(rows, m) for m in methods},
        "non_trivial_7": {
            m: mean_rows([r for r in rows if r["layer"] != "trivial"], m)
            for m in methods
        },
        "by_layer": {
            layer: {
                m: mean_rows([r for r in rows if r["layer"] == layer], m)
                for m in methods
            } for layer in ("trivial", "medium", "hard")
        },
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(f"{'method':<12} {'all10 F1':>8} {'nontriv7 F1':>11} "
          f"{'triv':>7} {'med':>7} {'hard':>7}")
    for m in methods:
        a = summary["all_10"][m]
        n = summary["non_trivial_7"][m]
        t = summary["by_layer"]["trivial"][m]
        md = summary["by_layer"]["medium"][m]
        h = summary["by_layer"]["hard"][m]
        print(f"{m:<12} {a['f1']:>8.3f} {n['f1']:>11.3f} "
              f"{t['f1']:>7.3f} {md['f1']:>7.3f} {h['f1']:>7.3f}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
