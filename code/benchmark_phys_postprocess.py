"""Physical post-processing: column-monotone projection of U-Net probs.

Direction-8 phase 1: the labels are nearly perfectly column-monotone
(stable mask non-increasing in axial depth; 0.33% violating columns over
all 90 cases) while the release U-Net violates the constraint in 9.6% of
columns. This script projects each column of the predicted probability
field to a non-increasing sequence (isotonic regression, decreasing=True)
and re-evaluates the 10 test cases, for each of the three seeds.

Run: python code/benchmark_phys_postprocess.py
Output: benchmark/results/phys_postprocess.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import isotonic_regression

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from benchmark_full_eval import make_input_np, sigmoid  # noqa: E402
from unet_surrogate import UNet, metrics  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "phys_postprocess.json")


def monotone_project(prob: np.ndarray) -> np.ndarray:
    """Per-column decreasing isotonic projection of P(stable)."""
    out = np.empty_like(prob)
    for c in range(prob.shape[1]):
        out[:, c] = isotonic_regression(prob[:, c],
                                        increasing=False).x
    return out


def col_violations(mask: np.ndarray) -> int:
    n = 0
    for c in range(mask.shape[1]):
        y = mask[:, c].astype(np.int8)
        deepest = np.maximum.accumulate(y[::-1])[::-1]
        if ((y == 0) & (deepest == 1)).any():
            n += 1
    return n


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    sidx = split_indices(meta)
    test_idx = list(sidx["test"])

    nets = {}
    for s in range(3):
        net = UNet(in_ch=4, seed=s)
        w = np.load(os.path.join(ROOT, "benchmark", "results",
                                 f"unet_s{s}_both.npz"))
        for k in w.files:
            net.p[k] = w[k]
        nets[s] = net

    ev = json.load(open(os.path.join(
        ROOT, "benchmark", "results", "benchmark_eval.json"),
        encoding="utf-8"))
    rows = []
    for idx in test_idx:
        case = meta["cases"][idx]
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        x = make_input_np(load_case(idx, "c16", 20)["rho"], case)
        per_seed = {}
        for s in range(3):
            logits = nets[s].forward(x[None])
            prob = sigmoid(logits)[0, 0]
            proj = monotone_project(prob)
            m_raw = metrics(y, (prob >= 0.5).astype(np.float32))
            m_proj = metrics(y, (proj >= 0.5).astype(np.float32))
            fs_proj = (m_proj["fp"] / (m_proj["tp"] + m_proj["fp"])
                       if (m_proj["tp"] + m_proj["fp"]) else 0.0)
            per_seed[s] = {
                "raw_f1": m_raw["f1"],
                "proj_f1": m_proj["f1"],
                "proj_false_stable": float(fs_proj),
                "raw_bad_cols": col_violations(
                    (prob >= 0.5).astype(np.float32)),
                "proj_bad_cols": col_violations(
                    (proj >= 0.5).astype(np.float32)),
            }
        row = {"idx": idx, "case": case,
               "release_f1_3seed": ev["unet"]["per_case"][str(idx)][
                   "f1_3seed_mean"],
               "proj_f1_3seed_mean": float(np.mean(
                   [per_seed[s]["proj_f1"] for s in range(3)])),
               "proj_false_stable_3seed_mean": float(np.mean(
                   [per_seed[s]["proj_false_stable"] for s in range(3)])),
               "per_seed": per_seed}
        rows.append(row)

    def mean_all(key):
        return float(np.mean([r[key] for r in rows]))

    def mean_nontriv(key):
        nontriv = [r for r in rows if r["case"] not in (
            {"aD": 0.05, "zeta": 0.011, "fn": 922.0},
            {"aD": 0.05, "zeta": 0.015, "fn": 850.0},
            {"aD": 0.05, "zeta": 0.02, "fn": 1000.0})]
        return float(np.mean([r[key] for r in nontriv]))

    result = {
        "description": "column-monotone (decreasing) isotonic projection of "
                       "U-Net probability fields; test split",
        "all_10": {
            "release_f1": mean_all("release_f1_3seed"),
            "proj_f1": mean_all("proj_f1_3seed_mean"),
            "proj_false_stable": mean_all(
                "proj_false_stable_3seed_mean"),
        },
        "non_trivial_7": {
            "release_f1": mean_nontriv("release_f1_3seed"),
            "proj_f1": mean_nontriv("proj_f1_3seed_mean"),
        },
        "total_bad_cols_raw": int(sum(
            r["per_seed"][0]["raw_bad_cols"] for r in rows)),
        "total_bad_cols_proj": int(sum(
            r["per_seed"][0]["proj_bad_cols"] for r in rows)),
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    a = result["all_10"]
    n = result["non_trivial_7"]
    print(f"all-10: release F1 {a['release_f1']:.4f} -> proj F1 "
          f"{a['proj_f1']:.4f} (fs {a['proj_false_stable']:.4f})")
    print(f"non-trivial-7: release F1 {n['release_f1']:.4f} -> proj F1 "
          f"{n['proj_f1']:.4f}")
    print(f"bad columns: raw {result['total_bad_cols_raw']} -> "
          f"proj {result['total_bad_cols_proj']}")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
