"""Physical consistency and error-mechanism analysis (direction 8, phase 1).

Two questions that the paper currently answers only with numbers, not with
mechanism:

1. Column monotonicity: for fixed spindle speed, the stable mask should be
   non-increasing in axial depth (shallow = stable, deeper = unstable).
   How often do the labels, the U-Net, the interpolations and ZOA violate
   this physical constraint?
2. Error mechanism: where do the U-Net's false-stable pixels lie relative
   to the true boundary, and how do U-Net errors overlap with ZOA errors
   (complementary vs correlated)?

Run: python code/benchmark_physics_analysis.py
Output: benchmark/results/physics_analysis.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from benchmark_full_eval import make_input_np, sigmoid  # noqa: E402
from sdm_solver import MillingParams  # noqa: E402
from unet_surrogate import UNet, metrics  # noqa: E402
from zoa_baseline import zoa_sld  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "physics_analysis.json")


def zoa_mask_worker(args) -> tuple[int, np.ndarray]:
    idx, case = args
    p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
    fine = load_case(idx, "fine", 80)
    zoa = zoa_sld(fine["n_rpms"], fine["a_p_mm"] * 1e-3,
                  p).astype(np.float32)
    return idx, zoa


def col_violations(mask: np.ndarray) -> tuple[int, int]:
    """mask: (128, 80), 1 = stable. Count columns with a 0-above-1 pattern
    and the number of violating pixels (0 that has a 1 deeper)."""
    n_bad_cols = 0
    n_bad_px = 0
    for c in range(mask.shape[1]):
        y = mask[:, c].astype(np.int8)
        # cumulative max from bottom: deepest stable depth seen
        deepest = np.maximum.accumulate(y[::-1])[::-1]
        bad = (y == 0) & (deepest == 1)
        if bad.any():
            n_bad_cols += 1
            n_bad_px += int(bad.sum())
    return n_bad_cols, n_bad_px


def prob_violations(prob: np.ndarray, tol: float = 1e-3) -> int:
    """Count column pairs where P(stable) increases with depth by > tol."""
    n = 0
    for c in range(prob.shape[1]):
        d = np.diff(prob[:, c])
        n += int((d > tol).sum())
    return n


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    sidx = split_indices(meta)
    test_idx = set(sidx["test"])

    net = UNet(in_ch=4, seed=0)
    w = np.load(os.path.join(ROOT, "benchmark", "results",
                             "unet_s0_both.npz"))
    for k in w.files:
        net.p[k] = w[k]

    with mp.Pool(8) as pool:
        zoa_masks = dict(pool.map(
            zoa_mask_worker, list(enumerate(meta["cases"]))))

    rows = []
    for idx, case in enumerate(meta["cases"]):
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        zoa = zoa_masks[idx]
        c16 = load_case(idx, "c16", 20)["rho"]
        logits = net.forward(make_input_np(c16, case)[None])
        prob = sigmoid(logits)[0, 0]
        unet = (prob >= 0.5).astype(np.float32)

        from scipy.ndimage import zoom
        b32 = (zoom(load_case(idx, "c32", 40)["rho"],
                    (128 / 32, 80 / 20), order=1) < 1.0).astype(np.float32)

        row = {"idx": idx, "case": case,
               "in_test": idx in test_idx,
               "label": {"bad_cols": 0, "bad_px": 0},
               "unet": {"bad_cols": 0, "bad_px": 0,
                        "prob_ascending_pairs": 0},
               "zoa": {"bad_cols": 0, "bad_px": 0},
               "b32": {"bad_cols": 0, "bad_px": 0}}
        row["label"]["bad_cols"], row["label"]["bad_px"] = col_violations(y)
        row["unet"]["bad_cols"], row["unet"]["bad_px"] = col_violations(unet)
        row["unet"]["prob_ascending_pairs"] = prob_violations(prob)
        row["zoa"]["bad_cols"], row["zoa"]["bad_px"] = col_violations(zoa)
        row["b32"]["bad_cols"], row["b32"]["bad_px"] = col_violations(b32)

        if idx in test_idx:
            # error mechanism on test cases
            u_fp = (unet == 1) & (y == 0)
            z_fp = (zoa == 1) & (y == 0)
            u_fn = (unet == 0) & (y == 1)
            z_fn = (zoa == 0) & (y == 1)
            # distance from each false-stable pixel to the true boundary
            yb = np.abs(np.gradient(y.astype(np.float32), axis=0)) > 0
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~yb)
            row["error_mechanism"] = {
                "u_fp": int(u_fp.sum()),
                "z_fp": int(z_fp.sum()),
                "u_fn": int(u_fn.sum()),
                "z_fn": int(z_fn.sum()),
                "fp_overlap": int((u_fp & z_fp).sum()),
                "u_only_fp": int((u_fp & ~z_fp).sum()),
                "z_only_fp": int((z_fp & ~u_fp).sum()),
                "fn_overlap": int((u_fn & z_fn).sum()),
                "u_only_fn": int((u_fn & ~z_fn).sum()),
                "z_only_fn": int((z_fn & ~u_fn).sum()),
                "u_fp_median_boundary_dist": float(
                    np.median(dist[u_fp])) if u_fp.any() else 0.0,
                "u_fp_frac_within_2px": float(
                    (dist[u_fp] <= 2.0).mean()) if u_fp.any() else 0.0,
                "z_fp_median_boundary_dist": float(
                    np.median(dist[z_fp])) if z_fp.any() else 0.0,
            }
        rows.append(row)

    label_cols = sum(r["label"]["bad_cols"] for r in rows)
    label_px = sum(r["label"]["bad_px"] for r in rows)
    total_cols = 90 * 80
    total_px = 90 * 128 * 80
    test_rows = [r for r in rows if r["in_test"]]
    result = {
        "description": "column monotonicity (stable mask non-increasing in "
                       "axial depth) over all 90 cases; error mechanism on "
                       "the 10 test cases",
        "label_violations": {
            "bad_cols": label_cols,
            "bad_cols_frac": label_cols / total_cols,
            "bad_px": label_px,
            "bad_px_frac": label_px / total_px,
        },
        "prediction_violations": {
            name: {
                "bad_cols": sum(r[name]["bad_cols"] for r in rows),
                "bad_px": sum(r[name]["bad_px"] for r in rows),
            } for name in ("unet", "zoa", "b32")
        },
        "unet_prob_ascending_pairs": sum(
            r["unet"]["prob_ascending_pairs"] for r in rows),
        "test_error_mechanism": {
            "u_fp": int(sum(r["error_mechanism"]["u_fp"]
                            for r in test_rows)),
            "z_fp": int(sum(r["error_mechanism"]["z_fp"]
                            for r in test_rows)),
            "fp_overlap": int(sum(r["error_mechanism"]["fp_overlap"]
                                  for r in test_rows)),
            "u_only_fp": int(sum(r["error_mechanism"]["u_only_fp"]
                                 for r in test_rows)),
            "z_only_fp": int(sum(r["error_mechanism"]["z_only_fp"]
                                 for r in test_rows)),
            "fn_overlap": int(sum(r["error_mechanism"]["fn_overlap"]
                                  for r in test_rows)),
            "u_only_fn": int(sum(r["error_mechanism"]["u_only_fn"]
                                 for r in test_rows)),
            "z_only_fn": int(sum(r["error_mechanism"]["z_only_fn"]
                                 for r in test_rows)),
            "u_fp_median_boundary_dist": float(np.mean([
                r["error_mechanism"]["u_fp_median_boundary_dist"]
                for r in test_rows if r["error_mechanism"]["u_fp"]])),
            "u_fp_frac_within_2px": float(np.mean([
                r["error_mechanism"]["u_fp_frac_within_2px"]
                for r in test_rows if r["error_mechanism"]["u_fp"]])),
        },
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"elapsed {result['elapsed_s']:.1f}s")
    print("label col violations:",
          result["label_violations"]["bad_cols"], "/", total_cols,
          f"({result['label_violations']['bad_cols_frac']:.4f})")
    for name in ("unet", "zoa", "b32"):
        v = result["prediction_violations"][name]
        print(f"{name}: bad_cols={v['bad_cols']} bad_px={v['bad_px']}")
    em = result["test_error_mechanism"]
    print("test: u_fp", em["u_fp"], "z_fp", em["z_fp"],
          "fp_overlap", em["fp_overlap"], "u_only_fp", em["u_only_fp"])
    print("test: u_fn", em["fn_overlap"] + em["u_only_fn"],
          "z_fn", em["fn_overlap"] + em["z_only_fn"],
          "fn_overlap", em["fn_overlap"])
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
