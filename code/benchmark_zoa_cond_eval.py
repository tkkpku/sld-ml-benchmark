"""Evaluation of the ZOA-conditioned U-Net (direction 8, phase 3).

Runs in WSL2 yolo_env (torch + scipy). Evaluates the three seeds of the
field_zoa model on the 10 test cases, with and without the column-monotone
isotonic projection, and reports per-case and summary metrics against the
release U-Net, bilinear32 and ZOA (read from benchmark_eval.json). Also
measures the end-to-end wall-clock time (coarse 16x10 ZOA + forward).

Run:
  /home/tan83/yolo_env/bin/python code/benchmark_zoa_cond_eval.py

Output: benchmark/results/zoa_cond_eval.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import isotonic_regression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from benchmark_train_torch import UNetTorch, metrics_t  # noqa: E402
from benchmark_train_zoa_cond import coarse_zoa, make_input  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results", "zoa_cond_eval.json")


def monotone_project(prob: np.ndarray) -> np.ndarray:
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


def in_zone(case: dict) -> bool:
    return sum((case["aD"] >= 0.5, case["zeta"] <= 0.008,
                case["fn"] <= 800.0)) >= 2


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    sidx = split_indices(meta)
    test_idx = list(sidx["test"])
    ev = json.load(open(os.path.join(
        ROOT, "benchmark", "results", "benchmark_eval.json"),
        encoding="utf-8"))

    nets = {}
    for s in range(3):
        net = UNetTorch(in_ch=2, seed=s).to(DEVICE)
        state = np.load(os.path.join(
            ROOT, "benchmark", "results", f"unet_zc_field_zoa_s{s}.npz"))
        net.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
        net.eval()
        nets[s] = net

    rows = []
    for idx in test_idx:
        case = meta["cases"][idx]
        fine = load_case(idx, "fine", 80)
        y = (fine["rho"] < 1.0).astype(np.float32)
        zc = coarse_zoa(case)
        x = make_input(load_case(idx, "c16", 20)["rho"], case, zc,
                       "field_zoa")
        xt = torch.tensor(x[None], device=DEVICE)
        per_seed = {}
        for s in range(3):
            with torch.no_grad():
                prob = torch.sigmoid(nets[s](xt)).cpu().numpy()[0, 0]
            proj = monotone_project(prob)
            m_raw = metrics_t(y, prob >= 0.5)
            m_proj = metrics_t(y, proj >= 0.5)
            fs_raw = (m_raw["fp"] / (m_raw["tp"] + m_raw["fp"])
                      if (m_raw["tp"] + m_raw["fp"]) else 0.0)
            fs_proj = (m_proj["fp"] / (m_proj["tp"] + m_proj["fp"])
                       if (m_proj["tp"] + m_proj["fp"]) else 0.0)
            per_seed[s] = {
                "raw_f1": m_raw["f1"],
                "raw_false_stable": float(fs_raw),
                "proj_f1": m_proj["f1"],
                "proj_false_stable": float(fs_proj),
                "raw_bad_cols": col_violations(
                    (prob >= 0.5).astype(np.float32)),
                "proj_bad_cols": col_violations(
                    (proj >= 0.5).astype(np.float32)),
            }
        b = ev["baselines"]["cases"][str(idx)]["baselines"]
        rows.append({
            "idx": idx,
            "case": case,
            "in_zone": in_zone(case),
            "release_unet_f1": ev["unet"]["per_case"][str(idx)][
                "f1_3seed_mean"],
            "bilinear32_f1": b["bilinear32"]["f1"],
            "zoa_f1": b["zoa"]["f1"],
            "zc_raw_f1_3seed": float(np.mean(
                [per_seed[s]["raw_f1"] for s in range(3)])),
            "zc_proj_f1_3seed": float(np.mean(
                [per_seed[s]["proj_f1"] for s in range(3)])),
            "zc_proj_false_stable_3seed": float(np.mean(
                [per_seed[s]["proj_false_stable"] for s in range(3)])),
            "zc_raw_false_stable_3seed": float(np.mean(
                [per_seed[s]["raw_false_stable"] for s in range(3)])),
            "raw_bad_cols_s0": per_seed[0]["raw_bad_cols"],
            "proj_bad_cols_s0": per_seed[0]["proj_bad_cols"],
            "per_seed": per_seed,
        })

    def mean(key, sel=None):
        rs = rows if sel is None else [r for r in rows if sel(r)]
        return float(np.mean([r[key] for r in rs]))

    zone_sel = lambda r: r["in_zone"]  # noqa: E731
    nontriv_sel = lambda r: r["case"] not in (  # noqa: E731
        {"aD": 0.05, "zeta": 0.011, "fn": 922.0},
        {"aD": 0.05, "zeta": 0.015, "fn": 850.0},
        {"aD": 0.05, "zeta": 0.02, "fn": 1000.0})

    # timing: coarse ZOA + one forward, median of 5 on one case
    case = meta["cases"][test_idx[0]]
    zc = coarse_zoa(case)
    x = make_input(load_case(test_idx[0], "c16", 20)["rho"], case, zc,
                   "field_zoa")
    xt = torch.tensor(x[None], device=DEVICE)
    ts = []
    for _ in range(5):
        t1 = time.perf_counter()
        coarse_zoa(case)
        with torch.no_grad():
            nets[0](xt)
        ts.append(time.perf_counter() - t1)
    timing = float(np.median(ts))

    result = {
        "description": "ZOA-conditioned U-Net (field_zoa, 3 seeds), raw and "
                       "column-monotone-projected, vs release U-Net / "
                       "bilinear32 / ZOA",
        "all_10": {
            "release_unet": mean("release_unet_f1"),
            "bilinear32": mean("bilinear32_f1"),
            "zoa": mean("zoa_f1"),
            "zc_raw": mean("zc_raw_f1_3seed"),
            "zc_proj": mean("zc_proj_f1_3seed"),
            "zc_proj_false_stable": mean(
                "zc_proj_false_stable_3seed"),
        },
        "non_trivial_7": {
            "release_unet": mean("release_unet_f1", nontriv_sel),
            "bilinear32": mean("bilinear32_f1", nontriv_sel),
            "zoa": mean("zoa_f1", nontriv_sel),
            "zc_raw": mean("zc_raw_f1_3seed", nontriv_sel),
            "zc_proj": mean("zc_proj_f1_3seed", nontriv_sel),
        },
        "failure_zone_5": {
            "release_unet": mean("release_unet_f1", zone_sel),
            "bilinear32": mean("bilinear32_f1", zone_sel),
            "zoa": mean("zoa_f1", zone_sel),
            "zc_raw": mean("zc_raw_f1_3seed", zone_sel),
            "zc_proj": mean("zc_proj_f1_3seed", zone_sel),
        },
        "monotonicity": {
            "raw_bad_cols_s0": int(sum(r["raw_bad_cols_s0"]
                                       for r in rows)),
            "proj_bad_cols_s0": int(sum(r["proj_bad_cols_s0"]
                                        for r in rows)),
        },
        "timing_s": {
            "coarse_zoa_plus_forward_median_s": timing,
            "fine_zoa_reference_s": 3.1872,
        },
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for key in ("all_10", "non_trivial_7", "failure_zone_5"):
        v = result[key]
        print(f"{key}: release={v['release_unet']:.4f} "
              f"b32={v['bilinear32']:.4f} zoa={v['zoa']:.4f} "
              f"zc_raw={v['zc_raw']:.4f} zc_proj={v['zc_proj']:.4f}")
    print("monotonicity bad cols:", result["monotonicity"])
    print(f"timing: {timing:.4f} s/map (coarse ZOA + forward)")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
