"""Safety-budget operating points for the ZOA-conditioned surrogate.

Last dance (fifth review, direction 8): can the conditioned surrogate be
operated inside the analytic baseline's safety budget? Applies the exact
same validation-based threshold rule as Sections 6.3/8.5 (smallest
threshold in 0.50..0.95 whose validation mean false-stable fraction is at
most the budget; per seed; test reported once) to the projected probability
fields of the three field_zoa seeds, for three budgets:
  0.08  (the release protocol budget)
  0.02  (~3x ZOA's test false-stable)
  0.01  (ZOA's level)

Run (WSL2 yolo_env):
  /home/tan83/yolo_env/bin/python code/benchmark_zoa_cond_safety.py

Output: benchmark/results/zoa_cond_safety.json
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
OUT = os.path.join(ROOT, "benchmark", "results", "zoa_cond_safety.json")
THRESHOLDS = [0.50 + 0.05 * k for k in range(10)]  # 0.50..0.95
BUDGETS = (0.08, 0.02, 0.01)


def monotone_project(prob: np.ndarray) -> np.ndarray:
    out = np.empty_like(prob)
    for c in range(prob.shape[1]):
        out[:, c] = isotonic_regression(prob[:, c],
                                        increasing=False).x
    return out


def fs_of(y: np.ndarray, prob: np.ndarray, t: float) -> float:
    m = metrics_t(y, prob >= t)
    return m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0


def f1_of(y: np.ndarray, prob: np.ndarray, t: float) -> float:
    return metrics_t(y, prob >= t)["f1"]


def main() -> int:
    t0 = time.perf_counter()
    meta = load_meta()
    sidx = split_indices(meta)
    val_idx = list(sidx["val"])
    test_idx = list(sidx["test"])
    nontriv = lambda c: c not in (  # noqa: E731
        {"aD": 0.05, "zeta": 0.011, "fn": 922.0},
        {"aD": 0.05, "zeta": 0.015, "fn": 850.0},
        {"aD": 0.05, "zeta": 0.02, "fn": 1000.0})
    in_zone = lambda c: sum((c["aD"] >= 0.5, c["zeta"] <= 0.008,  # noqa: E731
                             c["fn"] <= 800.0)) >= 2

    nets = {}
    for s in range(3):
        net = UNetTorch(in_ch=2, seed=s).to(DEVICE)
        state = np.load(os.path.join(
            ROOT, "benchmark", "results", f"unet_zc_field_zoa_s{s}.npz"))
        net.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
        net.eval()
        nets[s] = net

    def probs_for(idx_list):
        out = {}
        for idx in idx_list:
            case = meta["cases"][idx]
            x = make_input(load_case(idx, "c16", 20)["rho"], case,
                           coarse_zoa(case), "field_zoa")
            xt = torch.tensor(x[None], device=DEVICE)
            ps = []
            for s in range(3):
                with torch.no_grad():
                    prob = torch.sigmoid(nets[s](xt)).cpu().numpy()[0, 0]
                ps.append(monotone_project(prob))
            out[idx] = ps
        return out

    val_probs = probs_for(val_idx)
    test_probs = probs_for(test_idx)
    val_y = {i: (load_case(i, "fine", 80)["rho"] < 1.0).astype(np.float32)
             for i in val_idx}
    test_y = {i: (load_case(i, "fine", 80)["rho"] < 1.0).astype(np.float32)
              for i in test_idx}

    seeds_out = []
    for s in range(3):
        val_fs = {t: float(np.mean([fs_of(val_y[i], val_probs[i][s], t)
                                    for i in val_idx]))
                  for t in THRESHOLDS}
        chosen = {}
        for b in BUDGETS:
            t_star = next((t for t in THRESHOLDS if val_fs[t] <= b), None)
            chosen[str(b)] = t_star
        test_rows = []
        for i in test_idx:
            p = test_probs[i][s]
            row = {"idx": i, "case": meta["cases"][i]}
            for b in BUDGETS:
                t_star = chosen[str(b)]
                if t_star is None:
                    row[str(b)] = None
                    continue
                row[str(b)] = {
                    "t": t_star,
                    "f1": f1_of(test_y[i], p, t_star),
                    "false_stable": fs_of(test_y[i], p, t_star),
                }
            row["0.5"] = {"t": 0.5, "f1": f1_of(test_y[i], p, 0.5),
                          "false_stable": fs_of(test_y[i], p, 0.5)}
            test_rows.append(row)

        def summ(key, sel=None):
            rs = test_rows if sel is None else [r for r in test_rows
                                                if sel(r["case"])]
            out = {}
            for b in ("0.5", "0.08", "0.02", "0.01"):
                vals = [r[b] for r in rs if r[b] is not None]
                if not vals:
                    out[b] = None
                else:
                    out[b] = {
                        "f1": float(np.mean([v["f1"] for v in vals])),
                        "false_stable": float(np.mean(
                            [v["false_stable"] for v in vals])),
                        "t": float(np.mean([v["t"] for v in vals])),
                    }
            return out

        seeds_out.append({
            "seed": s,
            "val_fs_curve": val_fs,
            "chosen": chosen,
            "all_10": summ("all"),
            "non_trivial_7": summ("nt", lambda c: nontriv(c)),
            "failure_zone_5": summ("fz", lambda c: in_zone(c)),
            "rows": test_rows,
        })

    result = {
        "description": "validation-selected safety operating points for the "
                       "ZOA-conditioned surrogate (projected probs), "
                       "budgets 0.08/0.02/0.01, rule of Sections 6.3/8.5",
        "thresholds": THRESHOLDS,
        "budgets": list(BUDGETS),
        "reference": {
            "zoa": {"f1": 0.965, "false_stable": 0.007},
            "release_unet_t0.5": {"f1": 0.905, "false_stable": 0.107},
            "release_unet_tstar_0.65": {"f1": 0.900,
                                        "false_stable": 0.075},
        },
        "seeds": seeds_out,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    def mean_across(group_key: str, budget: str):
        vals = [s[group_key][budget] for s in seeds_out
                if s[group_key][budget] is not None]
        if not vals:
            return None
        return {"f1": float(np.mean([v["f1"] for v in vals])),
                "false_stable": float(np.mean(
                    [v["false_stable"] for v in vals])),
                "t": float(np.mean([v["t"] for v in vals]))}

    for label, key in (("all-10", "all_10"),
                       ("non-trivial-7", "non_trivial_7"),
                       ("failure-zone-5", "failure_zone_5")):
        print(f"3-seed means ({label}):")
        for b in ("0.5", "0.08", "0.02", "0.01"):
            print(f"  t~{b}:", mean_across(key, b))
    print("chosen t* per seed:")
    for s in seeds_out:
        print("  seed", s["seed"], s["chosen"])
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
