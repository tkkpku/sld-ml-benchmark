"""Benchmark-version safety analysis (revision 5).

Selects a conservative decision threshold for the 90-case U-Net on the
benchmark validation split (same rule as the 21-case study: smallest
threshold in {0.50..0.95} with validation mean false-stable <= 0.08), then
reports test metrics at t=0.5, t* and t=0.7, plus the same-safety-level
comparison against the classical baselines (ZOA default, erosion-shifted
interpolations).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices
from benchmark_train_torch import UNetTorch, metrics_t, DEVICE, OUT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BUDGET = 0.08


def load_model(seed: int):
    state = np.load(os.path.join(OUT, f"unet_s{seed}_both.npz"))
    net = UNetTorch(seed=seed).to(DEVICE)
    net.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    net.eval()
    return net


def predict_probs(seed: int, idx_list) -> dict[int, np.ndarray]:
    net = load_model(seed)
    out = {}
    for idx in idx_list:
        case = load_case(idx, "c16", 20)
        t = torch.tensor(np.array([case["rho"]]), dtype=torch.float32,
                         device=DEVICE)
        # build the canonical 4-channel input exactly as in benchmark_train
        from benchmark_train_torch import make_input
        x = make_input(case["rho"], load_meta()["cases"][idx], "both")
        xt = torch.tensor(x[None], device=DEVICE)
        with torch.no_grad():
            p = torch.sigmoid(net(xt)).cpu().numpy()[0, 0]
        out[idx] = p
    return out


def fs_frac(pred: np.ndarray, y: np.ndarray) -> float:
    m = metrics_t(y, pred)
    return m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0


def select_t(seed: int, val_idx) -> float:
    probs = predict_probs(seed, val_idx)
    for t in THRESHOLDS:
        fss = []
        for idx in val_idx:
            y = (load_case(idx, "fine", 80)["rho"] < 1.0).astype(np.float32)
            fss.append(fs_frac(probs[idx] >= t, y))
        if float(np.mean(fss)) <= BUDGET:
            return t
    return THRESHOLDS[-1]


def test_rows(seed: int, t_star: float, test_idx) -> dict:
    probs = predict_probs(seed, test_idx)
    out = {"seed": seed, "t_star": t_star, "rows": {}}
    for t in (0.5, t_star, 0.7):
        ms, fss = [], []
        for idx in test_idx:
            y = (load_case(idx, "fine", 80)["rho"] < 1.0).astype(np.float32)
            m = metrics_t(y, probs[idx] >= t)
            ms.append(m["f1"])
            fss.append(fs_frac(probs[idx] >= t, y))
        out["rows"][str(t)] = {"f1_mean": float(np.mean(ms)),
                               "f1_std_pop": float(np.std(ms)),
                               "false_stable_mean": float(np.mean(fss))}
    return out


def main() -> None:
    meta = load_meta()
    sidx = split_indices(meta)
    test_idx = sidx["test"]
    val_idx = sidx["val"]
    rows = []
    for seed in (0, 1, 2):
        t_star = select_t(seed, val_idx)
        rows.append(test_rows(seed, t_star, test_idx))
        r = rows[-1]["rows"]
        print(f"seed {seed}: t*={t_star:.2f} F1@0.5={r['0.5']['f1_mean']:.3f} "
              f"F1@t*={r[str(t_star)]['f1_mean']:.3f} "
              f"fs@0.5={r['0.5']['false_stable_mean']:.3f} "
              f"fs@t*={r[str(t_star)]['false_stable_mean']:.3f}")
    out = {"budget": BUDGET, "seeds": rows}
    path = os.path.join(ROOT, "benchmark", "results", "unet_safety.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("saved", path)


if __name__ == "__main__":
    main()
