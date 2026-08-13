"""Failure-zone fine-tuning for direction 2 (WSL2 GPU, PyTorch).

The full-90-case head-to-head (benchmark_full_eval.py) shows the release
U-Net loses to the analytic ZOA boundary in every failure zone. This script
runs the last decisive experiment: does fine-tuning the pretrained U-Net on
failure-zone training cases (or on all training cases with failure-zone
cases weighted) let it beat ZOA on the held-out failure-zone test cases?

Variants:
  --variant zone_only   fine-tune only on train cases inside the zone
  --variant reweighted  fine-tune on all train cases, zone cases x3 weight

The zone is "two or more weak factors" (aD>=0.5, zeta<=0.008, fn<=800),
defined from zoa_region_scan.json; test cases are never used for training.
Checkpoints are selected on the validation split (same rule as the release).

Run (WSL2, yolo_env):
  /home/tan83/yolo_env/bin/python code/benchmark_finetune_failure.py \
      --variant zone_only --seed 0 --epochs 30
  /home/tan83/yolo_env/bin/python code/benchmark_finetune_failure.py \
      --variant reweighted --seed 0 --epochs 20

Output: benchmark/results/finetune_failure_<variant>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from benchmark_train_torch import (  # noqa: E402
    UNetTorch, augment, make_input, metrics_t, rho_field,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results")


def in_zone(case: dict) -> bool:
    weak = sum((case["aD"] >= 0.5, case["zeta"] <= 0.008,
                case["fn"] <= 800.0))
    return weak >= 2


def build_split(sidx, split: str, zone_only: bool):
    meta = load_meta()
    xs, ys, ws, meta_cases = [], [], [], []
    for idx in sidx[split]:
        case = dict(meta["cases"][idx])
        case["idx"] = idx
        z = in_zone(case)
        if zone_only and not z:
            continue
        fine = load_case(idx, "fine", 80)["rho"]
        y = (fine < 1.0).astype(np.float32)
        if split == "train" and zone_only:
            rng = np.random.default_rng(1000 + idx)
            for f in augment(case, rng):
                if isinstance(f, tuple):
                    _, pp = f
                    xs.append(make_input(rho_field(case), pp, "both"))
                else:
                    xs.append(make_input(f, case, "both"))
                ys.append(y)
                ws.append(1.0)
                meta_cases.append(case)
        else:
            xs.append(make_input(load_case(idx, "c16", 20)["rho"],
                                 case, "both"))
            ys.append(y)
            ws.append(3.0 if z else 1.0)
            meta_cases.append(case)
    return (np.stack(xs), np.stack(ys), np.array(ws, dtype=np.float32),
            meta_cases)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("zone_only", "reweighted"),
                    default="zone_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    zone_only = args.variant == "zone_only"

    meta = load_meta()
    sidx = split_indices(meta)
    tr_x, tr_y, tr_w, _ = build_split(sidx, "train", zone_only)
    va_x, va_y, _, va_cases = build_split(sidx, "val", False)
    te_x, te_y, _, te_cases = build_split(sidx, "test", False)
    n_zone_train = sum(in_zone(meta["cases"][i]) for i in sidx["train"])
    n_zone_val = sum(in_zone(meta["cases"][i]) for i in sidx["val"])
    n_zone_test = sum(in_zone(meta["cases"][i]) for i in sidx["test"])
    print(f"variant={args.variant} device={DEVICE} "
          f"train={tr_x.shape[0]} (zone train cases {n_zone_train}) "
          f"val={va_x.shape[0]} test={te_x.shape[0]} "
          f"zone val/test={n_zone_val}/{n_zone_test}", flush=True)

    x = torch.tensor(tr_x, device=DEVICE)
    y = torch.tensor(tr_y[:, None].astype(np.float32), device=DEVICE)
    w = torch.tensor(tr_w[:, None, None, None], device=DEVICE)
    cx = torch.tensor(va_x, device=DEVICE)
    cy = va_y.astype(np.float32)

    net = UNetTorch(seed=args.seed).to(DEVICE)
    pretrained = np.load(os.path.join(OUT, f"unet_s{args.seed}_both.npz"))
    net.load_state_dict({k: torch.tensor(v) for k, v in pretrained.items()})
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    eps = 1e-7
    n = x.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(2000 + args.seed)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        rng.shuffle(idx)
        net.train()
        total = 0.0
        for s in range(0, n, 16):
            bi = idx[s:s + 16]
            logits = net(x[bi])
            prob = torch.sigmoid(logits)
            loss = -(y[bi] * torch.log(prob + eps) +
                     (1 - y[bi]) * torch.log(1 - prob + eps))
            loss = (loss * w[bi]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(bi)
        rec = {"epoch": ep, "train_loss": total / n}
        if ep % 5 == 0 or ep == args.epochs:
            net.eval()
            with torch.no_grad():
                probv = torch.sigmoid(net(cx)).cpu().numpy()[:, 0]
            f1s = [metrics_t(cy[i], probv[i] >= 0.5)["f1"]
                   for i in range(len(cy))]
            vf1 = float(np.mean(f1s))
            rec["val_f1"] = vf1
            if vf1 > best["f1"]:
                best["f1"] = vf1
                best["epoch"] = ep
                best["state"] = {k: v.detach().cpu().numpy().copy()
                                 for k, v in net.state_dict().items()}
        history.append(rec)
        if ep % 5 == 0:
            print(f"ep={ep}: loss={rec['train_loss']:.4f} "
                  f"valF1={rec.get('val_f1', float('nan')):.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    net.load_state_dict({k: torch.tensor(v) for k, v in best["state"].items()})
    net.eval()
    with torch.no_grad():
        prob_te = torch.sigmoid(net(torch.tensor(te_x, device=DEVICE))
                                ).cpu().numpy()[:, 0]
    rows = []
    for i in range(len(te_y)):
        m = metrics_t(te_y[i], prob_te[i] >= 0.5)
        fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        rows.append({
            "idx": te_cases[i]["idx"],
            "case": te_cases[i],
            "in_zone": in_zone(te_cases[i]),
            "f1": m["f1"],
            "false_stable": float(fs),
        })
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "pretrained": f"unet_s{args.seed}_both.npz",
        "zone_definition": "two or more weak factors: aD>=0.5, zeta<=0.008, "
                           "fn<=800",
        "zone_train_cases": n_zone_train,
        "zone_val_cases": n_zone_val,
        "zone_test_cases": n_zone_test,
        "best_val_f1": best["f1"],
        "best_epoch": best["epoch"],
        "history": history,
        "test_all_mean_f1": float(np.mean([r["f1"] for r in rows])),
        "test_zone_mean_f1": float(np.mean(
            [r["f1"] for r in rows if r["in_zone"]])),
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    out = os.path.join(OUT, f"finetune_failure_{args.variant}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"test all F1 {result['test_all_mean_f1']:.4f}, "
          f"zone F1 {result['test_zone_mean_f1']:.4f}, best val "
          f"{best['f1']:.4f}@ep{best['epoch']}")
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
