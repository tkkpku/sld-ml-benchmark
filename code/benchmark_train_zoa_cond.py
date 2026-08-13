"""ZOA-conditioned U-Net (direction 8, phase 2; WSL2 GPU, PyTorch).

The analytic ZOA boundary is nearly free on the coarse 16x10 grid
(~0.05 s vs 3.19 s for the fine 128x80 grid). Conditioning the U-Net on the
upsampled coarse ZOA mask gives the learned surrogate an analytic prior
while keeping the fast pipeline: coarse ZOA + U-Net inference is about
0.11 s/map, ~30x cheaper than the fine ZOA map.

Variants:
  --channels field_zoa  : 2 channels (c16 rho-1 field + coarse ZOA)
  --channels full       : 5 channels (+ normalized parameters)

Same architecture, augmentation and validation-based checkpoint rule as
the release model (benchmark_train_torch.py). Run from WSL2 yolo_env:
  /home/tan83/yolo_env/bin/python code/benchmark_train_zoa_cond.py \
      --channels field_zoa --seed 0 --epochs 35

Output: benchmark/results/unet_zc_<channels>_s<seed>.npz (+ _hist.json)
        benchmark/results/zoa_cond_test_<channels>_s<seed>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices  # noqa: E402
from benchmark_train_torch import UNetTorch, params_norm, metrics_t  # noqa: E402
from sdm_solver import MillingParams  # noqa: E402
from zoa_baseline import zoa_sld  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "benchmark", "results")

C16_N = np.linspace(4000.0, 16000.0, 10)
C16_A = np.linspace(0.05e-3, 1.5e-3, 16)


def coarse_zoa(case: dict) -> np.ndarray:
    p = MillingParams(aD=case["aD"], zeta=case["zeta"], fn=case["fn"])
    return zoa_sld(C16_N, C16_A, p).astype(np.float32)  # (16,10) 1=stable


def make_input(rho_c16: np.ndarray, case: dict, zoa_c: np.ndarray,
               channels: str) -> np.ndarray:
    def up(x, size=(128, 80)):
        t = torch.tensor(x[None, None].astype(np.float32))
        return F.interpolate(t, size=size, mode="bilinear",
                             align_corners=False)[0, 0].numpy()
    field = up(rho_c16 - 1.0)
    zoa = up(zoa_c)
    if channels == "field_zoa":
        return np.concatenate([field[None], zoa[None]], axis=0).astype(
            np.float32)
    ch = np.full((3, 128, 80), params_norm(case)[:, None, None],
                 dtype=np.float32)
    return np.concatenate([field[None], zoa[None], ch], axis=0).astype(
        np.float32)


def augment_variants(case: dict, rng: np.random.Generator):
    rho = load_case(case["idx"], "c16", 20)["rho"]
    t = torch.tensor(rho[None, None].astype(np.float32))
    nearest = F.interpolate(t, size=(16, 10), mode="nearest")[0, 0].numpy()
    sm = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)[0, 0].numpy()
    for f in (rho, nearest, sm, rho + 0.10):
        yield f, case
    pp = dict(case)
    pp["aD"] = float(np.clip(case["aD"] * (1 + 0.05 * rng.standard_normal()),
                             0.05, 1.0))
    yield rho, pp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=("field_zoa", "full"),
                    default="field_zoa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=35)
    args = ap.parse_args()
    in_ch = 2 if args.channels == "field_zoa" else 5
    meta = load_meta()
    sidx = split_indices(meta)

    # ---- build datasets in memory ----
    def build(split: str, train_aug: bool):
        xs, ys = [], []
        for idx in sidx[split]:
            case = dict(meta["cases"][idx])
            case["idx"] = idx
            fine = load_case(idx, "fine", 80)["rho"]
            y = (fine < 1.0).astype(np.float32)
            zc = coarse_zoa(case)
            if train_aug:
                rng = np.random.default_rng(1000 + idx)
                for f, cp in augment_variants(case, rng):
                    xs.append(make_input(f, cp, zc, args.channels))
                    ys.append(y)
            else:
                xs.append(make_input(load_case(idx, "c16", 20)["rho"],
                                     case, zc, args.channels))
                ys.append(y)
        return np.stack(xs), np.stack(ys)

    tr_x, tr_y = build("train", True)
    va_x, va_y = build("val", False)
    te_x, te_y = build("test", False)
    sobel = np.abs(np.gradient(tr_y.astype(np.float32), axis=1)) + \
        np.abs(np.gradient(tr_y.astype(np.float32), axis=2))
    tw = (1.0 + 4.0 * (sobel > 0).astype(np.float32))
    print(f"channels={args.channels} in_ch={in_ch} device={DEVICE} "
          f"train={tr_x.shape[0]} val={va_x.shape[0]} test={te_x.shape[0]}",
          flush=True)

    x = torch.tensor(tr_x, device=DEVICE)
    y = torch.tensor(tr_y[:, None].astype(np.float32), device=DEVICE)
    w = torch.tensor(tw[:, None], device=DEVICE)
    cx = torch.tensor(va_x, device=DEVICE)
    cy = va_y.astype(np.float32)

    net = UNetTorch(in_ch=in_ch, seed=args.seed).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    eps = 1e-7
    n = x.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(3000 + args.seed)
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
    test_idx = list(sidx["test"])
    rows = []
    for i in range(len(te_y)):
        m = metrics_t(te_y[i], prob_te[i] >= 0.5)
        fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        rows.append({"idx": test_idx[i], "case": meta["cases"][test_idx[i]],
                     "f1": m["f1"], "false_stable": float(fs)})
    result = {
        "channels": args.channels,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_val_f1": best["f1"],
        "best_epoch": best["epoch"],
        "test_all_mean_f1": float(np.mean([r["f1"] for r in rows])),
        "test_all_false_stable": float(np.mean(
            [r["false_stable"] for r in rows])),
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime()),
    }
    np.savez(os.path.join(OUT, f"unet_zc_{args.channels}_s{args.seed}.npz"),
             **best["state"])
    with open(os.path.join(
            OUT, f"unet_zc_{args.channels}_s{args.seed}_hist.json"),
            "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(
            OUT, f"zoa_cond_test_{args.channels}_s{args.seed}.json"),
            "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"best val F1 {best['f1']:.4f}@ep{best['epoch']}; "
          f"test F1 {result['test_all_mean_f1']:.4f} "
          f"(fs {result['test_all_false_stable']:.4f})")
    print("saved weights + metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
