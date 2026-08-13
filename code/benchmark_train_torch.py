"""SLD-ML Benchmark v1 U-Net training (GPU, WSL2 yolo_env).

Data: benchmark/data/rho (fine m80 labels, c16/c32 inputs, 90 cases).
Split: 70 train / 10 val / 10 test from benchmark/meta.json.
Input: bilinearly upsampled c16 (rho-1) field + normalized parameter
channels; target: fine m80 stable mask. Five augmentation variants per case
(same as revision-2 protocol). Validation-based checkpoint, three seeds,
ablations (map/param for seed 0).

Usage:
  python code/benchmark_train_torch.py prepare
  python code/benchmark_train_torch.py train <seed> <both|map|param>
  python code/benchmark_train_torch.py eval
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_meta, load_case, split_indices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BENCH = os.path.join(ROOT, "benchmark")
OUT = os.path.join(BENCH, "results")
OSD = os.path.join(BENCH, "data", "torch_data")


class UNetTorch(torch.nn.Module):
    """Same architecture/init as unet_surrogate.UNet (58,481 params)."""

    def __init__(self, in_ch: int = 4, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        shapes = {
            "w1": (16, in_ch), "b1": (16,), "w2": (32, 16), "b2": (32,),
            "w3": (64, 32), "b3": (64,), "w4": (32, 96), "b4": (32,),
            "w5": (16, 48), "b5": (16,), "w6": (1, 16), "b6": (1,),
        }
        for k, s in shapes.items():
            if k.startswith("w"):
                arr = rng.normal(0, np.sqrt(2.0 / (s[1] * 9)), s + (3, 3))
            else:
                arr = np.zeros(s)
            p = torch.nn.Parameter(torch.tensor(arr, dtype=torch.float32))
            self.register_parameter(k, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = F.conv2d(x, self.w1, self.b1, padding=1); r1 = F.relu(c1)
        p1 = F.max_pool2d(r1, 2)
        c2 = F.conv2d(p1, self.w2, self.b2, padding=1); r2 = F.relu(c2)
        p2 = F.max_pool2d(r2, 2)
        c3 = F.conv2d(p2, self.w3, self.b3, padding=1); r3 = F.relu(c3)
        u2 = F.interpolate(r3, scale_factor=2, mode="nearest")
        cat2 = torch.cat([u2, r2], dim=1)
        c4 = F.conv2d(cat2, self.w4, self.b4, padding=1); r4 = F.relu(c4)
        u1 = F.interpolate(r4, scale_factor=2, mode="nearest")
        cat1 = torch.cat([u1, r1], dim=1)
        c5 = F.conv2d(cat1, self.w5, self.b5, padding=1); r5 = F.relu(c5)
        return F.conv2d(r5, self.w6, self.b6, padding=1)


def params_norm(case: dict) -> np.ndarray:
    return np.array([case["aD"], case["zeta"] * 100.0, case["fn"] / 1000.0],
                    dtype=np.float32)


def make_input(rho_c16: np.ndarray, case: dict, channels: str = "both"
               ) -> np.ndarray:
    """(4, 128, 80) input from the 16x10 rho field."""
    t = torch.tensor((rho_c16 - 1.0)[None, None].astype(np.float32))
    field = F.interpolate(t, size=(128, 80), mode="bilinear",
                          align_corners=False)[0, 0].numpy()
    ch = np.full((3, 128, 80), params_norm(case)[:, None, None],
                 dtype=np.float32)
    if channels == "param":
        field = np.zeros_like(field)
    elif channels == "map":
        ch = np.zeros_like(ch)
    return np.concatenate([field[None].astype(np.float32), ch], axis=0)


def augment(case: dict, rng: np.random.Generator) -> list[np.ndarray]:
    d = load_case(case["idx"], "c16", 20)
    rho = d["rho"]
    out = []
    # bilinear (identity on this grid), nearest, smoothed, +0.10 offset
    t = torch.tensor(rho[None, None].astype(np.float32))
    nearest = F.interpolate(t, size=(16, 10), mode="nearest")[0, 0].numpy()
    sm = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)[0, 0].numpy()
    for f in (rho, nearest, sm, rho + 0.10):
        out.append(f)
    # parameter jitter variant (same field, perturbed aD in the input params)
    pp = dict(case)
    pp["aD"] = float(np.clip(case["aD"] * (1 + 0.05 * rng.standard_normal()),
                             0.05, 1.0))
    out.append(("jitter", pp))
    return out


def prepare() -> None:
    meta = load_meta()
    sidx = split_indices(meta)
    H, W = 128, 80
    os.makedirs(OSD, exist_ok=True)
    for split in ("train", "val", "test"):
        xs, ys = [], []
        for idx in sidx[split]:
            case = dict(meta["cases"][idx])
            case["idx"] = idx
            fine = load_case(idx, "fine", 80)["rho"]
            y = (fine < 1.0).astype(np.float32)
            rng = np.random.default_rng(1000 + idx)
            variants = augment(case, rng)
            if split == "train":
                for f in variants:
                    if isinstance(f, tuple):
                        _, pp = f
                        xs.append(make_input(rho_field(case), pp, "both"))
                    else:
                        xs.append(make_input(f, case, "both"))
                    ys.append(y)
            else:
                # canonical input for val/test: bilinear c16 field
                xs.append(make_input(load_case(idx, "c16", 20)["rho"],
                                     case, "both"))
                ys.append(y)
        arr = {"x": np.stack(xs), "y": np.stack(ys)}
        np.savez(os.path.join(OSD, f"{split}.npz"), **arr)
        print(split, arr["x"].shape, arr["y"].shape, flush=True)
    # Sobel boundary weights for training
    tr = np.load(os.path.join(OSD, "train.npz"))
    y = tr["y"]
    sobel = np.abs(np.gradient(y.astype(np.float32), axis=1)) + \
        np.abs(np.gradient(y.astype(np.float32), axis=2))
    w = 1.0 + 4.0 * (sobel > 0).astype(np.float32)
    np.savez(os.path.join(OSD, "train_w.npz"), w=w)
    print("prepared", OSD)


def rho_field(case: dict) -> np.ndarray:
    return load_case(case["idx"], "c16", 20)["rho"]


def metrics_t(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.reshape(-1).astype(bool)
    yp = y_pred.reshape(-1).astype(bool)
    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def train_one(seed: int, channels: str = "both", epochs: int = 35,
              eval_every: int = 5) -> dict:
    tr = np.load(os.path.join(OSD, "train.npz"))
    va = np.load(os.path.join(OSD, "val.npz"))
    tw = np.load(os.path.join(OSD, "train_w.npz"))
    x = torch.tensor(tr["x"], device=DEVICE)
    y = torch.tensor(tr["y"][:, None].astype(np.float32), device=DEVICE)
    w = torch.tensor(tw["w"][:, None], device=DEVICE)
    cx = torch.tensor(va["x"], device=DEVICE)
    cy = va["y"].astype(np.float32)
    if channels == "map":
        x = x.clone(); x[:, 1:] = 0.0
    elif channels == "param":
        x = x.clone(); x[:, 0] = 0.0
    net = UNetTorch(seed=seed).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    eps = 1e-7
    n = x.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(1000 + seed)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
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
        if ep % eval_every == 0 or ep == epochs:
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
            print(f"seed={seed} ch={channels} ep={ep}: "
                  f"loss={rec['train_loss']:.4f} "
                  f"valF1={rec.get('val_f1', float('nan')):.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    if best["state"] is not None:
        np.savez(os.path.join(OUT, f"unet_s{seed}_{channels}.npz"),
                 **best["state"])
    with open(os.path.join(OUT, f"unet_s{seed}_{channels}_hist.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"seed={seed} ch={channels}: best val F1 {best['f1']:.4f} "
          f"at ep {best['epoch']} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return best


def evaluate(seeds=(0, 1, 2), channels_list=("both", "map", "param"),
             thresholds=(0.5, 0.6, 0.7)) -> None:
    te = np.load(os.path.join(OSD, "test.npz"))
    y = te["y"]
    x16_all = torch.tensor(te["x"], device=DEVICE)
    rows = {}
    for ch in channels_list:
        for seed in seeds:
            path = os.path.join(OUT, f"unet_s{seed}_{ch}.npz")
            if not os.path.exists(path):
                continue
            state = np.load(path)
            net = UNetTorch(seed=seed).to(DEVICE)
            net.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
            net.eval()
            x16 = x16_all.clone()
            if ch == "map":
                x16[:, 1:] = 0.0
            elif ch == "param":
                x16[:, 0] = 0.0
            with torch.no_grad():
                p16 = torch.sigmoid(net(x16)).cpu().numpy()[:, 0]
            key = (ch, seed)
            rows[key] = {}
            for t in thresholds:
                ms = [metrics_t(y[i], p16[i] >= t) for i in range(len(y))]
                fs = [m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
                      for m in ms]
                rows[key][f"c16_t{t}"] = {
                    "f1_mean": float(np.mean([m["f1"] for m in ms])),
                    "f1_std": float(np.std([m["f1"] for m in ms])),
                    "false_stable_mean": float(np.mean(fs)),
                    "per_case": ms,
                }
    with open(os.path.join(OUT, "unet_test_metrics.json"), "w") as f:
        json.dump({f"{k[0]}_s{k[1]}": v for k, v in rows.items()}, f,
                  indent=1, default=float)
    for key, v in rows.items():
        print(f"{key}: c16_t0.5 F1={v['c16_t0.5']['f1_mean']:.4f} "
              f"falseStable={v['c16_t0.5']['false_stable_mean']:.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if mode == "prepare":
        prepare()
    elif mode == "train":
        train_one(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "both")
    elif mode == "eval":
        evaluate()
    else:
        raise SystemExit(f"unknown mode {mode}")
