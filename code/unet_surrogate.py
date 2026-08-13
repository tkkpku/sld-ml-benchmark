"""
Compact U-Net surrogate for coarse-to-fine SLD super-resolution,
implemented in pure NumPy/SciPy (no PyTorch dependency).

Input  channels: 1 (bilinearly upsampled coarse stable mask) + 3 scalar
                 parameter channels (aD, zeta, fn broadcast over the grid)
Output: 1-channel probability map at the fine resolution (H x W).

The network is small (~45k parameters) and trained with Adam, BCE loss
with boundary weighting, and multi-scale coarse-input augmentation.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def conv2d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (N,C,H,W), w: (O,C,3,3), b: (O,). Same-size convolution, zero padding."""
    n, c, h, w_in = x.shape
    o = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.empty((9, n, c, h, w_in), dtype=x.dtype)
    k = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            cols[k] = xp[:, :, 1 + di:1 + di + h, 1 + dj:1 + dj + w_in]
            k += 1
    ww = w.reshape(o, c * 9)
    out = np.einsum("knchw,ock->nohw", cols, ww.reshape(o, c, 9))
    return out + b.reshape(1, o, 1, 1)


def conv2d_backward(x: np.ndarray, w: np.ndarray, dout: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, c, h, w_in = x.shape
    o = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.empty((9, n, c, h, w_in), dtype=x.dtype)
    k = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            cols[k] = xp[:, :, 1 + di:1 + di + h, 1 + dj:1 + dj + w_in]
            k += 1
    dw9 = np.einsum("knchw,nohw->ock", cols, dout)
    dw = dw9.reshape(o, c, 3, 3)
    db = dout.sum(axis=(0, 2, 3))
    # exact dx via full correlation with the rotated kernel
    wrot = w[:, :, ::-1, ::-1]
    dp = np.pad(dout, ((0, 0), (0, 0), (1, 1), (1, 1)))
    shifts = np.empty((9, n, o, h, w_in), dtype=x.dtype)
    k = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            shifts[k] = dp[:, :, 1 + di:1 + di + h, 1 + dj:1 + dj + w_in]
            k += 1
    g = np.einsum("knohw,ock->knchw", shifts, wrot.reshape(o, c, 9))
    dx = g.sum(axis=0)
    return dx, dw, db


def relu_forward(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    return dout * (x > 0)


def maxpool_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, c, h, w_in = x.shape
    h2, w2 = h // 2, w_in // 2
    out = np.full((n, c, h2, w2), -np.inf, dtype=x.dtype)
    arg = np.zeros((n, c, h2, w2), dtype=int)
    for di in (0, 1):
        for dj in (0, 1):
            block = x[:, :, di::2, dj::2]
            better = block > out
            out = np.where(better, block, out)
            arg = np.where(better, di * 2 + dj, arg)
    return out, arg


def maxpool_backward(dout: np.ndarray, arg: np.ndarray, shape: tuple) -> np.ndarray:
    n, c, h2, w2 = dout.shape
    dx = np.zeros(shape)
    for k, (di, dj) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        m = arg == k
        dx[:, :, di::2, dj::2] += dout * m
    return dx


def upsample2x(x: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)


def upsample2x_backward(dout: np.ndarray) -> np.ndarray:
    return dout[:, :, ::2, ::2] + dout[:, :, 1::2, ::2] + \
        dout[:, :, ::2, 1::2] + dout[:, :, 1::2, 1::2]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class UNet:
    """Small U-Net. Layers:
    enc1: conv 4->16, relu, pool
    enc2: conv 16->32, relu, pool
    bot : conv 32->64, relu
    dec2: up, concat enc2, conv (32+64)->32, relu
    dec1: up, concat enc1, conv (16+32)->16, relu
    head: conv 16->1 (logits)
    """

    def __init__(self, in_ch: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {}
        self.grad = {}
        shapes = {
            "w1": (16, in_ch), "b1": (16,),
            "w2": (32, 16), "b2": (32,),
            "w3": (64, 32), "b3": (64,),
            "w4": (32, 96), "b4": (32,),
            "w5": (16, 48), "b5": (16,),
            "w6": (1, 16), "b6": (1,),
        }
        for k, s in shapes.items():
            if k.startswith("w"):
                self.p[k] = rng.normal(0, np.sqrt(2.0 / (s[1] * 9)), s + (3, 3))
            else:
                self.p[k] = np.zeros(s)

    def forward(self, x: np.ndarray, cache: dict | None = None) -> np.ndarray:
        p = self.p
        c1 = conv2d_forward(x, p["w1"], p["b1"]); r1 = relu_forward(c1)
        p1, a1 = maxpool_forward(r1)
        c2 = conv2d_forward(p1, p["w2"], p["b2"]); r2 = relu_forward(c2)
        p2, a2 = maxpool_forward(r2)
        c3 = conv2d_forward(p2, p["w3"], p["b3"]); r3 = relu_forward(c3)
        u2 = upsample2x(r3)
        cat2 = np.concatenate([u2, r2], axis=1)
        c4 = conv2d_forward(cat2, p["w4"], p["b4"]); r4 = relu_forward(c4)
        u1 = upsample2x(r4)
        cat1 = np.concatenate([u1, r1], axis=1)
        c5 = conv2d_forward(cat1, p["w5"], p["b5"]); r5 = relu_forward(c5)
        logits = conv2d_forward(r5, p["w6"], p["b6"])
        if cache is not None:
            cache.update(x=x, c1=c1, r1=r1, a1=a1, p1=p1, c2=c2, r2=r2, a2=a2,
                         p2=p2, c3=c3, r3=r3, u2=u2, cat2=cat2, c4=c4, r4=r4,
                         u1=u1, cat1=cat1, c5=c5, r5=r5)
        return logits

    def backward(self, dlogits: np.ndarray, cache: dict) -> None:
        p, g = self.p, self.grad
        dr5, dw6, db6 = conv2d_backward(cache["r5"], p["w6"], dlogits)
        dc5 = relu_backward(cache["c5"], dr5)
        dcat1, dw5, db5 = conv2d_backward(cache["cat1"], p["w5"], dc5)
        du1 = dcat1[:, :32]
        dr1_skip = dcat1[:, 32:]
        du1 = upsample2x_backward(du1)
        dc4 = relu_backward(cache["c4"], du1)
        dcat2, dw4, db4 = conv2d_backward(cache["cat2"], p["w4"], dc4)
        du2 = dcat2[:, :64]
        dr2_skip = dcat2[:, 64:]
        du2 = upsample2x_backward(du2)
        dc3 = relu_backward(cache["c3"], du2)
        dp2, dw3, db3 = conv2d_backward(cache["p2"], p["w3"], dc3)
        dr2 = maxpool_backward(dp2, cache["a2"], cache["r2"].shape) + dr2_skip
        dc2 = relu_backward(cache["c2"], dr2)
        dp1, dw2, db2 = conv2d_backward(cache["p1"], p["w2"], dc2)
        dr1 = maxpool_backward(dp1, cache["a1"], cache["r1"].shape)
        dc1 = relu_backward(cache["c1"], dr1 + dr1_skip)
        dx, dw1, db1 = conv2d_backward(cache["x"], p["w1"], dc1)
        g["w6"] = dw6; g["b6"] = db6
        g["w5"] = dw5; g["b5"] = db5
        g["w4"] = dw4; g["b4"] = db4
        g["w3"] = dw3; g["b3"] = db3
        g["w2"] = dw2; g["b2"] = db2
        g["w1"] = dw1; g["b1"] = db1

    def param_count(self) -> int:
        return int(sum(v.size for v in self.p.values()))


def bce_with_boundary_weight(logits: np.ndarray, y: np.ndarray,
                             w_boundary: float = 5.0):
    """Binary cross-entropy with extra weight on boundary pixels of y."""
    prob = sigmoid(logits)
    y4 = y[:, None]
    eps = 1e-7
    loss = -(y4 * np.log(prob + eps) + (1 - y4) * np.log(1 - prob + eps))
    from scipy.ndimage import sobel
    edge = np.hypot(sobel(y4.astype(float), axis=2),
                    sobel(y4.astype(float), axis=3)) > 0
    weight = np.where(edge, w_boundary, 1.0)
    loss = loss * weight
    return float(loss.mean()), loss, weight


def threshold(prob: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (prob >= t).astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.reshape(-1).astype(bool)
    yp = y_pred.reshape(-1).astype(bool)
    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()
    tn = np.logical_and(~yt, ~yp).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec),
            "accuracy": float(acc), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def mean_boundary_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt, sobel
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    e1 = np.hypot(sobel(y_true.astype(float), axis=0), sobel(y_true.astype(float), axis=1)) > 0
    e2 = np.hypot(sobel(y_pred.astype(float), axis=0), sobel(y_pred.astype(float), axis=1)) > 0
    d12 = distance_transform_edt(~e1)[e2]
    d21 = distance_transform_edt(~e2)[e1]
    if len(d12) == 0 or len(d21) == 0:
        return 0.0
    return float(0.5 * (d12.mean() + d21.mean()))


class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 weight_decay: float = 0.0):
        self.lr, self.b1, self.b2 = lr, beta1, beta2
        self.wd = weight_decay
        self.m, self.v = {}, {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            g = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


if __name__ == "__main__":
    # quick gradient check vs numerical differentiation on a tiny input
    rng = np.random.default_rng(0)
    net = UNet(seed=0)
    x = rng.normal(size=(1, 4, 8, 8)).astype(np.float64)
    cache = {}
    logits = net.forward(x, cache)
    dy = rng.normal(size=logits.shape)
    net.backward(dy, cache)
    eps = 1e-6
    for k in ("w1", "b1", "w6", "b6"):
        num = np.zeros_like(net.p[k])
        it = np.nditer(net.p[k], flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            orig = net.p[k][i]
            net.p[k][i] = orig + eps
            l1 = net.forward(x)
            net.p[k][i] = orig - eps
            l2 = net.forward(x)
            num[i] = ((l1 - l2) * dy).sum() / (2 * eps)
            net.p[k][i] = orig
            it.iternext()
        err = np.abs(num - net.grad[k]).max()
        print(f"grad check {k}: max err = {err:.3e}")
