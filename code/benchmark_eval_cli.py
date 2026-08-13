"""Standard evaluation entry point for SLD-ML Benchmark v1 (finding 5 /
review direction 5: turn the protocol into code).

Usage:
    python code/benchmark_eval_cli.py --pred <pred.npz|pred.npy> \
        --case <idx> [--key mask|rho] [--t 0.5] [--out out.json]

`pred` must contain either a binary stable mask (key "mask", shape 128x80 or
flat 10240) or a spectral-radius field (key "rho") that is thresholded at
`--t` (rho < t => stable). Reports F1, precision, recall, false-stable
fraction, mean boundary distance, label stable fraction and the difficulty
layer (trivial/medium/hard). This is the same metrics code as the release
evaluation (eval_benchmark.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_loader import load_case  # noqa: E402
from unet_surrogate import mean_boundary_distance, metrics  # noqa: E402


def layer_of(stable_frac: float) -> str:
    if stable_frac >= 0.99:
        return "trivial"
    if stable_frac >= 0.5:
        return "medium"
    return "hard"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="npz/npy prediction file")
    ap.add_argument("--case", type=int, required=True, help="benchmark case idx")
    ap.add_argument("--key", choices=("mask", "rho"), default="mask")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    fine = load_case(args.case, "fine", 80)
    y = (fine["rho"] < 1.0).astype(np.float32)
    if args.pred.endswith(".npy"):
        p = np.load(args.pred)
    else:
        d = np.load(args.pred, allow_pickle=True)
        if args.key not in d.files:
            raise SystemExit(f"--key '{args.key}' not in {args.pred}: "
                             f"{list(d.files)}")
        p = d[args.key]
    p = np.asarray(p, dtype=np.float32).reshape(y.shape)
    if args.key == "rho":
        pred = (p < args.t).astype(np.float32)
    else:
        pred = p
    m = metrics(y, pred)
    fs = m["fp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
    stable_frac = float(y.mean())
    result = {
        "case": args.case,
        "pred_file": args.pred,
        "key": args.key,
        "threshold": args.t,
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "false_stable": float(fs),
        "mean_boundary_dist": float(mean_boundary_distance(y, pred)),
        "label_stable_frac": stable_frac,
        "layer": layer_of(stable_frac),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print("saved", args.out)
    else:
        print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
