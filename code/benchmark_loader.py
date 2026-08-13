"""Unified loader for the SLD-ML Benchmark v1 data files.

Data layout (benchmark/data/rho/):
  case_<idx:03d>_<grid>_m<m>.npz
    rho      : (H*W,) spectral radii (row-major over (depth, speed))
    n_rpms   : (W,)
    a_p_mm   : (H,)
    case     : parameter dict
    m        : discretization count
Grids: fine (128x80, m=80 or m=160), c32 (32x20, m=40), c16 (16x10, m=20).
"""

from __future__ import annotations

import json
import os
import ast

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BENCH = os.path.join(ROOT, "benchmark")
DATA = os.path.join(BENCH, "data", "rho")


def load_meta() -> dict:
    with open(os.path.join(BENCH, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_case(idx: int, grid: str = "fine", m: int = 80) -> dict:
    path = os.path.join(DATA, f"case_{idx:03d}_{grid}_m{m}.npz")
    d = np.load(path, allow_pickle=True)
    H = len(d["a_p_mm"])
    W = len(d["n_rpms"])
    fc = d["case"]
    if isinstance(fc, np.ndarray):
        fc = fc.item()
    if isinstance(fc, str):
        fc = ast.literal_eval(fc)
    return {
        "idx": idx,
        "rho": d["rho"].reshape(H, W),
        "n_rpms": d["n_rpms"],
        "a_p_mm": d["a_p_mm"],
        "case": fc,
        "m": int(d["m"]),
    }


def load_all(grid: str = "fine", m: int = 80) -> list[dict]:
    meta = load_meta()
    out = []
    for idx in range(meta["n_cases"]):
        path = os.path.join(DATA, f"case_{idx:03d}_{grid}_m{m}.npz")
        if os.path.exists(path):
            out.append(load_case(idx, grid, m))
    return out


def split_indices(meta: dict) -> dict[str, list[int]]:
    """Map split name to global case indices."""
    cases = meta["cases"]
    out = {}
    for split, cs in meta["split"].items():
        keys = {(c["aD"], c["zeta"], c["fn"]) for c in cs}
        out[split] = [i for i, c in enumerate(cases)
                      if (c["aD"], c["zeta"], c["fn"]) in keys]
    return out


if __name__ == "__main__":
    meta = load_meta()
    print("n_cases:", meta["n_cases"])
    print("split:", {k: len(v) for k, v in split_indices(meta).items()})
    c = load_case(0, "fine", 80)
    print("case 0:", c["case"], "rho shape:", c["rho"].shape,
          "m:", c["m"])
