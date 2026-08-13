"""Windows-side exact spectral radii for GPU-assembled transition matrices.

Reads a .npz produced by sdm_solver_torch.build_transition_batch_fast
(arrays 'D' (B, dim, dim) float32 and grid metadata) and computes exact
spectral radii with numpy/LAPACK in parallel. On this machine
~15 ms/matrix, so a full 128x80 (10240 matrices) grid takes ~10 s with
16 workers.

Usage:
    python code/benchmark_eig_cpu.py path/to/D.npz --out path/to/rho.npz
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np


def _rho(D: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(D))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="input npz with D and metadata")
    ap.add_argument("--out", default="")
    ap.add_argument("--nproc", type=int, default=16)
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    D = d["D"]
    t0 = time.perf_counter()
    with Pool(args.nproc) as pool:
        rho = np.array(pool.map(_rho, [D[k] for k in range(D.shape[0])],
                                chunksize=16))
    dt = time.perf_counter() - t0
    print(f"eigvals {D.shape[0]} matrices in {dt:.1f}s "
          f"({dt / D.shape[0] * 1000:.1f} ms/matrix)")
    out = args.out or (args.npz[:-4] + "_rho.npz")
    meta = {}
    for k in d.files:
        if k == "D":
            continue
        v = d[k]
        if v.dtype == object:
            v = np.asarray(str(v), dtype=object)
        meta[k] = v
    np.savez(out, rho=rho, **meta)
    print("saved", out)


if __name__ == "__main__":
    main()
