#!/usr/bin/env bash
# SLD-ML Benchmark v1 - GPU assembly stage (run inside WSL2 yolo_env).
# Generates transition-matrix files benchmark/data/D/<case>_<grid>_m<m>.npz
# for every case in benchmark/meta.json.
#
# Usage: bash code/benchmark_generate.sh [grid] [m] [split]
#   grid: fine|c32|c16 (default fine)
#   m:    80|160 (default 80)
#   split: all|train|val|test (default all)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/tan83/yolo_env/bin/python
GRID="${1:-fine}"
M="${2:-80}"
SPLIT="${3:-all}"
OUT=benchmark/data/D
mkdir -p "$OUT"

$PY - <<EOF
import json, os, subprocess, sys, time
meta = json.load(open("benchmark/meta.json", encoding="utf-8"))
cases = meta["cases"]
split = meta["split"]
if "$SPLIT" != "all":
    keys = {(c["aD"], c["zeta"], c["fn"]) for c in split["$SPLIT"]}
    cases = [c for c in cases if (c["aD"], c["zeta"], c["fn"]) in keys]
grid = "$GRID"
m = int("$M")
grids = meta["grids"]
g = grids["fine" if grid == "fine" and m == 80 else
        ("fine160" if grid == "fine" and m == 160 else grid)]
n_rpm = g["n_rpm"][2]
a_p = g["a_p_mm"][2]
chunk = 32 if m == 160 else 128
t0 = time.perf_counter()
for idx, case in enumerate(cases):
    key = (case["aD"], case["zeta"], case["fn"])
    path = os.path.join("$OUT", f"case_{idx:03d}_{grid}_m{m}.npz")
    if os.path.exists(path):
        continue
    cmd = [sys.executable, "code/sdm_solver_torch.py",
           "--aD", str(case["aD"]), "--zeta", str(case["zeta"]),
           "--fn", str(case["fn"]), "--m", str(m), "--grid", grid,
           "--dump-D", "--out", path]
    t1 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"[{idx+1}/{len(cases)}] {key} {grid} m{m}: "
          f"{time.perf_counter()-t1:.0f}s (total {time.perf_counter()-t0:.0f}s)",
          flush=True)
print(f"DONE {len(cases)} cases in {time.perf_counter()-t0:.0f}s")
EOF
