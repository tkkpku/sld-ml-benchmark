"""SLD-ML Benchmark v1: 90-case parameter space and train/val/test split.

Design goals (see docs/2026-08-12_战略规划_三审意见与基准化路线.md):
  - 90 parameter sets: 70 train / 10 validation / 10 test;
  - deterministic subset of the 12 x 7 x 6 structured grid (not the full
    504-combination Cartesian product; no random draws);
  - test set contains extrapolation regions (low immersion aD<0.15,
    low damping zeta<0.008, low natural frequency fn<850 Hz);
  - the three original held-out cases (aD=0.1, zeta=0.011, fn=922),
    (aD=0.5, zeta=0.005, fn=922) and (aD=0.5, zeta=0.011, fn=800) are kept;
  - physical domain: 4000-16000 rpm, 0.05-1.5 mm.
"""

from __future__ import annotations

import itertools
import json
import os


def build_cases() -> dict:
    aD_vals = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50,
               0.60, 0.75, 0.90, 1.00]
    zeta_vals = [0.005, 0.008, 0.011, 0.015, 0.020, 0.025, 0.030]
    fn_vals = [700.0, 800.0, 850.0, 922.0, 1000.0, 1100.0]

    cases = []
    seen = set()

    def add(aD, zeta, fn):
        key = (round(aD, 4), round(zeta, 4), round(fn, 1))
        if key not in seen:
            seen.add(key)
            cases.append({"aD": aD, "zeta": zeta, "fn": fn})

    # --- fixed anchor cases (original 21, order preserved) ---
    anchors = [
        (0.50, 0.011, 922.0), (0.25, 0.011, 922.0), (0.75, 0.011, 922.0),
        (0.50, 0.020, 922.0), (0.50, 0.011, 1100.0), (1.00, 0.011, 922.0),
        (0.10, 0.011, 922.0), (0.50, 0.005, 922.0), (0.50, 0.011, 800.0),
        (0.35, 0.011, 922.0), (0.60, 0.011, 922.0), (0.90, 0.011, 922.0),
        (0.50, 0.015, 922.0), (0.50, 0.025, 922.0), (0.50, 0.011, 1000.0),
        (0.50, 0.011, 850.0), (0.25, 0.015, 1000.0), (0.75, 0.020, 850.0),
        (0.35, 0.008, 922.0), (0.60, 0.030, 922.0), (0.50, 0.008, 700.0),
    ]
    for a in anchors:
        add(*a)

    # --- systematic low-immersion coverage (extrapolation region) ---
    for aD in (0.05, 0.08, 0.10, 0.15):
        for zeta, fn in ((0.011, 922.0), (0.015, 850.0), (0.020, 1000.0),
                         (0.008, 800.0)):
            add(aD, zeta, fn)
    # --- systematic low-damping coverage ---
    for zeta in (0.005, 0.008):
        for aD, fn in ((0.25, 922.0), (0.60, 922.0), (0.90, 850.0),
                       (0.15, 1000.0)):
            add(aD, zeta, fn)
    # --- systematic low-fn coverage ---
    for fn in (700.0, 800.0, 850.0):
        for aD, zeta in ((0.20, 0.011), (0.35, 0.020), (0.75, 0.030),
                         (1.00, 0.015)):
            add(aD, zeta, fn)
    # --- in-domain diversity draws ---
    extra = [
        (0.15, 0.011, 922.0), (0.20, 0.011, 922.0), (0.20, 0.020, 922.0),
        (0.35, 0.015, 922.0), (0.35, 0.025, 922.0), (0.60, 0.015, 922.0),
        (0.75, 0.011, 922.0), (0.90, 0.020, 922.0), (1.00, 0.020, 922.0),
        (0.25, 0.011, 1000.0), (0.60, 0.011, 1000.0), (0.90, 0.011, 1000.0),
        (0.25, 0.011, 850.0), (0.60, 0.011, 850.0), (0.90, 0.011, 850.0),
        (0.50, 0.020, 850.0), (0.50, 0.020, 1000.0), (0.50, 0.030, 922.0),
        (0.50, 0.015, 1000.0), (0.50, 0.025, 850.0),
        (0.15, 0.005, 922.0), (0.10, 0.005, 922.0), (0.20, 0.008, 850.0),
        (0.35, 0.005, 800.0), (0.75, 0.008, 700.0), (1.00, 0.005, 800.0),
        (0.50, 0.005, 800.0), (0.50, 0.005, 700.0), (0.50, 0.008, 850.0),
        (0.10, 0.008, 922.0), (0.15, 0.008, 922.0), (0.08, 0.011, 922.0),
        (0.05, 0.011, 922.0), (0.05, 0.020, 922.0), (0.10, 0.020, 922.0),
        (0.08, 0.015, 1000.0), (0.05, 0.008, 850.0), (0.15, 0.030, 922.0),
    ]
    for a in extra:
        add(*a)

    # --- split ---
    # keep original test anchors first
    test_anchor = {(0.10, 0.011, 922.0), (0.50, 0.005, 922.0),
                   (0.50, 0.011, 800.0)}
    test = []
    for c in cases:
        key = (round(c["aD"], 4), round(c["zeta"], 4), round(c["fn"], 1))
        if key in test_anchor:
            test.append(c)
    # add extrapolation cases with per-region caps for diversity
    low_ad = [c for c in cases if c not in test and c["aD"] < 0.15]
    low_zeta = [c for c in cases if c not in test and c["zeta"] < 0.008]
    low_fn = [c for c in cases if c not in test and c["fn"] < 850.0]
    # stable deterministic picks (first occurrences in case order)
    test.extend(low_ad[:3])
    test.extend(low_zeta[:2])
    test.extend(low_fn[:2])
    # validation: in-domain draws not in test
    val = []
    for c in cases:
        if len(val) >= 10:
            break
        if c in test or c in val:
            continue
        val.append(c)
    train = [c for c in cases if c not in test and c not in val]

    return {
        "n_cases": len(cases),
        "cases": cases,
        "split": {"train": train, "val": val, "test": test},
        "grids": {
            "fine": {"n_rpm": [4000.0, 16000.0, 80],
                     "a_p_mm": [0.05, 1.5, 128], "m": 80},
            "fine160": {"n_rpm": [4000.0, 16000.0, 80],
                        "a_p_mm": [0.05, 1.5, 128], "m": 160},
            "c32": {"n_rpm": [4000.0, 16000.0, 20],
                    "a_p_mm": [0.05, 1.5, 32], "m": 40},
            "c16": {"n_rpm": [4000.0, 16000.0, 10],
                    "a_p_mm": [0.05, 1.5, 16], "m": 20},
        },
        "license": "CC-BY-4.0",
        "version": "1.0",
    }


def main() -> None:
    meta = build_cases()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "benchmark", "meta.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("n_cases:", meta["n_cases"])
    print("split:", {k: len(v) for k, v in meta["split"].items()})
    print("saved", out)


if __name__ == "__main__":
    main()
