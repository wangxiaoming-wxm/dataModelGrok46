#!/usr/bin/env python3
"""Run cheap reverse-engineering experiments 1,2,2b,3,4,6,7 then 5,8a. 8b is separate (slow)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ORDER = [
    "exp7_power.py",
    "exp4_rules.py",
    "exp1_persource.py",
    "exp3_freq.py",
    "exp6_resid.py",
    "exp2b_kernel.py",
    "exp2_hist2d.py",
    "exp5_mlp.py",
    "exp8a_portable.py",
]


def main():
    for name in ORDER:
        print("\n" + "=" * 70 + f"\nRUN {name}\n" + "=" * 70, flush=True)
        runpy.run_path(str(HERE / name), run_name="__main__")


if __name__ == "__main__":
    main()
