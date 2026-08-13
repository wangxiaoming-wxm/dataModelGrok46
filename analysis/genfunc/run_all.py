#!/usr/bin/env python3
"""Run genfunc experiments 1→5 then portable_score. CPU only, no early stopping."""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "exp1_spline.py",
    "exp2_car_shapes.py",
    "exp3_2d_smooth.py",
    "exp4_grid_closed.py",
    "exp5_fuse.py",
    "portable_score.py",
]


def main():
    sys.path.insert(0, str(HERE))
    for name in SCRIPTS:
        t0 = time.time()
        print(f"\n########## {name} ##########", flush=True)
        runpy.run_path(str(HERE / name), run_name="__main__")
        print(f"########## {name} done in {time.time() - t0:.1f}s ##########", flush=True)


if __name__ == "__main__":
    main()
