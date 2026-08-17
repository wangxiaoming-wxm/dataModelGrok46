#!/usr/bin/env python3
"""Keep submissions/submission.csv on the strongest assemble_best recipe.

The in-memory final_best.py run still writes VAL-ES over submission.csv when it
finishes. This guard restores via assemble_best.py whenever the on-disk report
drops below the locked opus5 gated score, and always reassembles after the
trainer exits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace") if Path("/workspace/data/train.csv").is_file() else Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
REPORT = SUB / "final_best_report.json"
BACKUP = SUB / "submission_opus5_gated.csv"
ASSEMBLE = ROOT / "analysis" / "assemble_best.py"
LOCK_GATED = 0.70335
TRAIN_PID = int(os.environ.get("GUARD_TRAIN_PID", "133208"))


def gated() -> float:
    if not REPORT.is_file():
        return -1.0
    try:
        return float(json.loads(REPORT.read_text(encoding="utf-8")).get("auc_gated") or -1.0)
    except Exception:
        return -1.0


def selected() -> str:
    if not REPORT.is_file():
        return ""
    try:
        return str(json.loads(REPORT.read_text(encoding="utf-8")).get("selected") or "")
    except Exception:
        return ""


def restore(reason: str) -> None:
    print(f"[guard] restore: {reason} gated={gated():.6f} selected={selected()}", flush=True)
    subprocess.check_call([sys.executable, "-u", str(ASSEMBLE)], cwd=str(ROOT))
    print(f"[guard] after assemble gated={gated():.6f} selected={selected()}", flush=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    if not BACKUP.is_file():
        print("[guard] missing backup", flush=True)
    last = gated()
    print(f"[guard] start pid={TRAIN_PID} alive={alive(TRAIN_PID)} gated={last:.6f} selected={selected()}", flush=True)
    while alive(TRAIN_PID):
        g = gated()
        sel = selected()
        if g + 1e-8 < LOCK_GATED or (sel and "opus" not in sel and g < LOCK_GATED - 1e-6):
            restore(f"drop to {g:.6f} sel={sel}")
        time.sleep(20)
    restore("trainer exited")
    print("[guard] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
