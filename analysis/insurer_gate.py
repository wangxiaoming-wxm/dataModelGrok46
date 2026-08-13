"""Frozen insurer-vs-customer score overlay.

Pre-registered from GENERATING_PROCESS.md (discovery+confirm), not from a
post-hoc grid on teacher OOF:

- Customer files when surplus = exposure * damage * (1 - deny) is high.
- Insurer does not pay in warranty/franchise pits:
    days in [1725, 1825)  — train n=103, 0 claims (confirm p=0.006)
    days in [700, 880)    — train rate 3.07% vs 10.02%
- End-of-cover dump: days in [9370, 9475) rate 18.9%

CatBoost still ranks the all-zero 5y pit too high (mean rank ~0.22).
Apply AFTER rank fusion. Magnitudes frozen (not nested-tuned):
  zero 1750;  rank -0.10 on 750;  rank +0.05 on hot.

Nested 10-fold (zero 1750 + fold-excess 750 + 0.05 hot) was 0.69310
vs teacher max2 0.69207. Do not re-search constants on full OOF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W750 = (700.0, 880.0)
W1750 = (1725.0, 1825.0)
WHOT = (9370.0, 9475.0)
SHIFT_750 = 0.10
SHIFT_HOT = 0.05


def _mask(days: np.ndarray, lo: float, hi: float) -> np.ndarray:
    d = np.asarray(days, dtype=np.float64)
    return (d >= lo) & (d < hi)


def apply_gate(score: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Rank-space overlay. All-zero pit goes strictly below every other row."""
    s = np.asarray(score, dtype=np.float64).copy()
    d = np.asarray(days, dtype=np.float64)
    finite = s[np.isfinite(s)]
    floor = (float(np.min(finite)) - 1.0) if finite.size else -1.0
    s[_mask(d, *W1750)] = floor
    s[_mask(d, *W750)] = s[_mask(d, *W750)] - SHIFT_750
    s[_mask(d, *WHOT)] = s[_mask(d, *WHOT)] + SHIFT_HOT
    return s


def attach_adv_columns(df: pd.DataFrame, cond_rk: np.ndarray | None = None) -> pd.DataFrame:
    """Deterministic customer/insurer columns (no fold stats except cond_rk)."""
    out = df
    days = out["days"].to_numpy(np.float64)
    deny = _mask(days, *W750) | _mask(days, *W1750)
    dump = _mask(days, *WHOT)
    out = out.copy()
    out["deny"] = deny.astype(np.float64)
    out["w_safe750"] = _mask(days, *W750).astype(np.float64)
    out["w_safe1750"] = _mask(days, *W1750).astype(np.float64)
    out["w_hot9370"] = dump.astype(np.float64)
    if cond_rk is not None:
        rk = np.asarray(cond_rk, dtype=np.float64)
        out["claim_util"] = out["days"].to_numpy(np.float64) * (1.0 - rk) * (1.0 - deny.astype(np.float64))
        out["dump_poor"] = dump.astype(np.float64) * (1.0 - rk)
    anniv = np.array([365.0, 730.0, 1095.0, 1460.0, 1825.0, 2190.0, 2555.0, 2920.0])
    out["anniv_dist"] = np.min(np.abs(days[:, None] - anniv[None, :]), axis=1)
    out["near_anniv"] = (out["anniv_dist"] < 40.0).astype(np.float64)
    return out
