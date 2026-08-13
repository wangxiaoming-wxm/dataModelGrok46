"""Fold-safe target encoding.

Encodings for the rows a model is fitted on come from an inner K-fold split, so
no row ever sees its own label.  Rows outside the fit set (validation / test)
use statistics pooled over the whole fit set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _stats(codes: np.ndarray, y: np.ndarray, n_cat: int) -> tuple[np.ndarray, np.ndarray]:
    cnt = np.bincount(codes, minlength=n_cat).astype(float)
    pos = np.bincount(codes, weights=y, minlength=n_cat)
    return pos, cnt


def encode(
    col_fit: pd.Series,
    y_fit: np.ndarray,
    others: list[pd.Series],
    smoothing: float = 30.0,
    inner_splits: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (encoding for the fit rows, encodings for each frame in `others`)."""
    cats = pd.Index(pd.unique(pd.concat([col_fit] + others).astype(str)))
    lut = {v: i for i, v in enumerate(cats)}
    n_cat = len(cats)
    cf = col_fit.astype(str).map(lut).to_numpy()
    prior = float(y_fit.mean())

    enc_fit = np.full(len(cf), prior)
    skf = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    for ii, oi in skf.split(cf, y_fit):
        pos, cnt = _stats(cf[ii], y_fit[ii], n_cat)
        val = (pos + smoothing * prior) / (cnt + smoothing)
        enc_fit[oi] = val[cf[oi]]

    pos, cnt = _stats(cf, y_fit, n_cat)
    full = (pos + smoothing * prior) / (cnt + smoothing)
    outs = [full[o.astype(str).map(lut).to_numpy()] for o in others]
    return enc_fit, outs


def count_encode(col_all: pd.Series) -> pd.Series:
    """Frequency over every available row - label-free, so leakage-free."""
    return col_all.astype(str).map(col_all.astype(str).value_counts())
