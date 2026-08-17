"""Deterministic jittered re-encodings of the informative columns.

The organisers' redundant columns (``x19``/``x20``/``t3``/``livability``) are
noisy copies of ``source``/``condition``/``region``.  They carry no signal of
their own, yet adding them lifts CatBoost by ~0.010 AUC: each one is a
differently-perturbed discretisation of the same interaction, so the ensemble
of trees effectively averages several target statistics instead of trusting
one.  This module reproduces that mechanism on purpose and under our control.

The jitter is derived from the row ``id`` hash, so it is identical for a row no
matter which frame or fold it appears in, and it never touches the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MASK = (1 << 64) - 1
_PRIMES = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0xBF58476D1CE4E5B9,
           0x94D049BB133111EB, 0x2545F4914F6CDD1D, 0xD6E8FEB86659FD93,
           0xA24BAED4963EE407, 0x9FB21C651E98DF25, 0xFF51AFD7ED558CCD,
           0xC4CEB9FE1A85EC53, 0x87C37B91114253D5, 0x4CF5AD432745937F,
           0xB492B66FBE98F273, 0x9AE16A3B2F90404F, 0xCC9E2D51C2B2AE35,
           0x1B873593D3B4A5C7, 0xE6546B64F7A2B1D9, 0x85EBCA6B7FEB352D)


def row_uniform(ids: pd.Series, stream: int) -> np.ndarray:
    """A stable Uniform[0,1) per row, independent across ``stream``."""
    h = ids.apply(lambda s: int(s, 16) & _MASK).to_numpy(dtype=object)
    p = _PRIMES[stream % len(_PRIMES)]
    mixed = np.array([((int(v) ^ p) * p) & _MASK for v in h], dtype=object)
    mixed = np.array([(int(v) ^ (int(v) >> 29)) & _MASK for v in mixed], dtype=object)
    return np.array([int(v) / 2.0**64 for v in mixed], dtype=float)


def add_jitter_views(
    out: pd.DataFrame,
    cats: list[str],
    df: pd.DataFrame,
    cond_r: pd.Series,
    days: pd.Series,
    n_views: int = 3,
    n_bins: int = 10,
    n_sub: int = 8,
    stream_offset: int = 0,
) -> None:
    """Add ``n_views`` perturbed encodings of the condition/days/source signals.

    ``stream_offset`` selects a different family of perturbations, so each model
    in a bagged ensemble can be shown its own re-encoding of the same data.
    """
    ids = df["id"]
    cr = cond_r.to_numpy(dtype=float)
    dv = days.to_numpy(dtype=float)
    cr_scale = float(np.nanstd(cr))
    for k in range(n_views):
        base = 3 * (k + n_views * stream_offset)
        u_c = row_uniform(ids, base)
        u_d = row_uniform(ids, base + 1)
        u_s = row_uniform(ids, base + 2)

        cj = cr + (u_c - 0.5) * 3.0 * cr_scale
        dj = dv * (1.0 + (u_d - 0.5) * 0.30)
        cj_b = pd.qcut(pd.Series(cj), n_bins, labels=False, duplicates="drop").astype(str)
        dj_b = pd.qcut(pd.Series(dj), n_bins, labels=False, duplicates="drop").astype(str)
        sub = pd.Series((u_s * n_sub).astype(int).astype(str), index=out.index)

        cj_b.index = out.index
        dj_b.index = out.index
        out[f"J{k}_cj"] = cj_b
        out[f"J{k}_dj"] = dj_b
        out[f"J{k}_srcsub"] = out["source"].astype(str) + "#" + sub
        out[f"J{k}_cj_src"] = cj_b + "|" + out["source"].astype(str)
        out[f"J{k}_cj_reg"] = cj_b + "|" + out["region"].astype(str)
        out[f"J{k}_dj_reg"] = dj_b + "|" + out["region"].astype(str)
        out[f"J{k}_dj_src"] = dj_b + "|" + out["source"].astype(str)
        out[f"J{k}_cj_age"] = cj_b + "|" + out["age_cat"].astype(str)
        cats += [f"J{k}_cj", f"J{k}_dj", f"J{k}_srcsub", f"J{k}_cj_src",
                 f"J{k}_cj_reg", f"J{k}_dj_reg", f"J{k}_dj_src", f"J{k}_cj_age"]
