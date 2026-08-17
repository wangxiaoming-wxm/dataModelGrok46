"""80+ handmade categorical crosses matching historical W62 (~121 features).

Atomic cats (13): source, region, age_range, grades, month, version,
days_q5, cond_q10, ratio_q, rate_q, t3_letter, win, x20_q.

All pairwise concatenations: C(13,2)+13 = 91 cat columns.
Plus dual-world numerics (~30) ≈ 120 columns — the rsm=0.3 regime.

3-way src|cond_q|days_q is TE-only (never a CatBoost cat).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from cb_features import NUM_ALT, NUM_MAIN, _qcut_apply, fold_features

ATOMIC = [
    "source",
    "region",
    "age_range",
    "grades",
    "month",
    "version",
    "days_q5",
    "cond_q",
    "ratio_q",
    "rate_q",
    "t3_letter",
    "win",
    "x20_q",
]

PAIR_NAMES = [f"{a}__{b}" for a, b in itertools.combinations(ATOMIC, 2)]
CATS_ATOMIC = list(ATOMIC)
CATS_80 = CATS_ATOMIC + PAIR_NAMES  # 13 + 78 = 91

def pair_name(a: str, b: str) -> str:
    ia, ib = ATOMIC.index(a), ATOMIC.index(b)
    lo, hi = (a, b) if ia < ib else (b, a)
    return f"{lo}__{hi}"


SEM_PAIRS = [
    pair_name("source", "region"),
    pair_name("source", "age_range"),
    pair_name("source", "days_q5"),
    pair_name("source", "cond_q"),
    pair_name("source", "ratio_q"),
    pair_name("source", "x20_q"),
    pair_name("region", "age_range"),
    pair_name("region", "days_q5"),
    pair_name("region", "cond_q"),
    pair_name("region", "x20_q"),
    pair_name("age_range", "days_q5"),
    pair_name("age_range", "cond_q"),
    pair_name("days_q5", "cond_q"),
    pair_name("cond_q", "ratio_q"),
    pair_name("source", "win"),
    pair_name("region", "win"),
    pair_name("grades", "cond_q"),
    pair_name("month", "days_q5"),
    pair_name("version", "region"),
]
CATS_SEM = CATS_ATOMIC + SEM_PAIRS


def _win_bucket(days: np.ndarray) -> np.ndarray:
    d = np.asarray(days, dtype=float)
    out = np.full(len(d), "other", dtype=object)
    out[d < 50] = "lt50"
    out[(d >= 700) & (d < 880)] = "s750"
    out[(d >= 1725) & (d < 1825)] = "s1750"
    out[(d >= 1950) & (d < 2000)] = "s1950"
    out[(d >= 9370) & (d < 9475)] = "h9370"
    return out.astype(str)


def add_atomic_extras(trn: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trn = trn.copy()
    val = val.copy()
    for df in (trn, val):
        t3 = df["t3"].astype(str)
        let = t3.str.extract(r"([A-Za-z]+)", expand=False).fillna("NA")
        df["t3_letter"] = let.astype(str)
        df["version"] = df["version"].astype(str)
        df["win"] = _win_bucket(df["days"].to_numpy(float))
        df["car"] = df["source"].astype(str).str.split("|").str[0]
    trn["x20_q"], val["x20_q"] = _qcut_apply(trn["x20"], val["x20"], q=10)
    return trn, val


def add_pairs(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    df = df.copy()
    for a in ATOMIC:
        df[a] = df[a].astype(str)
    for name in names:
        if "__" not in name:
            continue
        a, b = name.split("__", 1)
        df[name] = df[a].astype(str) + "|" + df[b].astype(str)
    return df


def fold_features_80(trn_raw: pd.DataFrame, val_raw: pd.DataFrame, which: str = "80"):
    """which: atomic | sem | 80"""
    trn, val = fold_features(trn_raw, val_raw)
    trn, val = add_atomic_extras(trn, val)
    if which == "atomic":
        extra = []
    elif which == "sem":
        extra = [c for c in CATS_SEM if "__" in c]
    else:
        extra = PAIR_NAMES
    trn = add_pairs(trn, extra)
    val = add_pairs(val, extra)
    if which == "atomic":
        cats = list(CATS_ATOMIC)
    elif which == "sem":
        cats = list(CATS_SEM)
    else:
        cats = list(CATS_80)
    return trn, val, cats


def te_many(tr_keys: np.ndarray, va_keys: np.ndarray, ytr: np.ndarray, m: float = 20.0) -> np.ndarray:
    prior = float(np.mean(ytr))
    tmp = pd.DataFrame({"k": np.asarray(tr_keys).astype(str), "y": np.asarray(ytr, dtype=float)})
    g = tmp.groupby("k")["y"].agg(["sum", "count"])
    mapped = pd.Series(np.asarray(va_keys).astype(str)).map((g["sum"] + prior * m) / (g["count"] + m))
    return mapped.fillna(prior).to_numpy(dtype=np.float64)


def loo_te(keys: np.ndarray, y: np.ndarray, m: float = 20.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    prior = float(np.mean(y))
    tmp = pd.DataFrame({"k": np.asarray(keys).astype(str), "y": y})
    g = tmp.groupby("k")["y"].agg(["sum", "count"])
    s = tmp["k"].map(g["sum"]).to_numpy(dtype=np.float64)
    c = tmp["k"].map(g["count"]).to_numpy(dtype=np.float64)
    return (s - y + prior * m) / np.maximum(c - 1.0 + m, 1e-6)


__all__ = [
    "ATOMIC",
    "CATS_80",
    "CATS_ATOMIC",
    "CATS_SEM",
    "NUM_ALT",
    "NUM_MAIN",
    "fold_features_80",
    "loo_te",
    "te_many",
]
