"""Fold-safe dual-world features for CatBoost teacher / Scala helper.

Never emit src|cond_q|days_q as a CatBoost categorical (high-card, empirically hurts).
Medium-card 2-way crosses are OK. 3-way keys are TE-only scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_MAIN = [
    "days",
    "days_log",
    "condition_f",
    "cond_r",
    "ratio",
    "ratio_sqrt",
    "u_shape",
    "inv_cond",
    "age_num",
    "V",
    "cc",
    "x20",
    "x1",
    "x5",
    "max_g",
    "x14",
    "x17",
    "age8",
    "cond_low",
    "safe_750",
    "safe_1750",
    "safe_1950",
    "days_lt50",
    "t3_num",
    "cond_miss",
]

NUM_ALT = [
    "days",
    "days_log",
    "condition_f",
    "cond_rk",
    "rate",
    "u_shape",
    "age_num",
    "V",
    "cc",
    "x20",
    "x1",
    "x5",
    "age8",
    "cond_low",
    "safe_750",
    "safe_1750",
    "safe_1950",
    "days_lt50",
    "t3_num",
    "cond_miss",
]

# Medium-cardinality cats for native CatBoost. NO src_cq_dq / 3-way.
CATS = [
    "source",
    "region",
    "age_range",
    "grades",
    "month",
    "src_reg",
    "src_age",
    "reg_age",
    "src_cq",
    "days_q",
    "cond_q",
    "src_dq",
    "reg_cq",
    "reg_dq",
]

HIGH_TE_KEYS = ["src_cq_dq", "cq_dq", "src_ratioq"]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    t3 = d["t3"].astype(str)
    d["t3_num"] = pd.to_numeric(t3.str.extract(r"^([0-9.]+)", expand=False), errors="coerce")
    d["cond_miss"] = d["condition"].isna().astype(np.int8)
    return d


def _qcut_apply(tr: pd.Series, va: pd.Series, q: int = 10) -> tuple[np.ndarray, np.ndarray]:
    cats, bins = pd.qcut(tr, q, duplicates="drop", retbins=True, labels=False)
    bins = bins.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    tr_q = cats.astype(int).astype(str).to_numpy()
    va_q = pd.cut(va, bins=bins, labels=False, include_lowest=True).fillna(0).astype(int).astype(str).to_numpy()
    return tr_q, va_q


def _rank_apply(tr_src: np.ndarray, tr_v: np.ndarray, va_src: np.ndarray, va_v: np.ndarray) -> np.ndarray:
    by = {}
    for s, v in zip(tr_src, tr_v):
        by.setdefault(s, []).append(v)
    for s in list(by):
        by[s] = np.sort(np.asarray(by[s], dtype=np.float64))
    out = np.empty(len(va_src), dtype=np.float64)
    for i, (s, v) in enumerate(zip(va_src, va_v)):
        ref = by.get(s)
        if ref is None or len(ref) == 0:
            out[i] = 0.5
        else:
            out[i] = float(np.searchsorted(ref, v, side="right")) / float(len(ref))
    return out


def fold_features(trn: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trn = enrich(trn)
    val = enrich(val)
    med = trn.groupby("source")["condition"].median()
    g = float(trn["condition"].median())

    def fill_cond(df: pd.DataFrame) -> pd.Series:
        return df["condition"].fillna(df["source"].map(med)).fillna(g)

    trn = trn.copy()
    val = val.copy()
    trn["condition_f"] = fill_cond(trn)
    val["condition_f"] = fill_cond(val)
    sm = trn.groupby("source")["condition_f"].median()
    sm_med = float(sm.median()) if len(sm) else 1.0

    def cond_r(df: pd.DataFrame) -> pd.Series:
        den = df["source"].map(sm).replace(0, np.nan).fillna(sm_med)
        return df["condition_f"] / den

    trn["cond_r"] = cond_r(trn)
    val["cond_r"] = cond_r(val)
    trn["cond_rk"] = trn.groupby("source")["condition_f"].rank(pct=True)
    val["cond_rk"] = _rank_apply(
        trn["source"].to_numpy(),
        trn["condition_f"].to_numpy(),
        val["source"].to_numpy(),
        val["condition_f"].to_numpy(),
    )

    for df in (trn, val):
        df["ratio"] = df["days"] / df["cond_r"]
        df["rate"] = df["days"] * (1.0 - df["cond_rk"])
        df["ratio_sqrt"] = df["days"] / np.sqrt(df["condition_f"].clip(lower=1e-6))
        df["u_shape"] = (df["cond_rk"] - 0.5) ** 2
        df["inv_cond"] = 1.0 / df["condition_f"].clip(lower=1e-4)
        df["days_log"] = np.log1p(df["days"])
        df["age_num"] = pd.to_numeric(df["age_range"], errors="coerce").fillna(0.0)
        df["age8"] = (df["age_num"] >= 8).astype(np.int8)
        df["cond_low"] = (df["condition_f"] < 0.05).astype(np.int8)
        df["safe_750"] = ((df["days"] >= 700) & (df["days"] < 880)).astype(np.int8)
        df["safe_1750"] = ((df["days"] >= 1725) & (df["days"] < 1825)).astype(np.int8)
        df["safe_1950"] = ((df["days"] >= 1950) & (df["days"] < 2000)).astype(np.int8)
        df["days_lt50"] = (df["days"] < 50).astype(np.int8)

    trn["days_q"], val["days_q"] = _qcut_apply(trn["days"], val["days"])
    trn["cond_q"], val["cond_q"] = _qcut_apply(trn["condition_f"], val["condition_f"])
    trn["ratio_q"], val["ratio_q"] = _qcut_apply(trn["ratio"], val["ratio"])
    trn["rate_q"], val["rate_q"] = _qcut_apply(trn["rate"], val["rate"])

    for df in (trn, val):
        df["source"] = df["source"].astype(str)
        df["region"] = df["region"].astype(str)
        df["age_range"] = df["age_range"].astype(int).astype(str)
        df["grades"] = df["grades"].astype(str)
        df["month"] = df["month"].astype(str)
        df["src_reg"] = df["source"] + "|" + df["region"]
        df["src_age"] = df["source"] + "|" + df["age_range"]
        df["reg_age"] = df["region"] + "|" + df["age_range"]
        df["src_cq"] = df["source"] + "|" + df["cond_q"]
        df["src_dq"] = df["source"] + "|" + df["days_q"]
        df["reg_cq"] = df["region"] + "|" + df["cond_q"]
        df["reg_dq"] = df["region"] + "|" + df["days_q"]
        df["cq_dq"] = df["cond_q"] + "|" + df["days_q"]
        df["src_ratioq"] = df["source"] + "|" + df["ratio_q"]
        df["src_cq_dq"] = df["source"] + "|" + df["cond_q"] + "|" + df["days_q"]
    return trn, val


def te_apply(tr_keys: np.ndarray, va_keys: np.ndarray, ytr: np.ndarray, m: float = 20.0) -> np.ndarray:
    prior = float(ytr.mean())
    ssum: dict[str, float] = {}
    scnt: dict[str, int] = {}
    for k, yi in zip(tr_keys, ytr):
        ssum[k] = ssum.get(k, 0.0) + float(yi)
        scnt[k] = scnt.get(k, 0) + 1
    out = np.empty(len(va_keys), dtype=np.float64)
    for i, k in enumerate(va_keys):
        if k in scnt:
            out[i] = (ssum[k] + prior * m) / (scnt[k] + m)
        else:
            out[i] = prior
    return out
