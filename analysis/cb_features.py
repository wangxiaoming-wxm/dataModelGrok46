"""Fold-safe dual-world features for CatBoost teacher / Scala helper.

Never emit src|cond_q|days_q as a CatBoost categorical (high-card, empirically hurts).
Medium-card 2-way crosses are OK. 3-way keys are TE-only scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

B_S = {
    "CAR_1": 1.5,
    "CAR_10": 1.2,
    "CAR_0": 0.3,
    "CAR_2": 0.4,
    "CAR_5": 0.1,
    "CAR_7": 0.1,
    "CAR_9": 0.1,
    "CAR_4": 0.8,
    "CAR_6": 0.8,
}

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
    "w_hot9370",
    "t3_num",
    "cond_miss",
    "v_r",
    "cc_r",
    "maxg_r",
    "x14_r",
    "x17_r",
    "pow_ratio_s",
    "pow_rate15",
    "ushape_car10",
    "mono_car1",
    "rev_car7",
]

NUM_ALT = [
    "days",
    "days_log",
    "condition_f",
    "cond_rk",
    "rate",
    "pow_rate15",
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
    "w_9370",
    "days_lt50",
    "t3_num",
    "cond_miss",
]

# W62 numeric: dual-world + v6 ratios + power-rate + windows. No x18/x19/id.
NUM_MAIN_W62 = [
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
    "v_r",
    "cc_r",
    "maxg_r",
    "x14_r",
    "x17_r",
    "pow_rate15",
    "age8",
    "cond_low",
    "safe_750",
    "safe_1750",
    "w_9370",
    "days_lt50",
    "t3_num",
    "cond_miss",
]

NUM_ALT_W62 = [
    "days",
    "days_log",
    "condition_f",
    "cond_rk",
    "rate",
    "pow_rate15",
    "u_shape",
    "age_num",
    "V",
    "cc",
    "x20",
    "x1",
    "x5",
    "v_r",
    "cc_r",
    "maxg_r",
    "x14_r",
    "x17_r",
    "age8",
    "cond_low",
    "safe_750",
    "safe_1750",
    "w_9370",
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

# W62 cats: days qcut=5, cond qcut=10, 2-way only. 16 columns.
CATS_W62 = [
    "source",
    "region",
    "age_range",
    "grades",
    "month",
    "days_q5",
    "cond_q10",
    "ratio_q",
    "src_reg",
    "src_age",
    "reg_age",
    "src_cq",
    "src_dq5",
    "reg_cq",
    "reg_dq",
    "src_ratioq",
]

HIGH_TE_KEYS = ["src_cq_dq5", "cq_dq5", "src_ratioq", "src_cq_dq"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("/workspace/data/train.csv")
    test = pd.read_csv("/workspace/data/test.csv")
    return train, test


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


def fold_features(
    trn: pd.DataFrame,
    val: pd.DataFrame,
    tes: pd.DataFrame | None = None,
    cats: list[str] | None = None,
    nums_main: list[str] | None = None,
    nums_alt: list[str] | None = None,
):
    """Fold-safe features. Median/rank/qcut fit on `trn` only.

    Optional `tes` is transformed with the same train-fold stats.
    `cats` / `nums_*` are accepted for callers; the full feature set is always built.
    """
    _ = (cats, nums_main, nums_alt)
    trn = enrich(trn).copy()
    val = enrich(val).copy()
    frames: list[pd.DataFrame] = [trn, val]
    if tes is not None:
        tes = enrich(tes).copy()
        frames.append(tes)

    med = trn.groupby("source")["condition"].median()
    g = float(trn["condition"].median())

    def fill_cond(df: pd.DataFrame) -> pd.Series:
        return df["condition"].fillna(df["source"].map(med)).fillna(g)

    for df in frames:
        df["condition_f"] = fill_cond(df)
    sm = trn.groupby("source")["condition_f"].median()
    sm_med = float(sm.median()) if len(sm) else 1.0

    def cond_r(df: pd.DataFrame) -> pd.Series:
        den = df["source"].map(sm).replace(0, np.nan).fillna(sm_med)
        return df["condition_f"] / den

    for df in frames:
        df["cond_r"] = cond_r(df)
    trn["cond_rk"] = trn.groupby("source")["condition_f"].rank(pct=True)
    src_tr = trn["source"].to_numpy()
    cf_tr = trn["condition_f"].to_numpy()
    for df in frames[1:]:
        df["cond_rk"] = _rank_apply(
            src_tr, cf_tr, df["source"].to_numpy(), df["condition_f"].to_numpy()
        )

    for df in frames:
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
        df["w_9370"] = ((df["days"] >= 9370) & (df["days"] < 9475)).astype(np.int8)
        df["w_hot9370"] = df["w_9370"]
        df["days_lt50"] = (df["days"] < 50).astype(np.int8)
        cr = df["cond_r"].replace(0, np.nan)
        df["v_r"] = df["V"] / cr
        df["cc_r"] = df["cc"] / cr
        df["maxg_r"] = df["max_g"] / cr
        df["x14_r"] = df["x14"] / cr
        df["x17_r"] = df["x17"] / cr
        rk = df["cond_rk"].clip(0.0, 1.0)
        df["pow_rate15"] = np.power(df["days"].clip(lower=0.0), 1.5) * np.power((1.0 - rk).clip(lower=0.0), 1.23)
        car = df["source"].astype(str).str.split("|").str[0]
        b = car.map(B_S).fillna(0.5).astype(float)
        df["pow_ratio_s"] = df["days"] / np.power(df["condition_f"].clip(lower=1e-6), b)
        df["ushape_car10"] = df["u_shape"] * (car == "CAR_10").astype(np.int8)
        df["mono_car1"] = (1.0 - df["cond_rk"]) * (car == "CAR_1").astype(np.int8)
        df["rev_car7"] = df["cond_rk"] * (car == "CAR_7").astype(np.int8)

    def _qcut_all(name: str, col: str, q: int) -> None:
        trn[name], val[name] = _qcut_apply(trn[col], val[col], q=q)
        if tes is not None:
            _, tes[name] = _qcut_apply(trn[col], tes[col], q=q)

    _qcut_all("days_q", "days", 10)
    _qcut_all("days_q5", "days", 5)
    _qcut_all("cond_q", "condition_f", 10)
    _qcut_all("ratio_q", "ratio", 10)
    _qcut_all("rate_q", "rate", 10)
    for df in frames:
        df["cond_q10"] = df["cond_q"]

    for df in frames:
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
        df["src_dq5"] = df["source"] + "|" + df["days_q5"]
        df["reg_cq"] = df["region"] + "|" + df["cond_q"]
        df["reg_dq"] = df["region"] + "|" + df["days_q5"]
        df["cq_dq"] = df["cond_q"] + "|" + df["days_q"]
        df["cq_dq5"] = df["cond_q"] + "|" + df["days_q5"]
        df["src_ratioq"] = df["source"] + "|" + df["ratio_q"]
        df["src_cq_dq"] = df["source"] + "|" + df["cond_q"] + "|" + df["days_q"]
        df["src_cq_dq5"] = df["source"] + "|" + df["cond_q"] + "|" + df["days_q5"]
    if tes is None:
        return trn, val
    return trn, val, tes


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
