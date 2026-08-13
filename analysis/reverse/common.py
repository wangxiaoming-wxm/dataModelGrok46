"""Shared honest-CV utilities for reverse-engineering the claim generating process."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path("/workspace/data")
OUT = Path("/workspace/analysis/reverse")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_FOLDS = 10


def load_train() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA / "train.csv")
    y = df["label"].astype(np.int32).to_numpy()
    return df, y


def auc(y, s) -> float:
    s = np.asarray(s, dtype=float)
    if np.isnan(s).any():
        s = np.where(np.isnan(s), np.nanmedian(s), s)
    if np.unique(s).size < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def both_auc(y, s) -> float:
    a = auc(y, s)
    b = auc(y, -np.asarray(s, dtype=float))
    return max(a, b)


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(pct=True).to_numpy()


def skf(y, n=N_FOLDS, seed=SEED):
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)


def fill_condition(trn: pd.DataFrame, *others: pd.DataFrame):
    """Source-median fill, fit on trn only. Returns list of filled series."""
    med = trn.groupby("source")["condition"].median()
    glob = float(trn["condition"].median())

    def _fill(df):
        return df["condition"].fillna(df["source"].map(med)).fillna(glob).astype(float)

    return [_fill(trn)] + [_fill(df) for df in others]


def source_median_map(trn_src, trn_cond) -> pd.Series:
    tmp = pd.DataFrame({"source": np.asarray(trn_src), "c": np.asarray(trn_cond)})
    return tmp.groupby("source")["c"].median()


def cond_r_of(src, cond, med_map: pd.Series) -> np.ndarray:
    med = pd.Series(np.asarray(src)).map(med_map).astype(float).to_numpy()
    med = np.where(np.isnan(med) | (med == 0), float(np.nanmedian(med_map.to_numpy())), med)
    return np.asarray(cond, dtype=float) / med


def rank_vs_ref(values, ref) -> np.ndarray:
    ref = np.sort(np.asarray(ref, dtype=float))
    if len(ref) == 0:
        return np.full(len(values), 0.5)
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def per_source_rank(src, cond, ref_src, ref_cond) -> np.ndarray:
    src = np.asarray(src)
    cond = np.asarray(cond, dtype=float)
    ref_src = np.asarray(ref_src)
    ref_cond = np.asarray(ref_cond, dtype=float)
    out = np.full(len(src), 0.5)
    for s in np.unique(src):
        m = src == s
        ref = ref_cond[ref_src == s]
        out[m] = rank_vs_ref(cond[m], ref if len(ref) else ref_cond)
    return out


def qcut_apply(tr, va, q=10):
    cats, bins = pd.qcut(tr, q, duplicates="drop", retbins=True, labels=False)
    bins = bins.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    va_b = pd.cut(va, bins=bins, labels=False, include_lowest=True)
    va_b = pd.Series(va_b).fillna(0).astype(int).to_numpy()
    return cats.astype(int).to_numpy(), va_b, bins


def te_stats(keys, y, m=20.0, prior=None):
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    if prior is None:
        prior = float(y.mean())
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    stats = {k: (float(r["sum"]), int(r["count"])) for k, r in g.iterrows()}
    return stats, prior, float(m)


def te_loo_from_stats(keys, y, stats, prior, m):
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    sm = np.fromiter((stats[k][0] if k in stats else prior * 0 for k in keys), dtype=float, count=len(keys))
    ct = np.fromiter((stats[k][1] if k in stats else 0 for k in keys), dtype=float, count=len(keys))
    return (sm - y + prior * m) / np.clip(ct - 1.0 + m, 1e-6, None)


def te_apply_from_stats(keys, stats, prior, m):
    keys = np.asarray(keys)
    out = np.empty(len(keys), dtype=float)
    for i, k in enumerate(keys):
        if k in stats:
            sm, ct = stats[k]
            out[i] = (sm + prior * m) / (ct + m)
        else:
            out[i] = prior
    return out


def te_pair(tr_keys, va_keys, ytr, m=20.0):
    stats, prior, m = te_stats(tr_keys, ytr, m=m)
    tr_enc = te_loo_from_stats(tr_keys, ytr, stats, prior, m)
    va_enc = te_apply_from_stats(va_keys, stats, prior, m)
    return tr_enc, va_enc


def te_val_only(tr_keys, va_keys, ytr, m=20.0):
    stats, prior, m = te_stats(tr_keys, ytr, m=m)
    return te_apply_from_stats(va_keys, stats, prior, m)


def dump_json(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return p


def windows_of(days: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(days, dtype=float)
    return {
        "w_new50": (d < 50).astype(float),
        "w_new200": (d < 200).astype(float),
        "w_safe750": ((d >= 700) & (d < 880)).astype(float),
        "w_safe1750": ((d >= 1725) & (d < 1825)).astype(float),
        "w_hot1950": ((d >= 1950) & (d < 2000)).astype(float),
        "w_old10k": (d >= 10000).astype(float),
    }


def numeric_block(df, cond, src, trn_src, trn_cond, extra: dict | None = None) -> pd.DataFrame:
    """Spark-portable numeric generating-process features. Stats from train fold."""
    days = df["days"].to_numpy(float)
    cond = np.asarray(cond, dtype=float)
    src = np.asarray(src)
    med_map = source_median_map(trn_src, trn_cond)
    cr = cond_r_of(src, cond, med_map)
    rk = per_source_rank(src, cond, trn_src, trn_cond)
    cond_clip = np.clip(cond, 1e-6, None)
    out = {
        "days": days,
        "days_log": np.log1p(days),
        "days_sqrt": np.sqrt(np.clip(days, 0, None)),
        "days2": (days / 5000.0) ** 2,
        "condition_f": cond,
        "cond_log": np.log(cond_clip),
        "cond_sqrt": np.sqrt(cond_clip),
        "inv_cond": 1.0 / np.clip(cond, 1e-4, None),
        "inv_sqrt_cond": 1.0 / np.sqrt(cond_clip),
        "cond_r": cr,
        "cond_rk": rk,
        "u_shape": (rk - 0.5) ** 2,
        "ratio": days / np.clip(cr, 1e-6, None),
        "rate": days * (1.0 - rk),
        "ratio_sqrt": days / np.sqrt(cond_clip),
        "log_ratio": np.log(np.clip(days, 1, None)) - 0.5 * np.log(cond_clip),
        "days_x_inv": days * (1.0 / np.clip(cond, 1e-4, None)),
        "age_range": df["age_range"].to_numpy(float),
        "age8": (df["age_range"].to_numpy() >= 8).astype(float),
        "cond_low": (cond < 0.05).astype(float),
        "cond_miss": df["condition"].isna().astype(float),
    }
    out.update(windows_of(days))
    car = pd.Series(src).str.split("|").str[0].to_numpy()
    out["ushape_car10"] = out["u_shape"] * (car == "CAR_10").astype(float)
    out["ushape_car0"] = out["u_shape"] * (car == "CAR_0").astype(float)
    out["mono_car1"] = (1.0 - rk) * (car == "CAR_1").astype(float)
    out["rev_car7"] = rk * (car == "CAR_7").astype(float)
    if "V" in df.columns:
        out["V"] = df["V"].to_numpy(float)
        out["cc"] = df["cc"].to_numpy(float)
        out["max_g"] = df["max_g"].to_numpy(float)
        out["x20"] = df["x20"].to_numpy(float)
        out["x1"] = df["x1"].to_numpy(float)
        out["x5"] = df["x5"].to_numpy(float)
        out["x14"] = df["x14"].to_numpy(float)
        out["x17"] = df["x17"].to_numpy(float)
        t3 = df["t3"].astype(str)
        out["t3_num"] = t3.map(lambda s: float(s[:-1]) if s[-1].isalpha() else np.nan).to_numpy()
        med_t3 = np.nanmedian(out["t3_num"])
        out["t3_num"] = np.where(np.isnan(out["t3_num"]), med_t3, out["t3_num"])
    if extra:
        out.update(extra)
    return pd.DataFrame(out, index=df.index)


def save_oof(name: str, arr: np.ndarray):
    np.save(OUT / name, arr)
