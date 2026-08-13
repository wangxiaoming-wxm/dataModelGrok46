"""Honest-CV utilities for closed generating-function search.

Protocol: StratifiedKFold, fold-internal fit only. No full-data TE, no test labels, no id.
Identity-link RMSE (Ridge / squared error). Never logloss as the training loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = Path("/workspace/data")
OUT = Path("/workspace/analysis/genfunc")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
PRIOR_FALLBACK = 0.1002

# Underwriting / warranty windows from the reverse-engineering notes.
WINDOW_SPEC = [
    ("w_new50", 0.0, 50.0),
    ("w_new200", 0.0, 200.0),
    ("w_safe750", 700.0, 880.0),
    ("w_safe1750", 1725.0, 1825.0),
    ("w_hot1950", 1950.0, 2000.0),
    ("w_hot9370", 9370.0, 9475.0),
]

# Fold-stable default exponents for days / cond^{b_s} (overridden fold-internally).
DEFAULT_B = {
    "CAR_1|ENG_591": 1.5,
    "CAR_10|ENG_651": 1.2,
    "CAR_0|ENG_709": 0.3,
    "CAR_2|ENG_262": 0.4,
    "CAR_5|ENG_062": 0.1,
    "CAR_7|ENG_966": 0.1,
    "CAR_9|ENG_415": 0.1,
    "CAR_4|ENG_153": 0.8,
    "CAR_6|ENG_844": 0.8,
    "CAR_3|ENG_377": 0.5,
    "CAR_8|ENG_228": 0.5,
}


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
    s = np.asarray(s, dtype=float)
    return max(auc(y, s), auc(y, -s))


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(a, dtype=float)).rank(pct=True).to_numpy()


def skf_splits(y, n: int, seed: int = SEED):
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    dummy = np.zeros(len(y))
    return list(cv.split(dummy, y))


def dump_json(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))
    return p


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_oof(name: str, arr: np.ndarray) -> Path:
    p = OUT / name
    np.save(p, np.asarray(arr, dtype=np.float64))
    return p


def fill_condition(trn: pd.DataFrame, *others: pd.DataFrame):
    med = trn.groupby("source")["condition"].median()
    glob = float(trn["condition"].median())

    def _fill(df):
        return df["condition"].fillna(df["source"].map(med)).fillna(glob).to_numpy(dtype=float)

    return [_fill(trn)] + [_fill(df) for df in others]


def car_of(src) -> np.ndarray:
    return pd.Series(np.asarray(src).astype(str)).str.split("|").str[0].to_numpy()


def rank_vs_ref(values, ref) -> np.ndarray:
    ref = np.sort(np.asarray(ref, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(ref) == 0:
        return np.full(len(values), 0.5)
    return np.searchsorted(ref, values, side="right") / float(len(ref))


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


def source_median_map(trn_src, trn_cond) -> dict:
    tmp = pd.DataFrame({"source": np.asarray(trn_src).astype(str), "c": np.asarray(trn_cond, dtype=float)})
    return tmp.groupby("source")["c"].median().to_dict()


def cond_r_of(src, cond, med_map: dict) -> np.ndarray:
    src = np.asarray(src).astype(str)
    cond = np.asarray(cond, dtype=float)
    fallback = float(np.nanmedian(list(med_map.values()))) if med_map else 1.0
    med = np.array([med_map.get(s, fallback) for s in src], dtype=float)
    med = np.where((med == 0) | np.isnan(med), fallback, med)
    return cond / med


def qcut_apply(tr, va, q=10):
    tr = np.asarray(tr, dtype=float)
    va = np.asarray(va, dtype=float)
    cats, bins = pd.qcut(tr, q, duplicates="drop", retbins=True, labels=False)
    bins = np.asarray(bins, dtype=float).copy()
    bins[0], bins[-1] = -np.inf, np.inf
    va_b = pd.cut(va, bins=bins, labels=False, include_lowest=True)
    va_b = pd.Series(va_b).fillna(0).astype(int).to_numpy()
    return np.asarray(cats, dtype=int), va_b, bins


def apply_bins(x, bins) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    b = np.asarray(bins, dtype=float).copy()
    b[0], b[-1] = -np.inf, np.inf
    out = pd.cut(x, bins=b, labels=False, include_lowest=True)
    return pd.Series(out).fillna(0).astype(int).to_numpy()


def windows_of(days: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(days, dtype=float)
    out = {}
    for name, lo, hi in WINDOW_SPEC:
        out[name] = ((d >= lo) & (d < hi)).astype(float)
    out["w_old10k"] = (d >= 10000.0).astype(float)
    return out


def window_matrix(days: np.ndarray) -> tuple[np.ndarray, list[str]]:
    w = windows_of(days)
    names = list(w.keys())
    return np.column_stack([w[k] for k in names]), names


def quantile_knots(x, n_seg: int) -> np.ndarray:
    """n_seg pieces => n_seg+1 knots at empirical quantiles (train only)."""
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0.0, 1.0, n_seg + 1)
    knots = np.quantile(x, qs)
    # de-duplicate (constant columns otherwise)
    knots = np.unique(knots)
    if len(knots) < 3:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0
        knots = np.linspace(lo, hi, max(n_seg + 1, 3))
    return knots.astype(float)


def pw_linear_basis(x, knots) -> np.ndarray:
    """Continuous piecewise-linear basis: [1, x] + (x-k)_+ for interior knots."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    interior = knots[1:-1] if len(knots) >= 3 else knots
    cols = [np.ones(len(x)), x]
    for k in interior:
        cols.append(np.maximum(x - k, 0.0))
    return np.column_stack(cols)


def te_group(keys, y) -> pd.DataFrame:
    return pd.DataFrame({"k": np.asarray(keys), "y": np.asarray(y, dtype=float)}).groupby("k")["y"].agg(["sum", "count"])


def te_apply_from_group(keys, g: pd.DataFrame, prior: float, m: float) -> np.ndarray:
    keys = np.asarray(keys)
    if len(g) == 0:
        return np.full(len(keys), prior)
    pmap = ((g["sum"] + prior * m) / (g["count"] + m)).to_dict()
    out = pd.Series(keys).map(pmap)
    return out.fillna(prior).to_numpy(dtype=float)


def te_loo_from_group(keys, y, g: pd.DataFrame, prior: float, m: float) -> np.ndarray:
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    sm = pd.Series(keys).map(g["sum"]).fillna(0.0).to_numpy(dtype=float)
    ct = pd.Series(keys).map(g["count"]).fillna(0.0).to_numpy(dtype=float)
    return (sm - y + prior * m) / np.clip(ct - 1.0 + m, 1e-6, None)


def te_pair(tr_keys, va_keys, ytr, m=20.0):
    ytr = np.asarray(ytr, dtype=float)
    prior = float(ytr.mean()) if len(ytr) else PRIOR_FALLBACK
    g = te_group(tr_keys, ytr)
    tr_enc = te_loo_from_group(tr_keys, ytr, g, prior, m)
    va_enc = te_apply_from_group(va_keys, g, prior, m)
    return tr_enc, va_enc, g, prior


def te_val_only(tr_keys, va_keys, ytr, m=20.0):
    ytr = np.asarray(ytr, dtype=float)
    prior = float(ytr.mean()) if len(ytr) else PRIOR_FALLBACK
    g = te_group(tr_keys, ytr)
    return te_apply_from_group(va_keys, g, prior, m)


def source_b_map(src, days, cond, y, bs=None) -> dict:
    """Per-source exponent b maximizing bidirectional AUC of days/cond^b on TRAIN only."""
    if bs is None:
        bs = np.round(np.linspace(0.1, 1.5, 8), 2)
    out = {}
    cond = np.clip(np.asarray(cond, dtype=float), 1e-6, None)
    days = np.asarray(days, dtype=float)
    src = np.asarray(src).astype(str)
    y = np.asarray(y)
    for s in np.unique(src):
        msk = src == s
        if msk.sum() < 40:
            out[s] = float(DEFAULT_B.get(s, 0.5))
            continue
        ys = y[msk]
        best_b, best_a = float(DEFAULT_B.get(s, 0.5)), -1.0
        d = days[msk]
        c = cond[msk]
        for b in bs:
            sc = d / np.power(c, b)
            a = both_auc(ys, sc)
            if a > best_a:
                best_a, best_b = a, float(b)
        out[s] = best_b
    return out


def power_score(days, cond, rk, src, a, c, bmap) -> np.ndarray:
    days = np.clip(np.asarray(days, dtype=float), 1e-6, None)
    cond = np.clip(np.asarray(cond, dtype=float), 1e-6, None)
    rk = np.clip(np.asarray(rk, dtype=float), 0.0, 1.0)
    src = np.asarray(src).astype(str)
    bvec = np.array([bmap.get(s, 0.5) for s in src], dtype=float)
    return np.power(days, a) / np.power(cond, bvec) * np.power(np.clip(1.0 - rk, 1e-6, None), c)


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def standardize_apply(X: np.ndarray, mu, sd) -> np.ndarray:
    return (X - mu) / sd


def ohe_levels_apply(values, levels) -> np.ndarray:
    values = np.asarray(values).astype(str)
    return np.column_stack([(values == lv).astype(float) for lv in levels]) if levels else np.zeros((len(values), 0))


def per_source_auc(y, s, src) -> dict:
    out = {}
    src = np.asarray(src).astype(str)
    for u in sorted(np.unique(src)):
        m = src == u
        if m.sum() < 20 or np.unique(y[m]).size < 2:
            out[u] = {"n": int(m.sum()), "auc": None, "rate": float(y[m].mean())}
        else:
            out[u] = {"n": int(m.sum()), "auc": auc(y[m], s[m]), "rate": float(y[m].mean())}
    return out


def fold_report(y, oof, src, per_fold) -> dict:
    return {
        "auc": auc(y, oof),
        "per_fold": per_fold,
        "per_source": per_source_auc(y, oof, src),
        "mean": float(np.mean(oof)),
        "std": float(np.std(oof)),
        "clip01_auc": auc(y, np.clip(oof, 0, 1)),
    }
