"""Model arms and the repeated-CV runner that produces their OOF / test scores.

Protocol
--------
* Repeated stratified 5-fold; every arm sees exactly the same partitions.
* Feature engineering never touches the label.  Quantile cut points, the
  per-source condition scale, frequency counts and the jitter streams are all
  label-free, so they are fitted once on train+test.
* Target encodings inside the LightGBM arm are produced by an inner K-fold over
  the fitting rows only, so no row ever sees its own label.
* No early stopping on the outer validation fold: the number of trees is fixed
  in advance for every arm.
* An arm's OOF is the rank-average over seeds; its test prediction is the
  rank-average over every (seed, fold) model.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

import lightgbm as lgb

from features import (BIN_COLS, _derive, add_noise_view, build, build_alt,
                      build_alt2, fit_edges, fit_edges_alt, fit_edges_alt2)
from jitter import add_jitter_views
from te import encode


# --------------------------------------------------------------------------
# feature frames
# --------------------------------------------------------------------------
def catboost_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 4):
    X, cats = build(raw, edges, "cross2")
    add_noise_view(X, cats, raw)
    der = _derive(raw, edges["__scale__"])
    add_jitter_views(X, cats, raw, der["cond_r"], pd.to_numeric(raw["days"]),
                     n_views=n_views, stream_offset=stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


def altboost_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    """Second encoding world: rank-normalised condition, different bin counts.

    Averaging over encodings is what buys accuracy on this dataset, so this view
    deliberately re-expresses the same information differently to decorrelate
    from the main CatBoost arm.
    """
    X, cats = build_alt(raw, edges)
    add_noise_view(X, cats, raw)
    rk = raw.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    add_jitter_views(X, cats, raw, rk, pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=13, stream_offset=50 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


def alt2_frame(raw: pd.DataFrame, edges: dict, stream_offset: int, n_views: int = 3):
    """Third encoding world: condition standardised inside region x source."""
    X, cats = build_alt2(raw, edges)
    add_noise_view(X, cats, raw)
    g = raw.groupby(["source", "region"])["condition"]
    cz = ((raw["condition"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    add_jitter_views(X, cats, raw, cz, pd.to_numeric(raw["days"]),
                     n_views=n_views, n_bins=7, stream_offset=100 + stream_offset)
    for c in cats:
        X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cats]
    X[num] = X[num].astype(float).fillna(-999.0)
    return X, cats


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
CAT_BASE = dict(loss_function="Logloss", learning_rate=0.03, l2_leaf_reg=10,
                random_strength=0.7, verbose=False, thread_count=4,
                allow_writing_files=False)

ARMS = {
    "cat_d5": dict(kind="cat", depth=5, iterations=1000, params={}),
    "cat_d6": dict(kind="cat", depth=6, iterations=700, params={"bagging_temperature": 1.0}),
    "cat_alt": dict(kind="cat", depth=6, iterations=800,
                    params={"l2_leaf_reg": 6, "one_hot_max_size": 12}),
    "cat_alt2": dict(kind="cat", depth=6, iterations=900,
                     params={"l2_leaf_reg": 14, "random_strength": 1.0}),
    "lgb_te": dict(kind="lgb"),
    "glm": dict(kind="glm"),
}

TE_COLS = ["region", "source", "age_cat", "bin_pat", "month", "version",
           "reg_src", "c10_src", "d10_reg", "d10_src", "r10_reg", "r10_src",
           "cr10_reg", "cr10_age", "d10_c10", "src_c10_age", "reg_c10_age",
           "d5_reg_src", "reg_age", "src_age", "d10_pat", "reg_pat", "dfx_src"]


# LightGBM has no ordered target statistics to protect it, so it only sees the
# columns we proved carry signal - the anonymisation-noise numerics are dropped.
LGB_DROP = ("cc", "max_g", "V", "x18")


def _lgb_matrix(Xf: pd.DataFrame, Xo: list[pd.DataFrame], y_fit: np.ndarray, seed: int):
    num = [c for c in Xf.columns
           if pd.api.types.is_numeric_dtype(Xf[c]) and c not in LGB_DROP]
    Af = Xf[num].to_numpy(dtype=float)
    Ao = [o[num].to_numpy(dtype=float) for o in Xo]
    fit_cols, other_cols = [], [[] for _ in Xo]
    for c in TE_COLS:
        ef, eo = encode(Xf[c], y_fit, [o[c] for o in Xo], smoothing=30.0, seed=seed)
        fit_cols.append(ef)
        for i, e in enumerate(eo):
            other_cols[i].append(e)
    Af = np.hstack([Af, np.array(fit_cols).T])
    Ao = [np.hstack([a, np.array(cols).T]) for a, cols in zip(Ao, other_cols)]
    return Af, Ao


def _glm_matrix(Xf: pd.DataFrame, Xo: list[pd.DataFrame], y_fit: np.ndarray, seed: int):
    """Splines on the continuous drivers + target-encoded segments."""
    cont = ["log_days", "log_cond_r", "log_ratio", "age_range", "bin_sum"]
    sp = SplineTransformer(n_knots=8, degree=3, include_bias=False)
    Sf = sp.fit_transform(Xf[cont].to_numpy(dtype=float))
    So = [sp.transform(o[cont].to_numpy(dtype=float)) for o in Xo]
    flags = Xf[BIN_COLS + ["condition_missing"]].to_numpy(dtype=float)
    flags_o = [o[BIN_COLS + ["condition_missing"]].to_numpy(dtype=float) for o in Xo]
    te_f, te_o = [], [[] for _ in Xo]
    for c in TE_COLS:
        ef, eo = encode(Xf[c], y_fit, [o[c] for o in Xo], smoothing=50.0, seed=seed)
        te_f.append(np.log(np.clip(ef, 1e-4, 1 - 1e-4) / (1 - np.clip(ef, 1e-4, 1 - 1e-4))))
        for i, e in enumerate(eo):
            p = np.clip(e, 1e-4, 1 - 1e-4)
            te_o[i].append(np.log(p / (1 - p)))
    Af = np.hstack([Sf, flags, np.array(te_f).T])
    Ao = [np.hstack([s, fl, np.array(t).T]) for s, fl, t in zip(So, flags_o, te_o)]
    sc = StandardScaler().fit(Af)
    return sc.transform(Af), [sc.transform(a) for a in Ao]


def fit_predict(name: str, Xtr, Xva, Xte, cats, y_fit, seed):
    spec = ARMS[name]
    if spec["kind"] == "cat":
        base = dict(CAT_BASE)
        base.update({k: v for k, v in spec["params"].items() if k in base})
        extra = {k: v for k, v in spec["params"].items() if k not in CAT_BASE}
        m = CatBoostClassifier(**base, depth=spec["depth"], iterations=spec["iterations"],
                               random_seed=seed, **extra)
        m.fit(Xtr, y_fit, cat_features=cats, verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    if spec["kind"] == "lgb":
        Af, (Av, At) = _lgb_matrix(Xtr, [Xva, Xte], y_fit, seed)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=450, learning_rate=0.025,
                               num_leaves=15, min_child_samples=60, feature_fraction=0.6,
                               bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0,
                               verbose=-1, n_jobs=4, random_state=seed)
        m.fit(Af, y_fit)
        return m.predict_proba(Av)[:, 1], m.predict_proba(At)[:, 1]
    if spec["kind"] == "glm":
        Af, (Av, At) = _glm_matrix(Xtr, [Xva, Xte], y_fit, seed)
        m = LogisticRegression(C=0.05, max_iter=3000, solver="lbfgs")
        m.fit(Af, y_fit)
        return m.predict_proba(Av)[:, 1], m.predict_proba(At)[:, 1]
    raise ValueError(name)


def run_arm(name: str, train: pd.DataFrame, test: pd.DataFrame, edges: dict,
            seeds: list[int], n_splits: int = 5, verbose: bool = True):
    y = train["label"].to_numpy()
    oof_seeds, test_parts = [], []
    for si, seed in enumerate(seeds):
        t0 = time.time()
        Xall, cats = catboost_frame(
            pd.concat([train.drop(columns=["label"]), test], ignore_index=True),
            edges, stream_offset=si + 1)
        Xtr_all = Xall.iloc[: len(train)].reset_index(drop=True)
        Xte_all = Xall.iloc[len(train):].reset_index(drop=True)
        oof = np.zeros(len(y))
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for f, (ti, vi) in enumerate(skf.split(Xtr_all, y)):
            pv, pt = fit_predict(name, Xtr_all.iloc[ti], Xtr_all.iloc[vi], Xte_all,
                                 cats, y[ti], seed + f)
            oof[vi] = pv
            test_parts.append(rankdata(pt) / len(pt))
        oof_seeds.append(rankdata(oof) / len(oof))
        if verbose:
            print(f"  {name} seed={seed} oof={roc_auc_score(y, oof):.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    oof = np.mean(oof_seeds, axis=0)
    pred = np.mean(test_parts, axis=0)
    return dict(oof=oof, test=pred, auc=float(roc_auc_score(y, oof)),
                seed_aucs=[float(roc_auc_score(y, o)) for o in oof_seeds])
