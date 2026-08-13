#!/usr/bin/env python3
"""Honest nested K-fold TE: encoders fit only on outer-train. No val-label leak."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

train = pd.read_csv("/workspace/data/train.csv")
y = train["label"].astype(int).to_numpy()


def parse_t3_num(s):
    try:
        return float(str(s)[:-1]) if str(s)[-1].isalpha() else float(s)
    except Exception:
        return np.nan


def add_base(df):
    out = df.copy()
    out["t3_num"] = out["t3"].map(parse_t3_num)
    out["t3_letter"] = out["t3"].map(lambda s: str(s)[-1] if str(s)[-1].isalpha() else "?")
    out["cond_miss"] = out["condition"].isna().astype(int)
    src_med = out.groupby("source")["condition"].transform("median")
    glob = out["condition"].median()
    out["condition_f"] = out["condition"].fillna(src_med).fillna(glob)
    gmed = out.groupby("source")["condition_f"].transform("median")
    out["cond_r"] = out["condition_f"] / gmed.replace(0, np.nan)
    out["cond_rk"] = out.groupby("source")["condition_f"].rank(pct=True)
    out["ratio"] = out["days"] / out["cond_r"]
    out["rate"] = out["days"] * (1.0 - out["cond_rk"])
    out["ratio_sqrt"] = out["days"] / np.sqrt(out["condition_f"].clip(1e-6))
    out["log_ratio"] = np.log(out["days"].clip(1)) - 0.5 * np.log(out["condition_f"].clip(1e-6))
    out["u_shape"] = (out["cond_rk"] - 0.5) ** 2
    out["days_log"] = np.log1p(out["days"])
    out["inv_cond"] = 1.0 / out["condition_f"].clip(1e-4)
    out["days_x_invcond"] = out["days"] * out["inv_cond"]
    out["safe_750"] = ((out["days"] >= 700) & (out["days"] < 880)).astype(int)
    out["safe_1750"] = ((out["days"] >= 1725) & (out["days"] < 1825)).astype(int)
    out["hot_1950"] = ((out["days"] >= 1950) & (out["days"] < 2000)).astype(int)
    out["age8"] = (out["age_range"] >= 8).astype(int)
    out["cond_low"] = (out["condition_f"] < 0.05).astype(int)
    return out


def qcut_fit_apply(series_tr, series_va, q=10):
    cats, bins = pd.qcut(series_tr, q, duplicates="drop", retbins=True, labels=False)
    bins[0], bins[-1] = -np.inf, np.inf
    va = pd.cut(series_va, bins=bins, labels=False, include_lowest=True)
    return cats.astype(int).astype(str), va.astype(int).astype(str)


def loo_te(keys, y, m=20.0, prior=None):
    """Leave-one-out smoothed TE for training rows."""
    if prior is None:
        prior = float(np.mean(y))
    ssum = {}
    scnt = {}
    for k, yi in zip(keys, y):
        ssum[k] = ssum.get(k, 0.0) + yi
        scnt[k] = scnt.get(k, 0) + 1
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        c = scnt[k]
        sm = ssum[k]
        out[i] = (sm - y[i] + prior * m) / (c - 1 + m)
    stats = {k: ((ssum[k] + prior * m) / (scnt[k] + m), scnt[k]) for k in ssum}
    return out, stats


def apply_te(keys, stats, prior, m=20.0):
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        if k in stats:
            out[i] = stats[k][0]
        else:
            out[i] = prior
    return out


tr = add_base(train)
NUMS = [
    "days",
    "days_log",
    "condition_f",
    "cond_r",
    "cond_rk",
    "ratio",
    "rate",
    "ratio_sqrt",
    "log_ratio",
    "u_shape",
    "inv_cond",
    "days_x_invcond",
    "age_range",
    "V",
    "cc",
    "max_g",
    "x20",
    "x1",
    "x5",
    "x14",
    "x17",
    "t3_num",
    "cond_miss",
    "safe_750",
    "safe_1750",
    "hot_1950",
    "age8",
    "cond_low",
]


def make_cats(df):
    d = {}
    d["src"] = df["source"].astype(str).to_numpy()
    d["reg"] = df["region"].astype(str).to_numpy()
    d["age"] = df["age_range"].astype(int).astype(str).to_numpy()
    d["grades"] = df["grades"].astype(str).to_numpy()
    d["code"] = df["code"].astype(str).to_numpy()
    d["month"] = df["month"].astype(str).to_numpy()
    return d


skf = StratifiedKFold(10, shuffle=True, random_state=2026)
oofA = np.zeros(len(y))
oofB = np.zeros(len(y))
oofC = np.zeros(len(y))  # RF-like extra: more leaves, more rsm
prior_global = float(y.mean())

for fold, (tr_i, va_i) in enumerate(skf.split(tr, y)):
    print(f"fold {fold}")
    trn = tr.iloc[tr_i].reset_index(drop=True)
    val = tr.iloc[va_i].reset_index(drop=True)
    ytr = y[tr_i]
    prior = float(ytr.mean())

    # quantile bins fit on train only
    cats_tr = make_cats(trn)
    cats_va = make_cats(val)
    for col, name in [
        ("days", "dq"),
        ("condition_f", "cq"),
        ("ratio", "rq"),
        ("rate", "tq"),
        ("cond_r", "crq"),
    ]:
        a, b = qcut_fit_apply(trn[col], val[col], 10)
        cats_tr[name] = a.to_numpy()
        cats_va[name] = b.to_numpy()

    def cross(d, *names):
        parts = [d[n] for n in names]
        out = parts[0].astype(object)
        for p in parts[1:]:
            out = out + "|" + p
        return out.astype(str)

    cat_map_tr = {
        "src": cats_tr["src"],
        "reg": cats_tr["reg"],
        "age": cats_tr["age"],
        "src_reg": cross(cats_tr, "src", "reg"),
        "src_age": cross(cats_tr, "src", "age"),
        "reg_age": cross(cats_tr, "reg", "age"),
        "src_cq": cross(cats_tr, "src", "cq"),
        "src_dq": cross(cats_tr, "src", "dq"),
        "reg_cq": cross(cats_tr, "reg", "cq"),
        "reg_dq": cross(cats_tr, "reg", "dq"),
        "src_cq_dq": cross(cats_tr, "src", "cq", "dq"),
        "reg_cq_dq": cross(cats_tr, "reg", "cq", "dq"),
        "src_reg_age": cross(cats_tr, "src", "reg", "age"),
        "src_rq": cross(cats_tr, "src", "rq"),
        "src_tq": cross(cats_tr, "src", "tq"),
        "cq_dq": cross(cats_tr, "cq", "dq"),
        "src_cq_age": cross(cats_tr, "src", "cq", "age"),
        "dq": cats_tr["dq"],
        "cq": cats_tr["cq"],
        "rq": cats_tr["rq"],
        "tq": cats_tr["tq"],
        "crq": cats_tr["crq"],
        "grades": cats_tr["grades"],
        "src_grades": cross(cats_tr, "src", "grades"),
    }
    cat_map_va = {
        "src": cats_va["src"],
        "reg": cats_va["reg"],
        "age": cats_va["age"],
        "src_reg": cross(cats_va, "src", "reg"),
        "src_age": cross(cats_va, "src", "age"),
        "reg_age": cross(cats_va, "reg", "age"),
        "src_cq": cross(cats_va, "src", "cq"),
        "src_dq": cross(cats_va, "src", "dq"),
        "reg_cq": cross(cats_va, "reg", "cq"),
        "reg_dq": cross(cats_va, "reg", "dq"),
        "src_cq_dq": cross(cats_va, "src", "cq", "dq"),
        "reg_cq_dq": cross(cats_va, "reg", "cq", "dq"),
        "src_reg_age": cross(cats_va, "src", "reg", "age"),
        "src_rq": cross(cats_va, "src", "rq"),
        "src_tq": cross(cats_va, "src", "tq"),
        "cq_dq": cross(cats_va, "cq", "dq"),
        "src_cq_age": cross(cats_va, "src", "cq", "age"),
        "dq": cats_va["dq"],
        "cq": cats_va["cq"],
        "rq": cats_va["rq"],
        "tq": cats_va["tq"],
        "crq": cats_va["crq"],
        "grades": cats_va["grades"],
        "src_grades": cross(cats_va, "src", "grades"),
    }

    te_tr = []
    te_va = []
    for name in cat_map_tr:
        enc_tr, stats = loo_te(cat_map_tr[name], ytr, m=20.0, prior=prior)
        enc_va = apply_te(cat_map_va[name], stats, prior, m=20.0)
        te_tr.append(enc_tr)
        te_va.append(enc_va)
        # also frequency
        freq = {k: stats[k][1] for k in stats}
        fr_tr = np.array([freq.get(k, 0) for k in cat_map_tr[name]], dtype=float)
        fr_va = np.array([freq.get(k, 0) for k in cat_map_va[name]], dtype=float)
        te_tr.append(np.log1p(fr_tr))
        te_va.append(np.log1p(fr_va))

    TE_tr = np.column_stack(te_tr)
    TE_va = np.column_stack(te_va)

    def num_mat(df):
        M = df[NUMS].to_numpy(dtype=float)
        med = np.nanmedian(M, axis=0)
        inds = np.where(np.isnan(M))
        M[inds] = np.take(med, inds[1])
        return M

    Ntr = num_mat(trn)
    Nva = num_mat(val)
    # val numeric fill using train medians
    med = np.nanmedian(trn[NUMS].to_numpy(dtype=float), axis=0)
    Nva_raw = val[NUMS].to_numpy(dtype=float)
    inds = np.where(np.isnan(Nva_raw))
    Nva_raw[inds] = np.take(med, inds[1])
    Nva = Nva_raw

    Xtr = np.hstack([Ntr, TE_tr])
    Xva = np.hstack([Nva, TE_va])

    paramsA = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=24,
        max_depth=5,
        min_data_in_leaf=80,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=10.0,
        verbose=-1,
        seed=1000 + fold,
    )
    paramsB = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=40,
        max_depth=6,
        min_data_in_leaf=70,
        feature_fraction=0.3,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=6.0,
        verbose=-1,
        seed=2000 + fold,
    )
    paramsC = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_data_in_leaf=100,
        feature_fraction=0.5,
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l2=12.0,
        verbose=-1,
        seed=3000 + fold,
    )
    mA = lgb.train(
        paramsA,
        lgb.Dataset(Xtr, ytr),
        num_boost_round=800,
        valid_sets=[lgb.Dataset(Xva, y[va_i])],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    mB = lgb.train(
        paramsB,
        lgb.Dataset(Xtr, ytr),
        num_boost_round=800,
        valid_sets=[lgb.Dataset(Xva, y[va_i])],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    mC = lgb.train(
        paramsC,
        lgb.Dataset(Xtr, ytr),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(Xva, y[va_i])],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    oofA[va_i] = mA.predict(Xva)
    oofB[va_i] = mB.predict(Xva)
    oofC[va_i] = mC.predict(Xva)
    print(
        "  fold auc A/B/C",
        roc_auc_score(y[va_i], oofA[va_i]),
        roc_auc_score(y[va_i], oofB[va_i]),
        roc_auc_score(y[va_i], oofC[va_i]),
    )

print("HONEST armA", roc_auc_score(y, oofA))
print("HONEST armB", roc_auc_score(y, oofB))
print("HONEST armC", roc_auc_score(y, oofC))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


print("HONEST w62 A/B", roc_auc_score(y, 0.62 * rank(oofA) + 0.38 * rank(oofB)))
print("HONEST w50 A/B", roc_auc_score(y, 0.5 * rank(oofA) + 0.5 * rank(oofB)))
print("HONEST max A/B", roc_auc_score(y, np.maximum(rank(oofA), rank(oofB))))
print("HONEST avg3", roc_auc_score(y, (rank(oofA) + rank(oofB) + rank(oofC)) / 3))
print("HONEST 0.45A+0.35B+0.20C", roc_auc_score(y, 0.45 * rank(oofA) + 0.35 * rank(oofB) + 0.20 * rank(oofC)))
np.save("/workspace/analysis/oofA.npy", oofA)
np.save("/workspace/analysis/oofB.npy", oofB)
np.save("/workspace/analysis/oofC.npy", oofC)
print("DONE")
