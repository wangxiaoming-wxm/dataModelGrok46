#!/usr/bin/env python3
"""CatBoost-like K-fold target encoding + RMSE boosting. Target: OOF >= 0.69."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

train = pd.read_csv("/workspace/data/train.csv")
test = pd.read_csv("/workspace/data/test.csv")
y = train["label"].astype(int).to_numpy()


def parse_t3_num(s):
    try:
        return float(str(s)[:-1]) if str(s)[-1].isalpha() else float(s)
    except Exception:
        return np.nan


def parse_t3_let(s):
    s = str(s)
    return s[-1] if s and s[-1].isalpha() else "?"


def add_base(df, src_med=None, fit=True):
    out = df.copy()
    out["t3_num"] = out["t3"].map(parse_t3_num)
    out["t3_letter"] = out["t3"].map(parse_t3_let)
    out["car"] = out["source"].str.split("|").str[0]
    out["cond_miss"] = out["condition"].isna().astype(int)
    if fit:
        src_med = out.groupby("source")["condition"].median()
        glob = out["condition"].median()
    else:
        glob = float(src_med.median())
    out["condition_f"] = out["condition"].fillna(out["source"].map(src_med)).fillna(glob)
    gmed = out["source"].map(src_med) if not fit else out.groupby("source")["condition_f"].transform("median")
    # for test, map
    if not fit:
        gmed = out["source"].map(src_med).fillna(glob)
    else:
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
    # quantile bins as category strings (global qcut on train later)
    return out, src_med


tr, src_med = add_base(train, fit=True)

# bins from train
for col, q, name in [
    ("days", 10, "days_q"),
    ("condition_f", 10, "cond_q"),
    ("ratio", 10, "ratio_q"),
    ("rate", 10, "rate_q"),
    ("cond_r", 10, "condr_q"),
]:
    tr[name] = pd.qcut(tr[col], q, duplicates="drop", labels=False).astype(int).astype(str)

# categorical crosses
tr["src"] = tr["source"].astype(str)
tr["reg"] = tr["region"].astype(str)
tr["age"] = tr["age_range"].astype(int).astype(str)
tr["src_reg"] = tr["src"] + "|" + tr["reg"]
tr["src_age"] = tr["src"] + "|" + tr["age"]
tr["reg_age"] = tr["reg"] + "|" + tr["age"]
tr["src_cq"] = tr["src"] + "|" + tr["cond_q"]
tr["src_dq"] = tr["src"] + "|" + tr["days_q"]
tr["reg_cq"] = tr["reg"] + "|" + tr["cond_q"]
tr["reg_dq"] = tr["reg"] + "|" + tr["days_q"]
tr["src_cq_dq"] = tr["src"] + "|" + tr["cond_q"] + "|" + tr["days_q"]
tr["reg_cq_dq"] = tr["reg"] + "|" + tr["cond_q"] + "|" + tr["days_q"]
tr["src_reg_age"] = tr["src"] + "|" + tr["reg"] + "|" + tr["age"]
tr["src_ratioq"] = tr["src"] + "|" + tr["ratio_q"]
tr["src_rateq"] = tr["src"] + "|" + tr["rate_q"]
tr["grades_s"] = tr["grades"].astype(str)
tr["code_s"] = tr["code"].astype(str)
tr["month_s"] = tr["month"].astype(str)
tr["src_grades"] = tr["src"] + "|" + tr["grades_s"]
tr["cq_dq"] = tr["cond_q"] + "|" + tr["days_q"]
tr["src_cq_age"] = tr["src"] + "|" + tr["cond_q"] + "|" + tr["age"]
tr["t3l"] = tr["t3_letter"].astype(str)
tr["src_t3"] = tr["src"] + "|" + tr["t3l"]  # likely redundant

CATS = [
    "src",
    "reg",
    "age",
    "src_reg",
    "src_age",
    "reg_age",
    "src_cq",
    "src_dq",
    "reg_cq",
    "reg_dq",
    "src_cq_dq",
    "reg_cq_dq",
    "src_reg_age",
    "src_ratioq",
    "src_rateq",
    "grades_s",
    "code_s",
    "month_s",
    "src_grades",
    "cq_dq",
    "src_cq_age",
    "t3l",
    "days_q",
    "cond_q",
    "ratio_q",
    "rate_q",
    "condr_q",
]

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


def kfold_te(series, y, folds, m=20.0, prior=None):
    """Out-of-fold target encoding with additive smoothing."""
    if prior is None:
        prior = float(y.mean())
    oof = np.zeros(len(y))
    full_sum = {}
    full_cnt = {}
    for k, v in zip(series, y):
        full_sum[k] = full_sum.get(k, 0.0) + v
        full_cnt[k] = full_cnt.get(k, 0) + 1
    for tr_idx, va_idx in folds:
        # means from tr only
        # faster: subtract va contribution from full
        # but va is a set; compute from train indices
        ssum = {}
        scnt = {}
        for i in tr_idx:
            k = series.iloc[i]
            ssum[k] = ssum.get(k, 0.0) + y[i]
            scnt[k] = scnt.get(k, 0) + 1
        enc = np.empty(len(va_idx))
        for j, i in enumerate(va_idx):
            k = series.iloc[i]
            c = scnt.get(k, 0)
            sm = ssum.get(k, 0.0)
            enc[j] = (sm + prior * m) / (c + m)
        oof[va_idx] = enc
    return oof, full_sum, full_cnt


skf = StratifiedKFold(10, shuffle=True, random_state=2026)
folds = list(skf.split(tr, y))
prior = float(y.mean())

te_mat = []
te_names = []
print("encoding", len(CATS), "categoricals...")
for c in CATS:
    oof, _, _ = kfold_te(tr[c], y, folds, m=20.0, prior=prior)
    te_mat.append(oof)
    te_names.append("te_" + c)
    print(f"  {c:16s} oof-auc-as-score {roc_auc_score(y, oof):.4f}")

TE = np.column_stack(te_mat)
NUM = tr[NUMS].to_numpy(dtype=float)
# fill nan
colmed = np.nanmedian(NUM, axis=0)
inds = np.where(np.isnan(NUM))
NUM[inds] = np.take(colmed, inds[1])
X = np.hstack([NUM, TE])
print("X", X.shape)

# LightGBM RMSE
oof = np.zeros(len(y))
for fold, (tr_i, va_i) in enumerate(folds):
    dtr = lgb.Dataset(X[tr_i], y[tr_i])
    dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_data_in_leaf=80,
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=10.0,
        verbose=-1,
        seed=2026 + fold,
    )
    m = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof[va_i] = m.predict(X[va_i])
print("LGB RMSE 10fold OOF", roc_auc_score(y, oof))

# dual world: drop rate-related vs drop ratio-related and blend
# world A: ratio family
# world B: rate family
idx = {n: i for i, n in enumerate(NUMS)}
keepA = [i for n, i in idx.items() if n not in ("rate",)]
keepB = [i for n, i in idx.items() if n not in ("ratio", "ratio_sqrt", "log_ratio", "days_x_invcond")]
# TE all in both
XA = np.hstack([NUM[:, keepA], TE])
XB = np.hstack([NUM[:, keepB], TE])
oofA = np.zeros(len(y))
oofB = np.zeros(len(y))
for fold, (tr_i, va_i) in enumerate(folds):
    paramsA = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=24,
        max_depth=5,
        min_data_in_leaf=80,
        feature_fraction=0.8,
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
    mA = lgb.train(
        paramsA,
        lgb.Dataset(XA[tr_i], y[tr_i]),
        num_boost_round=800,
        valid_sets=[lgb.Dataset(XA[va_i], y[va_i])],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    mB = lgb.train(
        paramsB,
        lgb.Dataset(XB[tr_i], y[tr_i]),
        num_boost_round=800,
        valid_sets=[lgb.Dataset(XB[va_i], y[va_i])],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    oofA[va_i] = mA.predict(XA[va_i])
    oofB[va_i] = mB.predict(XB[va_i])
print("armA", roc_auc_score(y, oofA), "armB", roc_auc_score(y, oofB))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


print("max rank", roc_auc_score(y, np.maximum(rank(oofA), rank(oofB))))
print("w62", roc_auc_score(y, 0.62 * rank(oofA) + 0.38 * rank(oofB)))
print("w50", roc_auc_score(y, 0.5 * rank(oofA) + 0.5 * rank(oofB)))
print("w70", roc_auc_score(y, 0.70 * rank(oofA) + 0.30 * rank(oofB)))
print("raw blend", roc_auc_score(y, 0.62 * oofA + 0.38 * oofB))

# ridge on TE only
oofR = np.zeros(len(y))
for tr_i, va_i in folds:
    m = Ridge(alpha=5.0)
    m.fit(TE[tr_i], y[tr_i])
    oofR[va_i] = m.predict(TE[va_i])
print("ridge TE", roc_auc_score(y, oofR))
print("w62+ridge", roc_auc_score(y, 0.85 * (0.62 * rank(oofA) + 0.38 * rank(oofB)) + 0.15 * rank(oofR)))
print("DONE")
