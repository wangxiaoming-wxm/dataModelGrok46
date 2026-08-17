#!/usr/bin/env python3
"""Per-source residual models and U-shape features — quantify extra lift."""
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
train["condition_f"] = train["condition"].fillna(train.groupby("source")["condition"].transform("median"))
train["condition_f"] = train["condition_f"].fillna(train["condition"].median())
gmed = train.groupby("source")["condition_f"].transform("median")
gstd = train.groupby("source")["condition_f"].transform("std").replace(0, 1)
train["cond_r"] = train["condition_f"] / gmed
train["cond_z"] = (train["condition_f"] - gmed) / gstd
train["cond_rk"] = train.groupby("source")["condition_f"].rank(pct=True)
train["u_shape"] = (train["cond_rk"] - 0.5) ** 2
train["ratio"] = train["days"] / train["cond_r"]
train["rate"] = train["days"] * (1.0 - train["cond_rk"])
train["ratio_sqrt"] = train["days"] / np.sqrt(train["condition_f"].clip(1e-6))
train["days_z"] = (train["days"] - train.groupby("source")["days"].transform("mean")) / train.groupby("source")["days"].transform("std").replace(0, 1)

# source-specific U vs monotone flags based on known shapes
# CAR_10 inverted-U; CAR_1 monotone protective
train["cond_ushape_x"] = train["u_shape"] * train["source"].isin(["CAR_10|ENG_651", "CAR_0|ENG_709", "CAR_2|ENG_262"]).astype(float)
train["cond_mono_x"] = train["cond_rk"] * train["source"].isin(["CAR_1|ENG_591"]).astype(float)


def auc(s):
    s = np.asarray(s, dtype=float)
    return float(roc_auc_score(y, np.where(np.isnan(s), np.nanmedian(s), s)))


print("univariate extra", {k: auc(train[k]) for k in ["u_shape", "cond_z", "cond_ushape_x", "cond_mono_x", "days_z"]})

skf = StratifiedKFold(10, shuffle=True, random_state=2026)
use = train.copy()
use["source"] = use["source"].astype("category")
use["region"] = use["region"].astype("category")
num = [
    "days",
    "condition_f",
    "ratio",
    "ratio_sqrt",
    "rate",
    "cond_r",
    "cond_z",
    "cond_rk",
    "u_shape",
    "days_z",
    "cond_ushape_x",
    "cond_mono_x",
    "age_range",
    "V",
    "x20",
]
cat = ["source", "region"]
X = use[num + cat]


def fit_lgb(Xtr, ytr, Xva, yva, seed, extra_cat=None):
    cats = cat if extra_cat is None else extra_cat
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, categorical_feature=cats, reference=dtr, free_raw_data=False)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=80,
        feature_fraction=0.75,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=8.0,
        verbose=-1,
        seed=seed,
    )
    return lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


oof_g = np.zeros(len(y))
oof_h = np.zeros(len(y))  # hierarchical: global + per-source residual
large = ["CAR_0|ENG_709", "CAR_1|ENG_591", "CAR_2|ENG_262", "CAR_5|ENG_062"]

for fold, (tr, va) in enumerate(skf.split(X, y)):
    m = fit_lgb(X.iloc[tr], y[tr], X.iloc[va], y[va], 2026 + fold)
    pred_g = m.predict(X.iloc[va])
    oof_g[va] = pred_g
    # residual per large source
    pred_h = pred_g.copy()
    resid = y[tr] - m.predict(X.iloc[tr])
    tr_df = use.iloc[tr].copy()
    va_df = use.iloc[va].copy()
    tr_df["_resid"] = resid
    rcols = ["days", "condition_f", "ratio", "rate", "cond_rk", "u_shape", "age_range", "region"]
    for src in large:
        gtr = tr_df[tr_df["source"].astype(str) == src]
        gva_idx = np.where(va_df["source"].astype(str).to_numpy() == src)[0]
        if len(gtr) < 200 or len(gva_idx) == 0:
            continue
        dtr = lgb.Dataset(gtr[rcols], gtr["_resid"], categorical_feature=["region"])
        params = dict(
            objective="regression",
            learning_rate=0.03,
            num_leaves=16,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            lambda_l2=12.0,
            verbose=-1,
            seed=fold,
        )
        m2 = lgb.train(params, dtr, num_boost_round=200)
        pred_h[gva_idx] = pred_g[gva_idx] + 0.5 * m2.predict(va_df.iloc[gva_idx][rcols])
    oof_h[va] = pred_h

print("global 10fold", auc(oof_g))
print("hier 10fold", auc(oof_h))
print("blend", auc(0.7 * oof_g + 0.3 * oof_h))

# dedicated per-source models replacing global for large sources
oof_ps = oof_g.copy()
for fold, (tr, va) in enumerate(skf.split(X, y)):
    tr_df = use.iloc[tr]
    va_df = use.iloc[va]
    for src in large:
        gtr = tr_df[tr_df["source"].astype(str) == src]
        mask = va_df["source"].astype(str).to_numpy() == src
        gva = va_df[mask]
        if len(gtr) < 200 or len(gva) == 0:
            continue
        cols = num + ["region"]
        dtr = lgb.Dataset(gtr[cols], gtr["label"], categorical_feature=["region"])
        dva = lgb.Dataset(gva[cols], gva["label"], categorical_feature=["region"])
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=24,
            min_data_in_leaf=40,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=6.0,
            verbose=-1,
            seed=fold,
        )
        m2 = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
        oof_ps[va[mask]] = m2.predict(gva[cols])

print("replace-large-source", auc(oof_ps))
print("max(global, ps)", auc(np.maximum(oof_g, oof_ps)))
print("avg", auc(0.5 * oof_g + 0.5 * oof_ps))
print("rank-ish avg of standardized")


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


print("rank blend 0.62", auc(0.62 * rank(oof_g) + 0.38 * rank(oof_ps)))
print("DONE")
