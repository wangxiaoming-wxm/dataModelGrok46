#!/usr/bin/env python3
"""Hunt piecewise generating rules and remaining signal after ratio."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

train = pd.read_csv("/workspace/data/train.csv")
y = train["label"].astype(int).to_numpy()
train["condition_f"] = train["condition"].fillna(train["condition"].median())
gmed = train.groupby("source")["condition_f"].transform("median")
train["cond_r"] = train["condition_f"] / gmed
train["ratio"] = train["days"] / train["cond_r"]
rk = train.groupby("source")["condition_f"].rank(pct=True)
train["rate"] = train["days"] * (1.0 - rk)
train["ratio_sqrt"] = train["days"] / np.sqrt(train["condition_f"].clip(1e-6))


def auc(s, y=y):
    s = np.asarray(s, dtype=float)
    s = np.where(np.isnan(s), np.nanmedian(s), s)
    return float(roc_auc_score(y, s))


# ---- days profile 50-wide bins ----
print("=== days 50-bin profile (rate, n) ===")
lo, hi = float(train["days"].min()), float(train["days"].max())
edges = np.arange(0, hi + 50, 50)
train["days_bin50"] = pd.cut(train["days"], edges, include_lowest=True)
tab = train.groupby("days_bin50", observed=True)["label"].agg(["mean", "count"])
# print unusual bins
base = y.mean()
for iv, row in tab.iterrows():
    if row["count"] >= 30 and (row["mean"] < 0.04 or row["mean"] > 0.18):
        print(f"  {iv} n={int(row['count']):4d} rate={row['mean']:.4f}")

# consecutive zero-ish
print("\n=== sliding window exact zeros n>=40 ===")
d = train["days"].to_numpy()
for w in (40, 60, 80, 100, 150, 200):
    for start in np.linspace(d.min(), d.max() - w, 400):
        m = (d >= start) & (d < start + w)
        n = int(m.sum())
        if n >= 40 and y[m].sum() == 0:
            print(f"  ZERO w={w} [{start:.1f},{start+w:.1f}) n={n}")

print("\n=== source x days-cliff ===")
for src, g in train.groupby("source"):
    m = g["days"].between(1700, 1850)
    if m.sum() >= 5:
        print(f"  {src} n1700-1850={int(m.sum())} rate={g.loc[m,'label'].mean():.3f} global={g['label'].mean():.3f}")

print("\n=== 2D days_q x cond_q claim heat (unusual) ===")
train["dq"] = pd.qcut(train["days"], 10, duplicates="drop", labels=False)
train["cq"] = pd.qcut(train["condition_f"], 10, duplicates="drop", labels=False)
pt = train.pivot_table(index="dq", columns="cq", values="label", aggfunc=["mean", "count"])
print("mean:")
print(pt["mean"].round(3).to_string())
print("count:")
print(pt["count"].astype(int).to_string())

print("\n=== source x cond quintile ===")
train["c5"] = pd.qcut(train["condition_f"], 5, labels=False)
print(train.pivot_table(index="source", columns="c5", values="label", aggfunc="mean").round(3).to_string())

print("\n=== age_range rates ===")
print(train.groupby("age_range")["label"].agg(["mean", "count"]))

# isotonic days only
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tr, va in skf.split(train, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train.loc[tr, "days"], y[tr])
    oof[va] = iso.predict(train.loc[va, "days"])
print("\nisotonic(days) oof", auc(oof))

oof2 = np.zeros(len(y))
for tr, va in skf.split(train, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train.loc[tr, "ratio"], y[tr])
    oof2[va] = iso.predict(train.loc[va, "ratio"])
print("isotonic(ratio) oof", auc(oof2))

# LightGBM RMSE
print("\n=== LGB RMSE ===")
use = train.copy()
use["source"] = use["source"].astype("category")
use["region"] = use["region"].astype("category")
use["age_range"] = use["age_range"].astype("category")
cols = [
    "days",
    "condition_f",
    "ratio",
    "ratio_sqrt",
    "rate",
    "age_range",
    "source",
    "region",
    "V",
    "cc",
    "x20",
    "x1",
    "x5",
]
X = use[cols]
oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=["source", "region", "age_range"])
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=["source", "region", "age_range"], reference=dtr)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=80,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        verbose=-1,
        seed=42 + fold,
    )
    m = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
print("lgb compact rmse oof", auc(oof))

# add window flags
use["safe_1725"] = ((use["days"] >= 1725) & (use["days"] < 1825)).astype(int)
use["safe_750"] = ((use["days"] >= 700) & (use["days"] < 880)).astype(int)
use["safe_2100"] = ((use["days"] >= 2100) & (use["days"] < 2200)).astype(int)
use["newcar"] = (use["days"] < 50).astype(int)
cols2 = cols + ["safe_1725", "safe_750", "safe_2100", "newcar"]
X = use[cols2]
oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=["source", "region", "age_range"])
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=["source", "region", "age_range"], reference=dtr)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=80,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        verbose=-1,
        seed=42 + fold,
    )
    m = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
print("lgb +window flags oof", auc(oof))

# more cats crosses as strings then category
use["src_reg"] = (use["source"].astype(str) + "|" + use["region"].astype(str)).astype("category")
use["src_age"] = (use["source"].astype(str) + "|" + use["age_range"].astype(str)).astype("category")
use["reg_age"] = (use["region"].astype(str) + "|" + use["age_range"].astype(str)).astype("category")
use["src_c5"] = (use["source"].astype(str) + "|" + pd.qcut(use["condition_f"], 10, labels=False, duplicates="drop").astype(str)).astype(
    "category"
)
cols3 = cols2 + ["src_reg", "src_age", "reg_age", "src_c5"]
X = use[cols3]
catf = ["source", "region", "age_range", "src_reg", "src_age", "reg_age", "src_c5"]
oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=catf)
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=catf, reference=dtr)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=48,
        min_data_in_leaf=60,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=8.0,
        verbose=-1,
        seed=42 + fold,
    )
    m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
print("lgb +cross cats oof", auc(oof))

# 10 fold stronger
print("\n=== LGB 10fold stronger ===")
skf10 = StratifiedKFold(10, shuffle=True, random_state=2026)
oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(skf10.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=catf)
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=catf, reference=dtr)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.02,
        num_leaves=40,
        min_data_in_leaf=70,
        feature_fraction=0.65,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=10.0,
        verbose=-1,
        seed=2026 + fold,
    )
    m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
print("lgb 10fold cross oof", auc(oof))

print("DONE")
