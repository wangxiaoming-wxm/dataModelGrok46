#!/usr/bin/env python3
"""Deep signal hunt: confirm known findings and search for extra lift toward 0.75 AUC."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

DATA = Path("/workspace/data")
OUT = Path("/workspace/analysis")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["label"].astype(int).to_numpy()
print("train", train.shape, "pos", y.mean(), "test", test.shape)


def auc(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.isnan(a).any():
        med = np.nanmedian(a)
        a = np.where(np.isnan(a), med, a)
    try:
        return float(roc_auc_score(b, a))
    except Exception:
        return float("nan")


def both_auc(s, y=y):
    a = auc(s, y)
    b = auc(-np.asarray(s, dtype=float), y)
    return max(a, b), a, b


# ---------- parse helpers ----------
def parse_t3(s):
    m = re.match(r"([0-9.]+)([A-Za-z]*)", str(s))
    if not m:
        return np.nan, "?"
    return float(m.group(1)), (m.group(2) or "?")


t3p = train["t3"].map(parse_t3)
train["t3_num"] = [p[0] for p in t3p]
train["t3_letter"] = [p[1] for p in t3p]
train["car"] = train["source"].str.split("|").str[0]
train["eng"] = train["source"].str.split("|").str[1]
train["cond_missing"] = train["condition"].isna().astype(int)
cond_med = train["condition"].median()
train["condition_f"] = train["condition"].fillna(cond_med)

# source-wise condition median / rank
gmed = train.groupby("source")["condition_f"].transform("median")
train["cond_r"] = train["condition_f"] / gmed.replace(0, np.nan)
train["ratio"] = train["days"] / train["cond_r"]
train["ratio_raw"] = train["days"] / train["condition_f"].clip(lower=1e-6)
train["ratio_sqrt"] = train["days"] / np.sqrt(train["condition_f"].clip(lower=1e-6))
train["log_ratio"] = np.log(train["days"].clip(1)) - 0.5 * np.log(train["condition_f"].clip(1e-6))
rk = train.groupby("source")["condition_f"].rank(pct=True)
train["rate"] = train["days"] * (1.0 - rk)

print("\n=== univariate ===")
for col in [
    "days",
    "condition_f",
    "ratio",
    "ratio_raw",
    "ratio_sqrt",
    "log_ratio",
    "rate",
    "age_range",
    "V",
    "cc",
    "max_g",
    "x19",
    "x18",
    "x20",
    "livability",
    "t3_num",
    "cond_missing",
]:
    m, a, b = both_auc(train[col])
    print(f"{col:16s} max={m:.4f} raw={a:.4f} neg={b:.4f}")

print("\n=== days cliff scan ===")
days = train["days"].to_numpy()
best = []
for lo in range(0, 4000, 25):
    for w in (25, 50, 75, 100, 150, 200, 300):
        hi = lo + w
        m = (days >= lo) & (days < hi)
        n = int(m.sum())
        if n < 40:
            continue
        rate = float(y[m].mean())
        # lift vs complement
        if n < 80 and abs(rate - 0.10) < 0.05:
            continue
        best.append((abs(rate - 0.1002), rate, n, lo, hi))
best.sort(reverse=True)
print("largest |rate-base| windows:")
for item in best[:25]:
    print(f"  days[{item[3]},{item[4]}) n={item[2]} rate={item[1]:.4f} absdiff={item[0]:.4f}")

print("\n=== condition missing ===")
print(train.groupby("cond_missing")["label"].agg(["mean", "count"]))

print("\n=== coverage flags ===")
for cols in [
    ["t1", "t2"],
    ["w1", "w2"],
    ["c1", "c2"],
    ["r1", "r2"],
]:
    s = train[cols].sum(axis=1)
    print(cols, train.groupby(s)["label"].agg(["mean", "count"]).to_dict())

print("\n=== t1+t2 both 1 ===")
both = (train["t1"] == 1) & (train["t2"] == 1)
print("n", both.sum(), "rate", y[both].mean(), "else", y[~both].mean(), "auc", both_auc(both.astype(float))[0])

print("\n=== livability vs region ===")
print("nunique livability", train["livability"].nunique(), "region", train["region"].nunique())
print("region->livability nunique max", train.groupby("region")["livability"].nunique().max())
print("livability->region nunique max", train.groupby("livability")["region"].nunique().max())

print("\n=== t3_letter vs source ===")
print(pd.crosstab(train["source"], train["t3_letter"]))

# residual after ratio: what else predicts?
print("\n=== residual after isotonic-like bin of ratio ===")
from sklearn.isotonic import IsotonicRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_iso = np.zeros(len(train))
for tr, va in skf.split(train, y):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train.loc[tr, "ratio"].to_numpy(), y[tr])
    oof_iso[va] = iso.predict(train.loc[va, "ratio"].to_numpy())
print("isotonic(ratio) oof", auc(oof_iso, y))
resid = y - oof_iso

# correlate residual with other columns
num_cols = [
    c
    for c in train.columns
    if c not in ("id", "label", "t3", "source", "month", "region", "code", "grades", "version", "car", "eng", "t3_letter")
    and np.issubdtype(train[c].dtype, np.number)
]
print("residual |corr| top:")
corrs = []
for c in num_cols:
    v = train[c].to_numpy(dtype=float)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    if v.std() < 1e-12:
        continue
    r = float(np.corrcoef(v, resid)[0, 1])
    corrs.append((abs(r), r, c))
for item in sorted(corrs, reverse=True)[:20]:
    print(f"  {item[2]:16s} r={item[1]:+.4f}")

# per-source residual mean
print("\n=== per-source residual after iso(ratio) ===")
tmp = train.copy()
tmp["resid"] = resid
print(tmp.groupby("source")["resid"].agg(["mean", "count"]))
print("region residual:")
print(tmp.groupby("region")["resid"].agg(["mean", "count"]).sort_values("mean"))

# OOF linear model on core features
print("\n=== OOF ridge on engineered features ===")


def oof_ridge(X, y, alpha=1.0):
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = Ridge(alpha=alpha)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof, auc(oof, y)


feats = np.column_stack(
    [
        train["days"],
        train["condition_f"],
        train["ratio"],
        train["ratio_sqrt"],
        train["rate"],
        train["log_ratio"],
        train["age_range"],
        np.log1p(train["days"]),
        train["cond_missing"],
        (train["days"].between(700, 880)).astype(float),
        (train["age_range"] >= 8).astype(float),
        (train["condition_f"] < 0.05).astype(float),
    ]
)
# standardize
feats_s = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
oof, a = oof_ridge(feats_s, y)
print("ridge core", a)

# add source OHE
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
src = ohe.fit_transform(train[["source"]])
X2 = np.hstack([feats_s, src])
oof, a = oof_ridge(X2, y)
print("ridge core+source", a)

ohe_r = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
reg = ohe_r.fit_transform(train[["region"]])
X3 = np.hstack([X2, reg])
oof, a = oof_ridge(X3, y)
print("ridge core+source+region", a)

age = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(train[["age_range"]].astype(str))
X4 = np.hstack([X3, age])
oof, a = oof_ridge(X4, y)
print("ridge +age ohe", a)

# source * ratio / source * condition interactions
src_ratio = src * train["ratio"].to_numpy()[:, None]
src_cond = src * train["condition_f"].to_numpy()[:, None]
src_days = src * train["days"].to_numpy()[:, None]
X5 = np.hstack([X4, src_ratio, src_cond, src_days])
# too many cols, stronger ridge
oof, a = oof_ridge(X5, y, alpha=10)
print("ridge +source interactions", a)

# LightGBM RMSE on a compact feature set (to estimate remaining headroom)
print("\n=== LightGBM RMSE compact ===")
import lightgbm as lgb

cat_cols = ["source", "region", "month", "code", "grades", "version", "t3_letter", "car"]
for c in cat_cols:
    train[c] = train[c].astype("category")

lgb_cols = [
    "days",
    "condition_f",
    "ratio",
    "ratio_sqrt",
    "rate",
    "log_ratio",
    "age_range",
    "cond_missing",
    "V",
    "cc",
    "max_g",
    "x20",
    "x1",
    "x5",
    "livability",
    "t3_num",
    "source",
    "region",
    "month",
    "t3_letter",
    "car",
    "grades",
]
X = train[lgb_cols]
oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols[:7])
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_cols[:7], reference=dtr)
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
print("lgb compact rmse oof", auc(oof, y))

# per-source models vs global
print("\n=== per-source LGB vs global (quick 5fold on days+cond+ratio+region) ===")
base_cols = ["days", "condition_f", "ratio", "ratio_sqrt", "rate", "age_range", "region"]
# global
Xg = train[base_cols].copy()
Xg["region"] = Xg["region"].astype("category")
oof_g = np.zeros(len(y))
oof_ps = np.zeros(len(y))
for tr, va in skf.split(train, y):
    dtr = lgb.Dataset(Xg.iloc[tr], y[tr], categorical_feature=["region"])
    dva = lgb.Dataset(Xg.iloc[va], y[va], categorical_feature=["region"], reference=dtr)
    params = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=24, min_data_in_leaf=60, verbose=-1, seed=0)
    m = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
    oof_g[va] = m.predict(Xg.iloc[va])
    # per source
    pred = np.zeros(len(va))
    for src, g in train.iloc[tr].groupby("source"):
        idx_va = np.where(train.iloc[va]["source"].to_numpy() == src)[0]
        if len(g) < 120 or len(idx_va) == 0:
            continue
        cols = ["days", "condition_f", "ratio", "ratio_sqrt", "rate", "age_range"]
        dtr2 = lgb.Dataset(g[cols], g["label"])
        m2 = lgb.train(
            dict(objective="regression", learning_rate=0.05, num_leaves=16, min_data_in_leaf=40, verbose=-1),
            dtr2,
            num_boost_round=200,
        )
        pred[idx_va] = m2.predict(train.iloc[va].iloc[idx_va][cols])
    # fill empty with global
    empty = pred == 0
    pred[empty] = oof_g[va][empty]
    oof_ps[va] = pred
print("global compact", auc(oof_g, y), "per-source blend", auc(oof_ps, y), "max blend", auc(np.maximum(oof_g, oof_ps), y))
print("avg blend", auc(0.6 * oof_g + 0.4 * oof_ps, y))

# Decision tree on ratio+source to hunt rules
print("\n=== shallow tree rules ===")
Xt = np.column_stack([train["days"], train["condition_f"], train["ratio"], train["age_range"], src])
tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=80)
tree.fit(Xt, y)
from sklearn.tree import export_text

print(export_text(tree, feature_names=["days", "cond", "ratio", "age"] + list(ohe.get_feature_names_out()), max_depth=4)[:4000])
print("tree auc in-sample", auc(tree.predict(Xt), y))

# region x days sign flip
print("\n=== region days corr ===")
rows = []
for r, g in train.groupby("region"):
    if len(g) < 80:
        continue
    rows.append((float(np.corrcoef(g["days"], g["label"])[0, 1]), g["label"].mean(), len(g), r))
for item in sorted(rows):
    print(f"  {item[3]} n={item[2]} rate={item[1]:.4f} corr_days={item[0]:+.3f}")

print("\nDONE")
