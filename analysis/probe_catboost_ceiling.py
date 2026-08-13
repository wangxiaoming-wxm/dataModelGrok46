#!/usr/bin/env python3
"""CatBoost RMSE dual-world + 3-way cats. Honest fold-wise bins. Ceiling check."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

train = pd.read_csv("/workspace/data/train.csv")
y = train["label"].astype(int).to_numpy()


def enrich(df):
    d = df.copy()
    d["t3_num"] = d["t3"].map(lambda s: float(str(s)[:-1]) if str(s)[-1].isalpha() else np.nan)
    d["t3_letter"] = d["t3"].map(lambda s: str(s)[-1] if str(s)[-1].isalpha() else "?")
    d["car"] = d["source"].str.split("|").str[0]
    d["cond_miss"] = d["condition"].isna().astype(int)
    return d


train = enrich(train)


def fold_features(trn, val):
    """Compute dual-world features; qcut edges from trn only."""
    med = trn.groupby("source")["condition"].median()
    g = float(trn["condition"].median())

    def fill_cond(df):
        c = df["condition"].fillna(df["source"].map(med)).fillna(g)
        return c

    trn = trn.copy()
    val = val.copy()
    trn["condition_f"] = fill_cond(trn)
    val["condition_f"] = fill_cond(val)
    sm = trn.groupby("source")["condition_f"].median()

    def cond_r(df):
        return df["condition_f"] / df["source"].map(sm).replace(0, np.nan).fillna(sm.median())

    trn["cond_r"] = cond_r(trn)
    val["cond_r"] = cond_r(val)

    # rank within source using train ranks; for val, percentile vs train distribution
    trn["cond_rk"] = trn.groupby("source")["condition_f"].rank(pct=True)

    def rank_apply(df):
        out = np.zeros(len(df))
        for i, (src, v) in enumerate(zip(df["source"].to_numpy(), df["condition_f"].to_numpy())):
            ref = trn.loc[trn["source"] == src, "condition_f"].to_numpy()
            if len(ref) == 0:
                out[i] = 0.5
            else:
                out[i] = (ref <= v).mean()
        return out

    val["cond_rk"] = rank_apply(val)
    for df in (trn, val):
        df["ratio"] = df["days"] / df["cond_r"]
        df["rate"] = df["days"] * (1.0 - df["cond_rk"])
        df["ratio_sqrt"] = df["days"] / np.sqrt(df["condition_f"].clip(1e-6))
        df["u_shape"] = (df["cond_rk"] - 0.5) ** 2
        df["inv_cond"] = 1.0 / df["condition_f"].clip(1e-4)
        df["days_log"] = np.log1p(df["days"])
        df["age8"] = (df["age_range"] >= 8).astype(int)
        df["cond_low"] = (df["condition_f"] < 0.05).astype(int)
        df["safe_750"] = ((df["days"] >= 700) & (df["days"] < 880)).astype(int)
        df["safe_1750"] = ((df["days"] >= 1725) & (df["days"] < 1825)).astype(int)

    def add_q(name, series_tr, series_va, q=10):
        cats, bins = pd.qcut(series_tr, q, duplicates="drop", retbins=True, labels=False)
        bins = bins.copy()
        bins[0], bins[-1] = -np.inf, np.inf
        trn[name] = cats.astype(int).astype(str)
        val[name] = pd.cut(series_va, bins=bins, labels=False, include_lowest=True).fillna(0).astype(int).astype(str)

    add_q("days_q", trn["days"], val["days"])
    add_q("cond_q", trn["condition_f"], val["condition_f"])
    add_q("ratio_q", trn["ratio"], val["ratio"])
    add_q("rate_q", trn["rate"], val["rate"])

    for df in (trn, val):
        df["src"] = df["source"].astype(str)
        df["reg"] = df["region"].astype(str)
        df["age"] = df["age_range"].astype(int).astype(str)
        df["src_reg"] = df["src"] + "|" + df["reg"]
        df["src_age"] = df["src"] + "|" + df["age"]
        df["reg_age"] = df["reg"] + "|" + df["age"]
        df["src_cq"] = df["src"] + "|" + df["cond_q"]
        df["src_dq"] = df["src"] + "|" + df["days_q"]
        df["reg_cq"] = df["reg"] + "|" + df["cond_q"]
        df["reg_dq"] = df["reg"] + "|" + df["days_q"]
        df["src_cq_dq"] = df["src"] + "|" + df["cond_q"] + "|" + df["days_q"]
        df["cq_dq"] = df["cond_q"] + "|" + df["days_q"]
        df["src_ratioq"] = df["src"] + "|" + df["ratio_q"]
        df["src_rateq"] = df["src"] + "|" + df["rate_q"]
        df["src_cq_age"] = df["src"] + "|" + df["cond_q"] + "|" + df["age"]
        df["reg_cq_dq"] = df["reg"] + "|" + df["cond_q"] + "|" + df["days_q"]
        df["grades_s"] = df["grades"].astype(str)
        df["month_s"] = df["month"].astype(str)
        df["code_s"] = df["code"].astype(str)
    return trn, val


NUM_MAIN = [
    "days",
    "days_log",
    "condition_f",
    "cond_r",
    "ratio",
    "ratio_sqrt",
    "u_shape",
    "inv_cond",
    "age_range",
    "V",
    "cc",
    "x20",
    "x1",
    "x5",
    "age8",
    "cond_low",
    "safe_750",
    "safe_1750",
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
    "age_range",
    "V",
    "x20",
    "x1",
    "age8",
    "cond_low",
    "safe_750",
    "safe_1750",
    "t3_num",
    "cond_miss",
]
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
    "cq_dq",
    "src_ratioq",
    "src_rateq",
    "src_cq_age",
    "reg_cq_dq",
    "grades_s",
    "month_s",
    "code_s",
    "days_q",
    "cond_q",
]


def fit_cb(trn, val, ytr, yva, nums, cats, params):
    cols = nums + cats
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    pool_tr = Pool(Xtr, ytr, cat_features=cats)
    pool_va = Pool(Xva, yva, cat_features=cats)
    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
    return model.predict(pool_va), model.best_iteration_


skf = StratifiedKFold(5, shuffle=True, random_state=2026)
oofA = np.zeros(len(y))
oofB = np.zeros(len(y))

paramsA = dict(
    loss_function="RMSE",
    iterations=800,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=10,
    random_seed=42,
    od_type="Iter",
    od_wait=60,
    allow_writing_files=False,
    thread_count=4,
    boosting_type="Ordered",
    rsm=1.0,
)
paramsB = dict(
    loss_function="RMSE",
    iterations=800,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=6,
    random_seed=43,
    od_type="Iter",
    od_wait=60,
    allow_writing_files=False,
    thread_count=4,
    boosting_type="Plain",
    rsm=0.3,
)

for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
    print("fold", fold, flush=True)
    trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
    ytr, yva = y[tr_i], y[va_i]
    oofA[va_i], itA = fit_cb(trn, val, ytr, yva, NUM_MAIN, CATS, {**paramsA, "random_seed": 42 + fold})
    oofB[va_i], itB = fit_cb(trn, val, ytr, yva, NUM_ALT, CATS, {**paramsB, "random_seed": 142 + fold})
    print(
        f"  A={roc_auc_score(yva, oofA[va_i]):.4f} it={itA}  B={roc_auc_score(yva, oofB[va_i]):.4f} it={itB}",
        flush=True,
    )


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


print("HONEST CatBoost 5fold armA", roc_auc_score(y, oofA))
print("HONEST CatBoost 5fold armB", roc_auc_score(y, oofB))
print("w62", roc_auc_score(y, 0.62 * rank(oofA) + 0.38 * rank(oofB)))
print("w50", roc_auc_score(y, 0.5 * rank(oofA) + 0.5 * rank(oofB)))
print("max", roc_auc_score(y, np.maximum(rank(oofA), rank(oofB))))
np.save("/workspace/analysis/cb_oofA.npy", oofA)
np.save("/workspace/analysis/cb_oofB.npy", oofB)
print("DONE")
