#!/usr/bin/env python3
"""Honest protocol: numeric GBT + standalone TE scores, no sparse-TE overfitting."""
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
train["cond_r"] = train["condition_f"] / gmed.replace(0, np.nan)
train["cond_rk"] = train.groupby("source")["condition_f"].rank(pct=True)
train["ratio"] = train["days"] / train["cond_r"]
train["rate"] = train["days"] * (1.0 - train["cond_rk"])
train["ratio_sqrt"] = train["days"] / np.sqrt(train["condition_f"].clip(1e-6))
train["u_shape"] = (train["cond_rk"] - 0.5) ** 2
train["days_log"] = np.log1p(train["days"])
train["inv_cond"] = 1.0 / train["condition_f"].clip(1e-4)
train["age8"] = (train["age_range"] >= 8).astype(int)
train["cond_low"] = (train["condition_f"] < 0.05).astype(int)
train["safe_750"] = ((train["days"] >= 700) & (train["days"] < 880)).astype(int)

NUMS = [
    "days",
    "days_log",
    "condition_f",
    "cond_r",
    "cond_rk",
    "ratio",
    "rate",
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
]


def qbins(tr, va, q=10):
    cats, bins = pd.qcut(tr, q, duplicates="drop", retbins=True, labels=False)
    bins = bins.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    v = pd.cut(va, bins=bins, labels=False, include_lowest=True).fillna(0).astype(int)
    return cats.astype(int).astype(str).to_numpy(), v.astype(str).to_numpy()


def te_apply(tr_keys, va_keys, ytr, m=20.0):
    prior = float(ytr.mean())
    ssum, scnt = {}, {}
    for k, yi in zip(tr_keys, ytr):
        ssum[k] = ssum.get(k, 0.0) + yi
        scnt[k] = scnt.get(k, 0) + 1
    # LOO train
    tr_enc = np.empty(len(tr_keys))
    for i, k in enumerate(tr_keys):
        tr_enc[i] = (ssum[k] - ytr[i] + prior * m) / (scnt[k] - 1 + m)
    va_enc = np.array([(ssum[k] + prior * m) / (scnt[k] + m) if k in scnt else prior for k in va_keys])
    return tr_enc, va_enc


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


skf = StratifiedKFold(10, shuffle=True, random_state=2026)
n = len(y)
oof_num = np.zeros(n)
oof_te3 = np.zeros(n)
oof_te_src_cq = np.zeros(n)
oof_te_cqdq = np.zeros(n)
oof_te_regdq = np.zeros(n)
oof_te_ratio = np.zeros(n)
oof_blend_lgb = np.zeros(n)  # lgb on nums + lowcard te only

for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
    trn, val = train.iloc[tr_i], train.iloc[va_i]
    ytr, yva = y[tr_i], y[va_i]
    dq_tr, dq_va = qbins(trn["days"], val["days"], 10)
    cq_tr, cq_va = qbins(trn["condition_f"], val["condition_f"], 10)
    rq_tr, rq_va = qbins(trn["ratio"], val["ratio"], 10)
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    reg_tr = trn["region"].astype(str).to_numpy()
    reg_va = val["region"].astype(str).to_numpy()
    age_tr = trn["age_range"].astype(int).astype(str).to_numpy()
    age_va = val["age_range"].astype(int).astype(str).to_numpy()

    k3_tr = src_tr + "|" + cq_tr + "|" + dq_tr
    k3_va = src_va + "|" + cq_va + "|" + dq_va
    src_cq_tr, src_cq_va = src_tr + "|" + cq_tr, src_va + "|" + cq_va
    cqdq_tr, cqdq_va = cq_tr + "|" + dq_tr, cq_va + "|" + dq_va
    regdq_tr, regdq_va = reg_tr + "|" + dq_tr, reg_va + "|" + dq_va
    srcrq_tr, srcrq_va = src_tr + "|" + rq_tr, src_va + "|" + rq_va
    srcreg_tr, srcreg_va = src_tr + "|" + reg_tr, src_va + "|" + reg_va
    regage_tr, regage_va = reg_tr + "|" + age_tr, reg_va + "|" + age_va

    _, oof_te3[va_i] = te_apply(k3_tr, k3_va, ytr, m=15)
    _, oof_te_src_cq[va_i] = te_apply(src_cq_tr, src_cq_va, ytr, m=15)
    _, oof_te_cqdq[va_i] = te_apply(cqdq_tr, cqdq_va, ytr, m=12)
    _, oof_te_regdq[va_i] = te_apply(regdq_tr, regdq_va, ytr, m=15)
    _, oof_te_ratio[va_i] = te_apply(srcrq_tr, srcrq_va, ytr, m=15)

    # numeric GBT
    Xtr = trn[NUMS].to_numpy(float)
    Xva = val[NUMS].to_numpy(float)
    m = lgb.train(
        dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            min_data_in_leaf=80,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=10,
            verbose=-1,
            seed=fold,
        ),
        lgb.Dataset(Xtr, ytr),
        num_boost_round=600,
        valid_sets=[lgb.Dataset(Xva, yva)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_num[va_i] = m.predict(Xva)

    # low-card TE features only (source, region, src_reg, src_cq, age, reg_age) + nums
    tes = []
    tes_va = []
    for a, b, mm in [
        (src_tr, src_va, 10),
        (reg_tr, reg_va, 10),
        (age_tr, age_va, 10),
        (srcreg_tr, srcreg_va, 15),
        (src_cq_tr, src_cq_va, 15),
        (regage_tr, regage_va, 15),
        (cqdq_tr, cqdq_va, 20),
    ]:
        t1, t2 = te_apply(a, b, ytr, m=mm)
        tes.append(t1)
        tes_va.append(t2)
    Xtr2 = np.column_stack([Xtr] + tes)
    Xva2 = np.column_stack([Xva] + tes_va)
    m2 = lgb.train(
        dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            min_data_in_leaf=90,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=12,
            verbose=-1,
            seed=100 + fold,
        ),
        lgb.Dataset(Xtr2, ytr),
        num_boost_round=600,
        valid_sets=[lgb.Dataset(Xva2, yva)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_blend_lgb[va_i] = m2.predict(Xva2)
    print(
        f"fold {fold} num={roc_auc_score(yva, oof_num[va_i]):.4f} "
        f"te3={roc_auc_score(yva, oof_te3[va_i]):.4f} "
        f"lowte={roc_auc_score(yva, oof_blend_lgb[va_i]):.4f} best={m.best_iteration}/{m2.best_iteration}"
    )

print("\n=== HONEST OOF ===")
print("numeric GBT", roc_auc_score(y, oof_num))
print("te src_cq_dq", roc_auc_score(y, oof_te3))
print("te src_cq", roc_auc_score(y, oof_te_src_cq))
print("te cq_dq", roc_auc_score(y, oof_te_cqdq))
print("te reg_dq", roc_auc_score(y, oof_te_regdq))
print("te src_ratioq", roc_auc_score(y, oof_te_ratio))
print("lgb num+lowcard TE", roc_auc_score(y, oof_blend_lgb))

# rank blends
cands = {
    "num": oof_num,
    "te3": oof_te3,
    "src_cq": oof_te_src_cq,
    "cqdq": oof_te_cqdq,
    "regdq": oof_te_regdq,
    "srcrq": oof_te_ratio,
    "lowte": oof_blend_lgb,
}
# greedy pairwise
print("\n=== rank blends ===")
print("0.6 num + 0.4 te3", roc_auc_score(y, 0.6 * rank(oof_num) + 0.4 * rank(oof_te3)))
print("0.5 num + 0.5 te3", roc_auc_score(y, 0.5 * rank(oof_num) + 0.5 * rank(oof_te3)))
print("0.5 lowte + 0.5 te3", roc_auc_score(y, 0.5 * rank(oof_blend_lgb) + 0.5 * rank(oof_te3)))
print(
    "eq 4way num/te3/cqdq/regdq",
    roc_auc_score(y, rank(oof_num) + rank(oof_te3) + rank(oof_te_cqdq) + rank(oof_te_regdq)),
)
print(
    "w num0.35 te3 0.25 cqdq 0.15 regdq 0.10 srcrq 0.15",
    roc_auc_score(
        y,
        0.35 * rank(oof_num)
        + 0.25 * rank(oof_te3)
        + 0.15 * rank(oof_te_cqdq)
        + 0.10 * rank(oof_te_regdq)
        + 0.15 * rank(oof_te_ratio),
    ),
)
print(
    "lowte 0.4 + te3 0.25 + cqdq 0.15 + srcrq 0.2",
    roc_auc_score(y, 0.4 * rank(oof_blend_lgb) + 0.25 * rank(oof_te3) + 0.15 * rank(oof_te_cqdq) + 0.2 * rank(oof_te_ratio)),
)
print(
    "all equal",
    roc_auc_score(y, sum(rank(v) for v in cands.values())),
)
print("DONE")
