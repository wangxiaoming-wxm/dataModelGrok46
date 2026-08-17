#!/usr/bin/env python3
"""Diagnostic: add 3-way cats to CatBoost val-ES (the thing notes say not to). 10-fold x 2-bag x 1-seed."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cb_features import NUM_ALT, NUM_MAIN, fold_features

OUT = Path("/workspace/analysis/reverse")
CATS3 = [
    "source",
    "region",
    "age_range",
    "grades",
    "month",
    "src_reg",
    "src_age",
    "reg_age",
    "src_cq",
    "src_dq",
    "days_q",
    "cond_q",
    "src_ratioq",
    "cq_dq",
    "src_cq_dq",
    "src_cq_age",
    "reg_cq_dq",
]


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit_vales(trn, val, ytr, yva, nums, params, cats):
    cols = [c for c in nums if c in trn.columns] + cats
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(Xtr, ytr, cat_features=cats),
        eval_set=Pool(Xva, yva, cat_features=cats),
        use_best_model=True,
        verbose=False,
    )
    return model.predict(Pool(Xva, cat_features=cats))


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    n = len(y)
    oofA = np.zeros(n)
    oofB = np.zeros(n)
    skf = StratifiedKFold(10, shuffle=True, random_state=2026)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        t0 = time.time()
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        for df in (trn, val):
            df["src_cq_age"] = df["source"].astype(str) + "|" + df["cond_q"].astype(str) + "|" + df["age_range"].astype(str)
            df["reg_cq_dq"] = df["region"].astype(str) + "|" + df["cond_q"].astype(str) + "|" + df["days_q"].astype(str)
        ytr, yva = y[tr_i], y[va_i]
        pm = np.zeros(len(va_i))
        pa = np.zeros(len(va_i))
        for b in range(2):
            pm += fit_vales(
                trn,
                val,
                ytr,
                yva,
                NUM_MAIN,
                dict(
                    loss_function="RMSE",
                    iterations=800,
                    learning_rate=0.03,
                    depth=5,
                    l2_leaf_reg=10,
                    random_seed=4000 + b + fold,
                    od_type="Iter",
                    od_wait=80,
                    allow_writing_files=False,
                    thread_count=4,
                    boosting_type="Ordered",
                    rsm=1.0,
                ),
                CATS3,
            )
            pa += fit_vales(
                trn,
                val,
                ytr,
                yva,
                NUM_ALT,
                dict(
                    loss_function="RMSE",
                    iterations=800,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=6,
                    random_seed=4100 + b + fold,
                    od_type="Iter",
                    od_wait=80,
                    allow_writing_files=False,
                    thread_count=4,
                    boosting_type="Plain",
                    rsm=0.3,
                ),
                CATS3,
            )
        oofA[va_i] = pm / 2
        oofB[va_i] = pa / 2
        print(
            f"fold {fold} A={roc_auc_score(yva, oofA[va_i]):.4f} B={roc_auc_score(yva, oofB[va_i]):.4f} dt={time.time()-t0:.1f}s",
            flush=True,
        )
    mx = np.maximum(rank(oofA), rank(oofB))
    w62 = 0.62 * rank(oofA) + 0.38 * rank(oofB)
    report = {
        "A": float(roc_auc_score(y, oofA)),
        "B": float(roc_auc_score(y, oofB)),
        "w62": float(roc_auc_score(y, w62)),
        "max2": float(roc_auc_score(y, mx)),
        "note": "3-way cats in CatBoost + val-ES; diagnostic only",
    }
    print("=== EXP8k 3WAY CATS ===", json.dumps(report, indent=2))
    (OUT / "exp8k_3way.json").write_text(json.dumps(report, indent=2))
    np.save(OUT / "exp8k_max2.npy", mx)
    prev = np.load(OUT / "exp8h_max2.npy")
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    s = 0.55 * rank(prev) + 0.30 * mx + 0.15 * rank(lgb)
    a = float(roc_auc_score(y, s))
    print("blend 8seed+3way+lgb", a)
    best, name, ba = mx, "3way_max2", report["max2"]
    if a > ba:
        best, name, ba = s, "blend", a
    if report["max2"] > 0.69676:
        np.save(OUT / "best_oof.npy", best)
    print("BEST", name, ba)
    if ba >= 0.70:
        print("HIT 0.70")


if __name__ == "__main__":
    main()
