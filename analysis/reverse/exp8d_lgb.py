#!/usr/bin/env python3
"""LightGBM RMSE 10-fold with generating-process numerics + medium-card native cats. No high-card TE in trees."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import auc, dump_json, fill_condition, load_train, numeric_block, qcut_apply, rank01, save_oof, skf
from exp3b_rich import source_b_map


def main():
    df, y = load_train()
    n = len(y)
    oof1 = np.zeros(n)
    oof2 = np.zeros(n)
    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i].copy(), df.iloc[va_i].copy()
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr)
        Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr)
        src_tr = trn["source"].astype(str).to_numpy()
        days_tr = trn["days"].to_numpy(float)
        bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
        b_tr = np.array([bmap.get(s, 0.5) for s in src_tr])
        b_va = np.array([bmap.get(s, 0.5) for s in val["source"].astype(str).to_numpy()])
        Ntr = Ntr.copy()
        Nva = Nva.copy()
        Ntr["pow_ratio"] = days_tr / np.power(np.clip(cond_tr, 1e-6, None), b_tr)
        Nva["pow_ratio"] = val["days"].to_numpy(float) / np.power(np.clip(cond_va, 1e-6, None), b_va)
        Ntr["pow_rate15"] = np.power(np.clip(days_tr, 1, None), 1.5) * np.power(np.clip(1 - Ntr["cond_rk"], 1e-6, None), 1.2)
        Nva["pow_rate15"] = np.power(np.clip(val["days"].to_numpy(float), 1, None), 1.5) * np.power(
            np.clip(1 - Nva["cond_rk"], 1e-6, None), 1.2
        )
        dtr, dva, _ = qcut_apply(trn["days"], val["days"], 5)
        ctr, cva, _ = qcut_apply(cond_tr, cond_va, 10)
        for d, src, reg, age, dq, cq in (
            (Ntr, trn["source"].astype(str), trn["region"].astype(str), trn["age_range"].astype(int).astype(str), dtr, ctr),
            (Nva, val["source"].astype(str), val["region"].astype(str), val["age_range"].astype(int).astype(str), dva, cva),
        ):
            d["src"] = src.to_numpy() if hasattr(src, "to_numpy") else src
            d["reg"] = reg.to_numpy() if hasattr(reg, "to_numpy") else reg
            d["agec"] = age.to_numpy() if hasattr(age, "to_numpy") else age
            d["src_reg"] = d["src"] + "|" + d["reg"]
            d["src_cq"] = d["src"] + "|" + np.asarray(cq).astype(str)
            d["src_dq"] = d["src"] + "|" + np.asarray(dq).astype(str)
            d["reg_age"] = d["reg"] + "|" + d["agec"]
        cats = ["src", "reg", "agec", "src_reg", "src_cq", "src_dq", "reg_age"]
        for c in cats:
            Ntr[c] = Ntr[c].astype("category")
            Nva[c] = pd.Categorical(Nva[c], categories=list(Ntr[c].cat.categories))
        params1 = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            min_data_in_leaf=90,
            feature_fraction=0.85,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=10.0,
            verbose=-1,
            num_threads=1,
            seed=fold,
        )
        params2 = dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=40,
            max_depth=6,
            min_data_in_leaf=70,
            feature_fraction=0.35,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=6.0,
            verbose=-1,
            num_threads=1,
            seed=100 + fold,
        )
        dtr_lgb = lgb.Dataset(Ntr, ytr, categorical_feature=cats, free_raw_data=False)
        dva_lgb = lgb.Dataset(Nva, y[va_i], categorical_feature=cats, reference=dtr_lgb, free_raw_data=False)
        m1 = lgb.train(params1, dtr_lgb, num_boost_round=800, valid_sets=[dva_lgb], callbacks=[lgb.early_stopping(60, verbose=False)])
        m2 = lgb.train(params2, dtr_lgb, num_boost_round=800, valid_sets=[dva_lgb], callbacks=[lgb.early_stopping(60, verbose=False)])
        oof1[va_i] = m1.predict(Nva)
        oof2[va_i] = m2.predict(Nva)
        print(f"fold {fold} a={auc(y[va_i], oof1[va_i]):.4f} b={auc(y[va_i], oof2[va_i]):.4f} it={m1.best_iteration}/{m2.best_iteration}", flush=True)

    fuse = 0.62 * rank01(oof1) + 0.38 * rank01(oof2)
    report = {"lgb_main": auc(y, oof1), "lgb_alt": auc(y, oof2), "w62": auc(y, fuse)}
    print("=== LGB GEN ===", report)
    dump_json("exp8d_lgb.json", report)
    save_oof("exp8d_lgb_main.npy", oof1)
    save_oof("exp8d_lgb_alt.npy", oof2)
    save_oof("exp8d_lgb_w62.npy", fuse)


if __name__ == "__main__":
    main()
