#!/usr/bin/env python3
"""Exp8b: CatBoost 10-fold dual-world RMSE with generating-process numerics.

Early stopping uses an INNER split of the train fold (not the outer val) — honest OOF.
Checkpoints per fold. Multiple seeds. Rank fusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from common import auc, dump_json, fill_condition, load_train, qcut_apply, rank01, save_oof, skf, numeric_block, per_source_rank

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
    "cq_dq",
    "src_cq_dq",
    "src_ratioq",
    "src_rateq",
    "src_cq_age",
    "grades_s",
    "days_q",
    "cond_q",
]


def add_cats(trn, val, cond_tr, cond_va):
    trn = trn.copy()
    val = val.copy()
    trn["condition_f"] = cond_tr
    val["condition_f"] = cond_va
    Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr)
    Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr)
    for c in Ntr.columns:
        trn[c] = Ntr[c].to_numpy()
        val[c] = Nva[c].to_numpy()
    dtr, dva, _ = qcut_apply(trn["days"], val["days"], 10)
    ctr, cva, _ = qcut_apply(cond_tr, cond_va, 10)
    rtr, rva, _ = qcut_apply(trn["ratio"], val["ratio"], 10)
    ttr, tva, _ = qcut_apply(trn["rate"], val["rate"], 10)
    for df, d, c, r, t in ((trn, dtr, ctr, rtr, ttr), (val, dva, cva, rva, tva)):
        df["days_q"] = pd.Series(d, index=df.index).astype(str)
        df["cond_q"] = pd.Series(c, index=df.index).astype(str)
        df["ratio_q"] = pd.Series(r, index=df.index).astype(str)
        df["rate_q"] = pd.Series(t, index=df.index).astype(str)
        df["src"] = df["source"].astype(str)
        df["reg"] = df["region"].astype(str)
        df["age"] = df["age_range"].astype(int).astype(str)
        df["grades_s"] = df["grades"].astype(str)
        df["src_reg"] = df["src"] + "|" + df["reg"]
        df["src_age"] = df["src"] + "|" + df["age"]
        df["reg_age"] = df["reg"] + "|" + df["age"]
        df["src_cq"] = df["src"] + "|" + df["cond_q"]
        df["src_dq"] = df["src"] + "|" + df["days_q"]
        df["reg_cq"] = df["reg"] + "|" + df["cond_q"]
        df["reg_dq"] = df["reg"] + "|" + df["days_q"]
        df["cq_dq"] = df["cond_q"] + "|" + df["days_q"]
        df["src_cq_dq"] = df["src"] + "|" + df["cond_q"] + "|" + df["days_q"]
        df["src_ratioq"] = df["src"] + "|" + df["ratio_q"]
        df["src_rateq"] = df["src"] + "|" + df["rate_q"]
        df["src_cq_age"] = df["src"] + "|" + df["cond_q"] + "|" + df["age"]
    return trn, val


NUM_MAIN = [
    "days",
    "days_log",
    "condition_f",
    "cond_r",
    "ratio",
    "ratio_sqrt",
    "log_ratio",
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
    "w_safe750",
    "w_safe1750",
    "w_new50",
    "ushape_car10",
    "mono_car1",
    "rev_car7",
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
    "w_safe750",
    "w_safe1750",
    "ushape_car10",
    "mono_car1",
    "t3_num",
    "cond_miss",
]


def fit_cb(trn, ytr, val, nums, cats, params, inner_seed):
    cols = nums + cats
    # inner split of TRAIN for early stopping (honest)
    idx = np.arange(len(trn))
    tr_in, es_in = train_test_split(idx, test_size=0.12, stratify=ytr, random_state=inner_seed)
    X = trn[cols].copy()
    for c in cats:
        X[c] = X[c].astype(str)
    Xva = val[cols].copy()
    for c in cats:
        Xva[c] = Xva[c].astype(str)
    pool_tr = Pool(X.iloc[tr_in], ytr[tr_in], cat_features=cats)
    pool_es = Pool(X.iloc[es_in], ytr[es_in], cat_features=cats)
    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_es, use_best_model=True, verbose=False)
    return model.predict(Pool(Xva, cat_features=cats)), int(model.best_iteration_ or 0)


def main():
    df, y = load_train()
    n = len(y)
    seeds = [2026, 2027, 2028]
    oofA = {s: np.zeros(n) for s in seeds}
    oofB = {s: np.zeros(n) for s in seeds}

    paramsA = dict(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        thread_count=4,
        boosting_type="Ordered",
        rsm=1.0,
        bootstrap_type="Bayesian",
    )
    paramsB = dict(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        thread_count=4,
        boosting_type="Plain",
        rsm=0.3,
        subsample=0.8,
    )

    ckpt = Path("/workspace/analysis/reverse/cb_ckpt.npz")
    done = set()
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        done = set(z["done"].tolist())
        for s in seeds:
            if f"A{s}" in z:
                oofA[s] = z[f"A{s}"]
            if f"B{s}" in z:
                oofB[s] = z[f"B{s}"]
        print("resumed", done)

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn0, val0 = df.iloc[tr_i], df.iloc[va_i]
        ytr, yva = y[tr_i].astype(float), y[va_i]
        cond_tr, cond_va = fill_condition(trn0, val0)
        trn, val = add_cats(trn0, val0, cond_tr, cond_va)
        for s in seeds:
            keyA, keyB = f"{fold}-A-{s}", f"{fold}-B-{s}"
            if keyA not in done:
                pred, it = fit_cb(
                    trn, ytr, val, NUM_MAIN, CATS, {**paramsA, "random_seed": int(s + fold)}, inner_seed=s + fold
                )
                oofA[s][va_i] = pred
                done.add(keyA)
                print(f"fold {fold} seed {s} A={auc(yva, pred):.4f} it={it}", flush=True)
            if keyB not in done:
                pred, it = fit_cb(
                    trn, ytr, val, NUM_ALT, CATS, {**paramsB, "random_seed": int(s + 100 + fold)}, inner_seed=s + 50 + fold
                )
                oofB[s][va_i] = pred
                done.add(keyB)
                print(f"fold {fold} seed {s} B={auc(yva, pred):.4f} it={it}", flush=True)
            np.savez(
                ckpt,
                done=np.array(list(done), dtype=object),
                **{f"A{s}": oofA[s] for s in seeds},
                **{f"B{s}": oofB[s] for s in seeds},
            )

    # pool seeds
    A = np.mean([rank01(oofA[s]) for s in seeds], axis=0)
    B = np.mean([rank01(oofB[s]) for s in seeds], axis=0)
    rawA = np.mean([oofA[s] for s in seeds], axis=0)
    rawB = np.mean([oofB[s] for s in seeds], axis=0)
    fuse = 0.62 * A + 0.38 * B
    report = {
        "per_seed_A": {str(s): auc(y, oofA[s]) for s in seeds},
        "per_seed_B": {str(s): auc(y, oofB[s]) for s in seeds},
        "mean_raw_A": auc(y, rawA),
        "mean_raw_B": auc(y, rawB),
        "rank_mean_A": auc(y, A),
        "rank_mean_B": auc(y, B),
        "w62": auc(y, fuse),
        "w50": auc(y, 0.5 * A + 0.5 * B),
        "max2": auc(y, np.maximum(A, B)),
        "note": "10-fold, inner-ES on train fold only, 3 seeds, Ordered d5 + Plain d6 rsm0.3",
    }
    print("\n=== EXP8b CatBoost 10fold ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    dump_json("exp8b_catboost.json", report)
    save_oof("exp8b_oofA.npy", rawA)
    save_oof("exp8b_oofB.npy", rawB)
    save_oof("exp8b_w62.npy", fuse)
    print("WROTE exp8b_catboost.json")


if __name__ == "__main__":
    main()
