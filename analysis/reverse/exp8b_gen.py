#!/usr/bin/env python3
"""Extra CatBoost seeds with generating-process numerics + days_q5 (exp2 winner) + nested freq score.

Honest: inner ES on train fold only. Checkpoint per fold/seed. 10-fold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from common import (
    auc,
    dump_json,
    fill_condition,
    load_train,
    numeric_block,
    per_source_rank,
    qcut_apply,
    rank01,
    save_oof,
    skf,
)
from exp3_freq import additive_design
from exp3b_rich import source_b_map

OUT = Path("/workspace/analysis/reverse")
SEEDS = [2036, 2046, 2056, 2066]
N_BAGS = 2
CKPT = OUT / "exp8b_gen_ckpt.npz"

CATS = [
    "src",
    "reg",
    "age",
    "src_reg",
    "src_age",
    "reg_age",
    "src_cq",
    "src_dq5",
    "reg_cq",
    "reg_dq5",
    "days_q5",
    "cond_q10",
    "src_ratioq",
    "grades_s",
]


def build(trn0, val0):
    trn, val = trn0.copy(), val0.copy()
    cond_tr, cond_va = fill_condition(trn, val)
    Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr)
    Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr)
    for c in Ntr.columns:
        trn[c] = Ntr[c].to_numpy()
        val[c] = Nva[c].to_numpy()
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    ytr = trn0["label"].to_numpy(float) if "label" in trn0 else None
    days_tr = trn["days"].to_numpy(float)
    days_va = val["days"].to_numpy(float)
    if ytr is not None:
        bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
    else:
        bmap = {s: 0.5 for s in np.unique(src_tr)}
    b_tr = np.array([bmap.get(s, 0.5) for s in src_tr])
    b_va = np.array([bmap.get(s, 0.5) for s in src_va])
    trn["pow_ratio"] = days_tr / np.power(np.clip(cond_tr, 1e-6, None), b_tr)
    val["pow_ratio"] = days_va / np.power(np.clip(cond_va, 1e-6, None), b_va)
    trn["pow_rate15"] = np.power(np.clip(days_tr, 1, None), 1.5) * np.power(np.clip(1 - trn["cond_rk"].to_numpy(), 1e-6, None), 1.2)
    val["pow_rate15"] = np.power(np.clip(days_va, 1, None), 1.5) * np.power(np.clip(1 - val["cond_rk"].to_numpy(), 1e-6, None), 1.2)
    trn["w_hot9374"] = ((days_tr >= 9370) & (days_tr < 9475)).astype(float)
    val["w_hot9374"] = ((days_va >= 9370) & (days_va < 9475)).astype(float)
    d5_tr, d5_va, _ = qcut_apply(days_tr, days_va, 5)
    c10_tr, c10_va, _ = qcut_apply(cond_tr, cond_va, 10)
    rq_tr, rq_va, _ = qcut_apply(trn["ratio"], val["ratio"], 10)
    for df, d5, c10, rq in ((trn, d5_tr, c10_tr, rq_tr), (val, d5_va, c10_va, rq_va)):
        df["days_q5"] = pd.Series(d5, index=df.index).astype(str)
        df["cond_q10"] = pd.Series(c10, index=df.index).astype(str)
        df["ratio_q"] = pd.Series(rq, index=df.index).astype(str)
        df["src"] = df["source"].astype(str)
        df["reg"] = df["region"].astype(str)
        df["age"] = df["age_range"].astype(int).astype(str)
        df["grades_s"] = df["grades"].astype(str)
        df["src_reg"] = df["src"] + "|" + df["reg"]
        df["src_age"] = df["src"] + "|" + df["age"]
        df["reg_age"] = df["reg"] + "|" + df["age"]
        df["src_cq"] = df["src"] + "|" + df["cond_q10"]
        df["src_dq5"] = df["src"] + "|" + df["days_q5"]
        df["reg_cq"] = df["reg"] + "|" + df["cond_q10"]
        df["reg_dq5"] = df["reg"] + "|" + df["days_q5"]
        df["src_ratioq"] = df["src"] + "|" + df["ratio_q"]
    # nested frequency score
    if ytr is not None:
        src_levels = sorted(pd.unique(src_tr))
        reg_levels = sorted(pd.unique(trn["region"].astype(str)))
        rk_tr = trn["cond_rk"].to_numpy()
        rk_va = val["cond_rk"].to_numpy()
        Xtr = additive_design(days_tr, cond_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"], src_levels, reg_levels)
        Xva = additive_design(days_va, cond_va, rk_va, src_va, val["region"].astype(str), val["age_range"], src_levels, reg_levels)
        rr = Ridge(alpha=8.0).fit(Xtr, ytr)
        trn["freq_score"] = rr.predict(Xtr)
        val["freq_score"] = rr.predict(Xva)
    return trn, val


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
    "pow_ratio",
    "pow_rate15",
    "ushape_car10",
    "mono_car1",
    "rev_car7",
    "age_range",
    "V",
    "x20",
    "x1",
    "x5",
    "age8",
    "cond_low",
    "w_safe750",
    "w_safe1750",
    "w_new50",
    "w_hot9374",
    "freq_score",
    "t3_num",
    "cond_miss",
]


def paramsA(seed):
    return dict(
        loss_function="RMSE",
        iterations=650,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        random_seed=seed,
        od_type="Iter",
        od_wait=55,
        allow_writing_files=False,
        thread_count=4,
        boosting_type="Ordered",
        rsm=1.0,
    )


def paramsB(seed):
    return dict(
        loss_function="RMSE",
        iterations=650,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6,
        random_seed=seed,
        od_type="Iter",
        od_wait=55,
        allow_writing_files=False,
        thread_count=4,
        boosting_type="Plain",
        rsm=0.3,
        subsample=0.8,
    )


def fit_one(trn, ytr, val, nums, cats, params, inner_seed):
    cols = [c for c in nums if c in trn.columns] + cats
    X = trn[cols].copy()
    Xva = val[cols].copy()
    for c in cats:
        X[c] = X[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    idx = np.arange(len(X))
    tr_in, es_in = train_test_split(idx, test_size=0.12, stratify=(ytr > 0.5).astype(int), random_state=inner_seed)
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(X.iloc[tr_in], ytr[tr_in], cat_features=cats),
        eval_set=Pool(X.iloc[es_in], ytr[es_in], cat_features=cats),
        use_best_model=True,
        verbose=False,
    )
    return model.predict(Pool(Xva, cat_features=cats))


def main():
    df, y = load_train()
    n = len(y)
    oofA = {s: np.zeros(n) for s in SEEDS}
    oofB = {s: np.zeros(n) for s in SEEDS}
    done = set()
    if CKPT.exists():
        z = np.load(CKPT, allow_pickle=True)
        done = set(map(str, z["done"].tolist()))
        for s in SEEDS:
            if f"A{s}" in z.files:
                oofA[s] = z[f"A{s}"]
            if f"B{s}" in z.files:
                oofB[s] = z[f"B{s}"]
        print("resume", sorted(done)[:8], "...", len(done))

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        need = any(f"{fold}-{s}-{b}-A" not in done or f"{fold}-{s}-{b}-B" not in done for s in SEEDS for b in range(N_BAGS))
        if not need:
            continue
        trn0, val0 = df.iloc[tr_i].copy(), df.iloc[va_i].copy()
        ytr, yva = y[tr_i].astype(float), y[va_i]
        trn, val = build(trn0, val0)
        for s in SEEDS:
            accA = np.zeros(len(va_i))
            accB = np.zeros(len(va_i))
            na = nb = 0
            for b in range(N_BAGS):
                ka, kb = f"{fold}-{s}-{b}-A", f"{fold}-{s}-{b}-B"
                if ka not in done:
                    pred = fit_one(trn, ytr, val, NUMS, CATS, paramsA(s + fold * 3 + b), inner_seed=s + fold + b)
                    accA += pred
                    na += 1
                    done.add(ka)
                    print(f"fold {fold} seed {s} bag {b} A={auc(yva, pred):.4f}", flush=True)
                if kb not in done:
                    pred = fit_one(trn, ytr, val, NUMS, CATS, paramsB(s + 100 + fold * 3 + b), inner_seed=s + 50 + fold + b)
                    accB += pred
                    nb += 1
                    done.add(kb)
                    print(f"fold {fold} seed {s} bag {b} B={auc(yva, pred):.4f}", flush=True)
            if na:
                # if resuming mid-fold this is incomplete; we store bag-mean only when all bags of this seed/fold done
                pass
            # always recompute from scratch store: keep running mean in oof if all bags done
            nA = sum(1 for b in range(N_BAGS) if f"{fold}-{s}-{b}-A" in done)
            nB = sum(1 for b in range(N_BAGS) if f"{fold}-{s}-{b}-B" in done)
            if na == N_BAGS:
                oofA[s][va_i] = accA / N_BAGS
            if nb == N_BAGS:
                oofB[s][va_i] = accB / N_BAGS
            np.savez(
                CKPT,
                done=np.array(list(done), dtype=object),
                **{f"A{s}": oofA[s] for s in SEEDS},
                **{f"B{s}": oofB[s] for s in SEEDS},
            )

    A = np.mean([rank01(oofA[s]) for s in SEEDS], axis=0)
    B = np.mean([rank01(oofB[s]) for s in SEEDS], axis=0)
    rawA = np.mean([oofA[s] for s in SEEDS], axis=0)
    rawB = np.mean([oofB[s] for s in SEEDS], axis=0)
    w62 = 0.62 * A + 0.38 * B
    report = {
        "per_seed_A": {str(s): auc(y, oofA[s]) for s in SEEDS},
        "per_seed_B": {str(s): auc(y, oofB[s]) for s in SEEDS},
        "meanA": auc(y, rawA),
        "meanB": auc(y, rawB),
        "w62": auc(y, w62),
        "max2": auc(y, np.maximum(A, B)),
    }
    print("=== EXP8b GEN CB ===", json.dumps(report, indent=2))
    dump_json("exp8b_gen.json", report)
    save_oof("exp8b_gen_A.npy", rawA)
    save_oof("exp8b_gen_B.npy", rawB)
    save_oof("exp8b_gen_w62.npy", w62)


if __name__ == "__main__":
    main()
