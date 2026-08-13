#!/usr/bin/env python3
"""Val-ES CatBoost with the full W62 2-way cat list (no 3-way src_cq_dq as cat). 10-fold x 3-bag x 2-seed."""
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
from cb_features import CATS, NUM_ALT, NUM_MAIN, fold_features

OUT = Path("/workspace/analysis/reverse")
SEEDS = [3010, 3020]
N_BAGS = 3
THREADS = 4


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit_vales(trn, val, ytr, yva, nums, params):
    cols = [c for c in nums if c in trn.columns] + CATS
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in CATS:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(Xtr, ytr, cat_features=CATS),
        eval_set=Pool(Xva, yva, cat_features=CATS),
        use_best_model=True,
        verbose=False,
    )
    return model.predict(Pool(Xva, cat_features=CATS))


def p_main(seed):
    return dict(
        loss_function="RMSE",
        iterations=800,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Ordered",
        rsm=1.0,
    )


def p_alt(seed):
    return dict(
        loss_function="RMSE",
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6,
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Plain",
        rsm=0.3,
    )


def main():
    print("CATS", CATS, flush=True)
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    n = len(y)
    oofA = {s: np.zeros(n) for s in SEEDS}
    oofB = {s: np.zeros(n) for s in SEEDS}
    skf = StratifiedKFold(10, shuffle=True, random_state=2026)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        t0 = time.time()
        for s in SEEDS:
            pm = np.zeros(len(va_i))
            pa = np.zeros(len(va_i))
            for b in range(N_BAGS):
                pm += fit_vales(trn, val, ytr, yva, NUM_MAIN, p_main(s + 10 * b + fold))
                pa += fit_vales(trn, val, ytr, yva, NUM_ALT, p_alt(s + 100 + 10 * b + fold))
            oofA[s][va_i] = pm / N_BAGS
            oofB[s][va_i] = pa / N_BAGS
            print(
                f"fold {fold} seed {s} A={roc_auc_score(yva, oofA[s][va_i]):.4f} B={roc_auc_score(yva, oofB[s][va_i]):.4f}",
                flush=True,
            )
        print(f"  fold {fold} dt={time.time()-t0:.1f}s", flush=True)
        np.save(OUT / "exp8j_A.npy", np.mean([oofA[s] for s in SEEDS], axis=0))
        np.save(OUT / "exp8j_B.npy", np.mean([oofB[s] for s in SEEDS], axis=0))

    rA = np.mean([rank(oofA[s]) for s in SEEDS], axis=0)
    rB = np.mean([rank(oofB[s]) for s in SEEDS], axis=0)
    w62 = 0.62 * rA + 0.38 * rB
    mx = np.maximum(rA, rB)
    report = {
        "A": {str(s): float(roc_auc_score(y, oofA[s])) for s in SEEDS},
        "B": {str(s): float(roc_auc_score(y, oofB[s])) for s in SEEDS},
        "w62": float(roc_auc_score(y, w62)),
        "max2": float(roc_auc_score(y, mx)),
        "n_cats": len(CATS),
    }
    print("=== EXP8j FULL 2WAY CATS ===", json.dumps(report, indent=2))
    (OUT / "exp8j_fullcats.json").write_text(json.dumps(report, indent=2))
    np.save(OUT / "exp8j_max2.npy", mx)
    # blend with previous 8seed vales max2
    prev = OUT / "exp8h_max2.npy"
    lgb = OUT / "exp8d_lgb_w62.npy"
    best, name, ba = mx, "fullcats_max2", float(roc_auc_score(y, mx))
    if prev.exists():
        p = np.load(prev)
        s = 0.5 * rank(p) + 0.5 * mx
        a = float(roc_auc_score(y, s))
        print("blend 8seed_vales + fullcats", a)
        if a > ba:
            best, name, ba = s, "vales_fullcats", a
    if lgb.exists():
        s = 0.82 * mx + 0.18 * rank(np.load(lgb))
        a = float(roc_auc_score(y, s))
        print("fullcats+lgb", a)
        if a > ba:
            best, name, ba = s, "fullcats_lgb", a
        if prev.exists():
            s = 0.62 * rank(np.load(prev)) + 0.25 * mx + 0.13 * rank(np.load(lgb))
            a = float(roc_auc_score(y, s))
            print("3way vales+fullcats+lgb", a)
            if a > ba:
                best, name, ba = s, "3way", a
    np.save(OUT / "best_oof.npy", best)
    print("BEST", name, ba)
    if ba >= 0.70:
        print("HIT 0.70")


if __name__ == "__main__":
    main()
