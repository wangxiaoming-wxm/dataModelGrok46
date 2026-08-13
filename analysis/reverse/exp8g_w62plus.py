#!/usr/bin/env python3
"""W62 CatBoost + generating-process extras. 10-fold x 3-bag x 1 seed, inner ES. Compare to teacher 0.692."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cb_features import CATS, NUM_ALT, NUM_MAIN, fold_features, te_apply

OUT = Path("/workspace/analysis/reverse")
N_BAGS = 3
SEED0 = 2099
THREADS = 4
ITERS = 800


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit_predict(trn, val, ytr, nums, params):
    cols = nums + CATS
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in CATS:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    Xa, Xb, ya, yb = train_test_split(
        Xtr, ytr, test_size=0.12, random_state=int(params.get("random_seed", 0)), stratify=(ytr > 0.5).astype(int)
    )
    model = CatBoostRegressor(**params)
    kw = dict(verbose=False, eval_set=Pool(Xb, yb, cat_features=CATS), use_best_model=True)
    model.fit(Pool(Xa, ya, cat_features=CATS), **kw)
    return model.predict(Pool(Xva, cat_features=CATS))


def p_main(seed):
    return dict(
        loss_function="RMSE",
        iterations=ITERS,
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
        iterations=ITERS,
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
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    n = len(y)
    oofA = np.zeros(n)
    oofB = np.zeros(n)
    oof_te = np.zeros(n)
    skf = StratifiedKFold(10, shuffle=True, random_state=2026)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        t0 = time.time()
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        pm = np.zeros(len(va_i))
        pa = np.zeros(len(va_i))
        for b in range(N_BAGS):
            sd = SEED0 + 10 * b
            pm += fit_predict(trn, val, ytr, NUM_MAIN, p_main(sd + fold))
            pa += fit_predict(trn, val, ytr, NUM_ALT, p_alt(sd + 100 + fold))
            print(f"  fold {fold} bag {b} done", flush=True)
        oofA[va_i] = pm / N_BAGS
        oofB[va_i] = pa / N_BAGS
        oof_te[va_i] = te_apply(
            (trn["source"] + "|" + trn["cond_q"].astype(str) + "|" + trn["days_q5"].astype(str)).to_numpy(),
            (val["source"] + "|" + val["cond_q"].astype(str) + "|" + val["days_q5"].astype(str)).to_numpy(),
            ytr,
            m=10,
        )
        print(
            f"[fold {fold}] A={roc_auc_score(yva, oofA[va_i]):.5f} B={roc_auc_score(yva, oofB[va_i]):.5f} "
            f"te5={roc_auc_score(yva, oof_te[va_i]):.5f} dt={time.time()-t0:.1f}s",
            flush=True,
        )
        np.save(OUT / "exp8g_A.npy", oofA)
        np.save(OUT / "exp8g_B.npy", oofB)

    rA, rB = rank(oofA), rank(oofB)
    w62 = 0.62 * rA + 0.38 * rB
    mx = np.maximum(rA, rB)
    report = {
        "A": float(roc_auc_score(y, oofA)),
        "B": float(roc_auc_score(y, oofB)),
        "w62": float(roc_auc_score(y, w62)),
        "max2": float(roc_auc_score(y, mx)),
        "te_src_cq_dq5": float(roc_auc_score(y, oof_te)),
        "note": "W62 cats + days_q5/src_dq5 + pow_rate15 + car-shape flags + hot9374; 10fold 3bag innerES",
    }
    print("=== EXP8g W62+GEN ===", json.dumps(report, indent=2))
    (OUT / "exp8g_w62plus.json").write_text(json.dumps(report, indent=2))
    np.save(OUT / "exp8g_w62.npy", w62)
    np.save(OUT / "exp8g_max2.npy", mx)

    # fuse with teacher
    tea = pd.read_csv("/workspace/submissions/cb_teacher_oof.csv").set_index("id").loc[train["id"].astype(str)]
    tmx = np.maximum(rank(tea.pred_cb_main), rank(tea.pred_cb_alt))
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    blends = {
        "g_max2": mx,
        "0.5t_0.5g": 0.5 * tmx + 0.5 * mx,
        "0.6t_0.4g": 0.6 * tmx + 0.4 * mx,
        "0.7t_0.3g": 0.7 * tmx + 0.3 * mx,
        "g_lgb": 0.85 * mx + 0.15 * rank(lgb),
        "t_g_lgb": 0.70 * tmx + 0.18 * mx + 0.12 * rank(lgb),
    }
    best_name, best_s, best_a = None, None, -1
    for k, s in blends.items():
        a = float(roc_auc_score(y, s))
        print(f"  blend {k}: {a:.5f}")
        if a > best_a:
            best_name, best_s, best_a = k, s, a
    np.save(OUT / "best_oof.npy", best_s)
    print("BEST", best_name, best_a)


if __name__ == "__main__":
    main()
