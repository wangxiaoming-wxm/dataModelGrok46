#!/usr/bin/env python3
"""Extra W62-recipe CatBoost seeds on the same 10 folds as teacher 2026. Rank-average toward 0.70."""
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
from cb_features import CATS, NUM_ALT, NUM_MAIN, fold_features

OUT = Path("/workspace/analysis/reverse")
SEEDS = [2027, 2028, 2029, 2030, 2031, 2032]
N_BAGS = 2
N_FOLDS = 10
SKF_SEED = 2026
ITERS = 800
THREADS = 4
CKPT = OUT / "exp8f_seeds_ckpt.npz"
TEACHER_CSV = Path("/workspace/submissions/cb_teacher_oof.csv")


def auc(y, s):
    return float(roc_auc_score(y, s))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def params_main(seed: int) -> dict:
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


def params_alt(seed: int) -> dict:
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


def fit_predict(trn, val, ytr, nums, params):
    cols = nums + CATS
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in CATS:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    strat = (ytr > 0.5).astype(int)
    Xa, Xb, ya, yb = train_test_split(
        Xtr, ytr, test_size=0.12, random_state=int(params.get("random_seed", 0)), stratify=strat
    )
    model = CatBoostRegressor(**params)
    kw = dict(verbose=False)
    if len(yb) >= 30:
        kw["eval_set"] = Pool(Xb, yb, cat_features=CATS)
        kw["use_best_model"] = True
    model.fit(Pool(Xa, ya, cat_features=CATS), **kw)
    return model.predict(Pool(Xva, cat_features=CATS))


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    n = len(y)
    oofA = {s: np.zeros(n) for s in SEEDS}
    oofB = {s: np.zeros(n) for s in SEEDS}
    done_fold = {s: -1 for s in SEEDS}
    if CKPT.exists():
        z = np.load(CKPT, allow_pickle=True)
        for s in SEEDS:
            if f"A{s}" in z.files:
                oofA[s] = z[f"A{s}"]
            if f"B{s}" in z.files:
                oofB[s] = z[f"B{s}"]
            if f"f{s}" in z.files:
                done_fold[s] = int(z[f"f{s}"])
        print("resume", done_fold, flush=True)

    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SKF_SEED).split(train, y))
    for fold, (tr_i, va_i) in enumerate(folds):
        need = [s for s in SEEDS if done_fold[s] < fold]
        if not need:
            continue
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        t0 = time.time()
        for s in need:
            pm = np.zeros(len(va_i))
            pa = np.zeros(len(va_i))
            for b in range(N_BAGS):
                sd = s + 10 * b
                pm += fit_predict(trn, val, ytr, NUM_MAIN, params_main(sd + fold))
                pa += fit_predict(trn, val, ytr, NUM_ALT, params_alt(sd + 100 + fold))
            oofA[s][va_i] = pm / N_BAGS
            oofB[s][va_i] = pa / N_BAGS
            done_fold[s] = fold
            print(f"fold {fold} seed {s} A={auc(yva, oofA[s][va_i]):.4f} B={auc(yva, oofB[s][va_i]):.4f}", flush=True)
        np.savez(
            CKPT,
            **{f"A{s}": oofA[s] for s in SEEDS},
            **{f"B{s}": oofB[s] for s in SEEDS},
            **{f"f{s}": done_fold[s] for s in SEEDS},
        )
        print(f"  fold {fold} dt={time.time()-t0:.1f}s", flush=True)

    tea = pd.read_csv(TEACHER_CSV).set_index("id").loc[train["id"].astype(str)]
    arms_A = [tea["pred_cb_main"].to_numpy()] + [oofA[s] for s in SEEDS]
    arms_B = [tea["pred_cb_alt"].to_numpy()] + [oofB[s] for s in SEEDS]
    rA = np.mean([rank(a) for a in arms_A], axis=0)
    rB = np.mean([rank(a) for a in arms_B], axis=0)
    w62 = 0.62 * rA + 0.38 * rB
    mx = np.maximum(rA, rB)
    rawA = np.mean(arms_A, axis=0)
    rawB = np.mean(arms_B, axis=0)
    report = {
        "teacher_A": auc(y, tea["pred_cb_main"]),
        "teacher_B": auc(y, tea["pred_cb_alt"]),
        "per_extra": {str(s): {"A": auc(y, oofA[s]), "B": auc(y, oofB[s])} for s in SEEDS},
        "pool_raw_A": auc(y, rawA),
        "pool_raw_B": auc(y, rawB),
        "rank_mean_A": auc(y, rA),
        "rank_mean_B": auc(y, rB),
        "w62": auc(y, w62),
        "max2": auc(y, mx),
        "n_seeds_total": 1 + len(SEEDS),
        "n_bags_extra": N_BAGS,
    }
    print("=== EXP8f MULTI-SEED ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    (OUT / "exp8f_seeds.json").write_text(json.dumps(report, indent=2, default=str))
    np.save(OUT / "exp8f_w62.npy", w62)
    np.save(OUT / "exp8f_max2.npy", mx)

    best, best_name, best_auc = mx, "max2_multiseed", auc(y, mx)
    lgb_p = OUT / "exp8d_lgb_w62.npy"
    if lgb_p.exists():
        rl = rank(np.load(lgb_p))
        for w, name in [(0.92, "ms0.92_lgb"), (0.88, "ms0.88_lgb"), (0.84, "ms0.84_lgb")]:
            s = w * mx + (1 - w) * rl
            a = auc(y, s)
            print(f"  {name}: {a:.5f}")
            if a > best_auc:
                best, best_name, best_auc = s, name, a
        s = 0.88 * w62 + 0.12 * rl
        a = auc(y, s)
        print(f"  w62_0.88_lgb: {a:.5f}")
        if a > best_auc:
            best, best_name, best_auc = s, "w62_0.88_lgb", a
    np.save(OUT / "best_oof.npy", best)
    print("BEST", best_name, best_auc)


if __name__ == "__main__":
    main()
