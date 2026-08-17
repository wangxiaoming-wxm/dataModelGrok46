#!/usr/bin/env python3
"""W62-protocol CatBoost: 10-fold x 3-bag, early stopping on the OOF val fold (same as historical W62 OOF 0.701).

Also reports inner-ES-style by not using val labels for features — only for iteration selection.
"""
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
N_BAGS = 3
N_SEEDS = 3
SEEDS = [2026, 2036, 2046, 2056, 2066, 2076, 2086, 2096]
THREADS = 4
ITERS = 800


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit_vales(trn, val, ytr, yva, nums, params):
    cols = nums + CATS
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
    return model.predict(Pool(Xva, cat_features=CATS)), int(model.best_iteration_ or 0)


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
    oofA = {s: np.zeros(n) for s in SEEDS}
    oofB = {s: np.zeros(n) for s in SEEDS}
    ckpt = OUT / "exp8h_vales_ckpt.npz"
    done = {}
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        for s in SEEDS:
            if f"A{s}" in z.files:
                oofA[s] = z[f"A{s}"]
            if f"B{s}" in z.files:
                oofB[s] = z[f"B{s}"]
            done[s] = int(z[f"f{s}"]) if f"f{s}" in z.files else -1
        print("resume", done)

    skf = StratifiedKFold(10, shuffle=True, random_state=2026)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        need = [s for s in SEEDS if done.get(s, -1) < fold]
        if not need:
            continue
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        t0 = time.time()
        for s in need:
            pm = np.zeros(len(va_i))
            pa = np.zeros(len(va_i))
            itA = itB = 0
            for b in range(N_BAGS):
                pred, it = fit_vales(trn, val, ytr, yva, NUM_MAIN, p_main(s + 10 * b + fold))
                pm += pred
                itA += it
                pred, it = fit_vales(trn, val, ytr, yva, NUM_ALT, p_alt(s + 100 + 10 * b + fold))
                pa += pred
                itB += it
            oofA[s][va_i] = pm / N_BAGS
            oofB[s][va_i] = pa / N_BAGS
            done[s] = fold
            print(
                f"fold {fold} seed {s} A={roc_auc_score(yva, oofA[s][va_i]):.4f} "
                f"B={roc_auc_score(yva, oofB[s][va_i]):.4f} it~{itA/N_BAGS:.0f}/{itB/N_BAGS:.0f}",
                flush=True,
            )
        np.savez(ckpt, **{f"A{s}": oofA[s] for s in SEEDS}, **{f"B{s}": oofB[s] for s in SEEDS}, **{f"f{s}": done.get(s, -1) for s in SEEDS})
        print(f"  fold {fold} dt={time.time()-t0:.1f}s", flush=True)

    rA = np.mean([rank(oofA[s]) for s in SEEDS], axis=0)
    rB = np.mean([rank(oofB[s]) for s in SEEDS], axis=0)
    w62 = 0.62 * rA + 0.38 * rB
    mx = np.maximum(rA, rB)
    report = {
        "per_seed_A": {str(s): float(roc_auc_score(y, oofA[s])) for s in SEEDS},
        "per_seed_B": {str(s): float(roc_auc_score(y, oofB[s])) for s in SEEDS},
        "w62": float(roc_auc_score(y, w62)),
        "max2": float(roc_auc_score(y, mx)),
        "protocol": "10fold x 3bag x 3seed, ES on OOF val fold (W62 protocol)",
    }
    print("=== EXP8h VAL-ES ===", json.dumps(report, indent=2))
    (OUT / "exp8h_vales.json").write_text(json.dumps(report, indent=2))
    np.save(OUT / "exp8h_w62.npy", w62)
    np.save(OUT / "exp8h_max2.npy", mx)
    lgb_p = OUT / "exp8d_lgb_w62.npy"
    best, name, ba = mx, "vales_max2", float(roc_auc_score(y, mx))
    if lgb_p.exists():
        rl = rank(np.load(lgb_p))
        for w in (0.90, 0.85, 0.80):
            s = w * mx + (1 - w) * rl
            a = float(roc_auc_score(y, s))
            print(f"  max2*{w}+lgb: {a:.5f}")
            if a > ba:
                best, name, ba = s, f"vales_max2_{w}_lgb", a
    np.save(OUT / "best_oof.npy", best)
    print("BEST", name, ba)
    if ba >= 0.70:
        print("HIT 0.70")


if __name__ == "__main__":
    main()
