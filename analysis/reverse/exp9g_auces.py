#!/usr/bin/env python3
"""AUC early-stopping (still RMSE loss) + per-source CatBoost. 5-fold val-ES.

If AUC-ES 5-fold dual-arm >= 0.690, scale to 10-fold x 3-bag x 3-seed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cb_features import CATS, NUM_ALT, NUM_MAIN, fold_features

OUT = Path("/workspace/analysis/reverse")
THREADS = 1
LARGE = ["CAR_1|ENG_591", "CAR_0|ENG_709", "CAR_2|ENG_262", "CAR_5|ENG_062"]


def auc(y, s):
    return float(roc_auc_score(y, s))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit(trn, val, ytr, yva, nums, cats, params):
    cats = [c for c in cats if c in trn.columns]
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
    return model.predict(Pool(Xva, cat_features=cats)), int(model.best_iteration_ or 0)


def p_main(seed, eval_auc=False, iters=800):
    p = dict(
        loss_function="RMSE",
        iterations=iters,
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
    if eval_auc:
        p["eval_metric"] = "AUC"
    return p


def p_alt(seed, eval_auc=False, iters=800):
    p = dict(
        loss_function="RMSE",
        iterations=iters,
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
    if eval_auc:
        p["eval_metric"] = "AUC"
    return p


def dual(train, y, name, eval_auc, n_folds=5, n_bag=1, seeds=(2026,), iters=800):
    n = len(y)
    accA = {s: np.zeros(n) for s in seeds}
    accB = {s: np.zeros(n) for s in seeds}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    ckpt = OUT / f"exp9g_{name}_ckpt.npz"
    done = -1
    if ckpt.exists() and n_folds >= 10:
        z = np.load(ckpt)
        for s in seeds:
            accA[s] = z[f"A{s}"]
            accB[s] = z[f"B{s}"]
        done = int(z["done"])
        print("resume", name, done, flush=True)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        if fold <= done:
            continue
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        for s in seeds:
            pm = pa = None
            itA = itB = 0
            sm = np.zeros(len(va_i))
            sa = np.zeros(len(va_i))
            for b in range(n_bag):
                pred, it = fit(trn, val, ytr, yva, NUM_MAIN, CATS, p_main(s + 10 * b + fold, eval_auc, iters))
                sm += pred
                itA += it
                pred, it = fit(trn, val, ytr, yva, NUM_ALT, CATS, p_alt(s + 100 + 10 * b + fold, eval_auc, iters))
                sa += pred
                itB += it
            accA[s][va_i] = sm / n_bag
            accB[s][va_i] = sa / n_bag
        print(
            f"{name} fold {fold} A={auc(yva, accA[seeds[0]][va_i]):.4f} B={auc(yva, accB[seeds[0]][va_i]):.4f} it~{itA/n_bag:.0f}/{itB/n_bag:.0f}",
            flush=True,
        )
        np.savez(ckpt, done=fold, **{f"A{s}": accA[s] for s in seeds}, **{f"B{s}": accB[s] for s in seeds})
    rA = np.mean([rank(accA[s]) for s in seeds], axis=0)
    rB = np.mean([rank(accB[s]) for s in seeds], axis=0)
    mx = np.maximum(rA, rB)
    w62 = 0.62 * rA + 0.38 * rB
    rec = dict(name=name, max2=auc(y, mx), w62=auc(y, w62), A=auc(y, rA), B=auc(y, rB), dt=time.time() - t0, eval_auc=eval_auc, n_folds=n_folds, n_bag=n_bag, n_seed=len(seeds))
    print(f"=== {name} ===", json.dumps(rec, indent=2), flush=True)
    np.save(OUT / f"exp9g_{name}_max2.npy", mx)
    np.save(OUT / f"exp9g_{name}_w62.npy", w62)
    return rec, mx, w62


def persource(train, y, n_folds=5):
    n = len(y)
    oof_g = np.zeros(n)
    oof_ps = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        pred_g, it = fit(trn, val, ytr, yva, NUM_MAIN, CATS, p_main(4000 + fold, False, 600))
        oof_g[va_i] = pred_g
        pred_ps = pred_g.copy()
        src_va = val["source"].astype(str).to_numpy()
        src_tr = trn["source"].astype(str).to_numpy()
        for s in LARGE:
            mtr = src_tr == s
            mva = src_va == s
            if mtr.sum() < 80 or mva.sum() < 20:
                continue
            cats_s = [c for c in CATS if not c.startswith("src_") and c != "source"]
            p, _ = fit(trn.loc[mtr], val.loc[mva], ytr[mtr], yva[mva], NUM_MAIN, cats_s, p_main(5000 + fold, False, 400))
            pred_ps[mva] = 0.5 * pred_g[mva] + 0.5 * p
        oof_ps[va_i] = pred_ps
        print(f"ps fold {fold} g={auc(yva, pred_g):.4f} ps={auc(yva, pred_ps):.4f}", flush=True)
    rec = dict(global_=auc(y, oof_g), persource=auc(y, oof_ps), dt=time.time() - t0)
    print("=== persource ===", rec, flush=True)
    np.save(OUT / "exp9g_ps.npy", oof_ps)
    return rec, oof_ps


def maybe_update(y, pred, tag):
    cur = np.load(OUT / "best_oof.npy")
    a = auc(y, pred)
    ca = auc(y, cur)
    print(f"candidate {tag} {a:.5f} vs best {ca:.5f}")
    if a > ca:
        np.save(OUT / "best_oof.npy", pred)
        print("UPDATED best_oof", tag, a)
    if a >= 0.70:
        print("HIT 0.70", tag, a)
    return a


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    report = {}
    rec, mx, w62 = dual(train, y, "auces5", eval_auc=True, n_folds=5, n_bag=1, seeds=(2026,), iters=800)
    report["auces5"] = rec
    h = np.load(OUT / "exp8h_max2.npy")
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    report["auces5_blends"] = {
        "max(h,mx)": maybe_update(y, np.maximum(rank(h), rank(mx)), "max(h,auces5)"),
        "0.8h+0.2": auc(y, 0.8 * rank(h) + 0.2 * rank(mx)),
        "0.8mx+0.2lgb": auc(y, 0.8 * rank(mx) + 0.2 * rank(lgb)),
    }
    rec_ps, oof_ps = persource(train, y)
    report["persource"] = rec_ps
    report["ps_blends"] = {
        "max(h,ps)": maybe_update(y, np.maximum(rank(h), rank(oof_ps)), "max(h,ps)"),
        "0.8h+0.2ps": auc(y, 0.8 * rank(h) + 0.2 * rank(oof_ps)),
    }
    (OUT / "exp9g_auces.json").write_text(json.dumps(report, indent=2, default=str))

    if rec["max2"] >= 0.689:
        print("SCALING AUC-ES to 10-fold 3bag 3seed", flush=True)
        rec10, mx10, w10 = dual(
            train, y, "auces10", eval_auc=True, n_folds=10, n_bag=3, seeds=(2026, 2036, 2046), iters=800
        )
        report["auces10"] = rec10
        maybe_update(y, mx10, "auces10_max2")
        maybe_update(y, 0.8 * rank(mx10) + 0.2 * rank(lgb), "auces10_lgb")
        maybe_update(y, np.maximum(rank(h), rank(mx10)), "max(h,auces10)")
        (OUT / "exp9g_auces.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
