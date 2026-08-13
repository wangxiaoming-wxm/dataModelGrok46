#!/usr/bin/env python3
"""5-fold val-ES dual-arm probe of W62-faithful features (v6 ratios, CATS_W62)
vs reverse NUM_MAIN/CATS. Then optionally scale winner to 10-fold x 3-bag.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w62f = load_mod("cb_w62feat", "/workspace/analysis/cb_features.py")
revf = load_mod("cb_revfeat", "/workspace/analysis/reverse/cb_features.py")
CATS_W62 = w62f.CATS_W62
NUM_ALT_W62 = w62f.NUM_ALT_W62
NUM_MAIN_W62 = w62f.NUM_MAIN_W62
fold_w62 = w62f.fold_features
CATS_REV = revf.CATS
NUM_ALT_REV = revf.NUM_ALT
NUM_MAIN_REV = revf.NUM_MAIN
fold_rev = revf.fold_features

OUT = Path("/workspace/analysis/reverse")
THREADS = 1


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def auc(y, s):
    return float(roc_auc_score(y, s))


def fit(trn, val, ytr, yva, nums, cats, params):
    cols = [c for c in nums if c in trn.columns] + list(cats)
    miss = [c for c in cols if c not in trn.columns]
    if miss:
        raise KeyError(miss)
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    # extra window
    if "safe_2160" not in Xtr.columns and "days" in trn.columns:
        pass
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(Xtr, ytr, cat_features=list(cats)),
        eval_set=Pool(Xva, yva, cat_features=list(cats)),
        use_best_model=True,
        verbose=False,
    )
    return model.predict(Pool(Xva, cat_features=list(cats))), int(model.best_iteration_ or 0)


def p_main(seed, iters=600):
    return dict(
        loss_function="RMSE",
        iterations=iters,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        random_seed=seed,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Ordered",
        rsm=1.0,
    )


def p_alt(seed, iters=600):
    return dict(
        loss_function="RMSE",
        iterations=iters,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6,
        random_seed=seed,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Plain",
        rsm=0.3,
    )


def add_extra(df):
    d = df.copy()
    d["safe_2160"] = ((d["days"] >= 2110) & (d["days"] < 2210)).astype(np.int8)
    return d


def run_cfg(train, y, name, folder, nums_m, nums_a, cats, n_folds=5, n_bag=1, seeds=(2026,), iters=600):
    n = len(y)
    oofA = np.zeros(n)
    oofB = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    n_seed = len(seeds)
    accA = {s: np.zeros(n) for s in seeds}
    accB = {s: np.zeros(n) for s in seeds}
    ckpt = OUT / f"exp9e_{name}_ckpt.npz"
    done_fold = -1
    if ckpt.exists() and n_folds >= 10:
        z = np.load(ckpt, allow_pickle=True)
        for s in seeds:
            accA[s] = z[f"A{s}"]
            accB[s] = z[f"B{s}"]
        done_fold = int(z["done_fold"])
        print(f"resume {name} from fold {done_fold}", flush=True)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        if fold <= done_fold:
            continue
        raw_tr, raw_va = train.iloc[tr_i], train.iloc[va_i]
        if folder == "w62":
            trn, val = fold_w62(raw_tr, raw_va)
        else:
            trn, val = fold_rev(raw_tr, raw_va)
        trn, val = add_extra(trn), add_extra(val)
        nums_m2 = list(nums_m) + (["safe_2160"] if "safe_2160" not in nums_m else [])
        nums_a2 = list(nums_a) + (["safe_2160"] if "safe_2160" not in nums_a else [])
        ytr, yva = y[tr_i], y[va_i]
        for s in seeds:
            pm = np.zeros(len(va_i))
            pa = np.zeros(len(va_i))
            itA = itB = 0
            for b in range(n_bag):
                pred, it = fit(trn, val, ytr, yva, nums_m2, cats, p_main(s + 10 * b + fold, iters))
                pm += pred
                itA += it
                pred, it = fit(trn, val, ytr, yva, nums_a2, cats, p_alt(s + 100 + 10 * b + fold, iters))
                pa += pred
                itB += it
            accA[s][va_i] = pm / n_bag
            accB[s][va_i] = pa / n_bag
        print(
            f"{name} fold {fold} A={auc(yva, accA[seeds[0]][va_i]):.4f} B={auc(yva, accB[seeds[0]][va_i]):.4f} it~{itA/n_bag:.0f}/{itB/n_bag:.0f}",
            flush=True,
        )
        np.savez(ckpt, done_fold=fold, **{f"A{s}": accA[s] for s in seeds}, **{f"B{s}": accB[s] for s in seeds})
    rA = np.mean([rank(accA[s]) for s in seeds], axis=0)
    rB = np.mean([rank(accB[s]) for s in seeds], axis=0)
    w62 = 0.62 * rA + 0.38 * rB
    mx = np.maximum(rA, rB)
    rec = {
        "name": name,
        "A": auc(y, rA) if n_seed > 1 else auc(y, accA[seeds[0]]),
        "B": auc(y, rB) if n_seed > 1 else auc(y, accB[seeds[0]]),
        "w62": auc(y, w62),
        "max2": auc(y, mx),
        "n_folds": n_folds,
        "n_bag": n_bag,
        "n_seed": n_seed,
        "n_cats": len(cats),
        "dt": time.time() - t0,
    }
    print(f"=== {name} ===", json.dumps(rec, indent=2), flush=True)
    np.save(OUT / f"exp9e_{name}_max2.npy", mx)
    np.save(OUT / f"exp9e_{name}_w62.npy", w62)
    return rec, mx, w62


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    report = []
    rec, _, _ = run_cfg(
        train, y, "w62_5", "w62", NUM_MAIN_W62, NUM_ALT_W62, CATS_W62, n_folds=5, n_bag=1, seeds=(2026,)
    )
    report.append(rec)
    (OUT / "exp9e_probe.json").write_text(json.dumps(report, indent=2))

    rec, mx, w62s = run_cfg(
        train,
        y,
        "w62_10",
        "w62",
        NUM_MAIN_W62,
        NUM_ALT_W62,
        CATS_W62,
        n_folds=10,
        n_bag=3,
        seeds=(2026, 2036, 2046),
        iters=800,
    )
    report.append(rec)
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    h = np.load(OUT / "exp8h_max2.npy")
    blends = {
        "w62_10_max2": rec["max2"],
        "w62_10_w62": rec["w62"],
        "max(h,new)": auc(y, np.maximum(rank(h), rank(mx))),
        "0.8h+0.2new": auc(y, 0.8 * rank(h) + 0.2 * rank(mx)),
        "0.8new+0.2lgb": auc(y, 0.8 * rank(mx) + 0.2 * rank(lgb)),
        "0.7h+0.3new": auc(y, 0.7 * rank(h) + 0.3 * rank(mx)),
    }
    print("SCALE BLENDS", blends, flush=True)
    rec["blends"] = blends
    best, ba, name = mx, rec["max2"], "w62_10_max2"
    mapping = {
        "max(h,new)": np.maximum(rank(h), rank(mx)),
        "0.8new+0.2lgb": 0.8 * rank(mx) + 0.2 * rank(lgb),
        "0.8h+0.2new": 0.8 * rank(h) + 0.2 * rank(mx),
        "0.7h+0.3new": 0.7 * rank(h) + 0.3 * rank(mx),
        "w62_10_w62": w62s,
        "w62_10_max2": mx,
    }
    for k, v in blends.items():
        if v > ba:
            ba, name = v, k
            best = mapping[k]
    cur = auc(y, np.load(OUT / "best_oof.npy"))
    if ba > cur:
        np.save(OUT / "best_oof.npy", best)
        print("UPDATED best_oof", name, ba, "from", cur, flush=True)
    else:
        print("no improvement", name, ba, "vs", cur, flush=True)
    if ba >= 0.70:
        print("HIT 0.70", name, ba, flush=True)
    (OUT / "exp9e_probe.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
