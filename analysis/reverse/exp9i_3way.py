#!/usr/bin/env python3
"""Explicit source|region|age 3-way (named in W62 notes) + low-card 3-ways.

5-fold Ordered val-ES. If >= 0.690, scale 10-fold x 3-bag x 3-seed.
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


def auc(y, s):
    return float(roc_auc_score(y, s))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def add3(df):
    d = df.copy()
    d["t3_letter"] = d["t3"].astype(str).str.extract(r"([A-Za-z]+)", expand=False).fillna("NA")
    d["src_reg_age"] = d["source"].astype(str) + "|" + d["region"].astype(str) + "|" + d["age_range"].astype(str)
    d["src_gr_age"] = d["source"].astype(str) + "|" + d["grades"].astype(str) + "|" + d["age_range"].astype(str)
    d["reg_gr_age"] = d["region"].astype(str) + "|" + d["grades"].astype(str) + "|" + d["age_range"].astype(str)
    d["src_reg_gr"] = d["source"].astype(str) + "|" + d["region"].astype(str) + "|" + d["grades"].astype(str)
    d["safe_2160"] = ((d["days"] >= 2110) & (d["days"] < 2210)).astype(np.int8)
    return d


def fit(trn, val, ytr, yva, nums, cats, params):
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


def probe(train, y, name, extra_cats, dual=False, n_folds=5, n_bag=1, seeds=(2026,)):
    n = len(y)
    accA = {s: np.zeros(n) for s in seeds}
    accB = {s: np.zeros(n) for s in seeds}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    ckpt = OUT / f"exp9i_{name}_ckpt.npz"
    done = -1
    if ckpt.exists() and n_folds >= 10:
        z = np.load(ckpt)
        for s in seeds:
            accA[s] = z[f"A{s}"]
            accB[s] = z[f"B{s}"]
        done = int(z["done"])
        print("resume", done, flush=True)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        if fold <= done:
            continue
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        trn, val = add3(trn), add3(val)
        cats = list(CATS) + extra_cats
        nums_m = list(NUM_MAIN) + ["safe_2160"]
        nums_a = list(NUM_ALT) + ["safe_2160"]
        ytr, yva = y[tr_i], y[va_i]
        for s in seeds:
            sm = np.zeros(len(va_i))
            sa = np.zeros(len(va_i))
            itA = itB = 0
            for b in range(n_bag):
                pred, it = fit(trn, val, ytr, yva, nums_m, cats, p_main(s + 10 * b + fold))
                sm += pred
                itA += it
                if dual:
                    pred, it = fit(trn, val, ytr, yva, nums_a, cats, p_alt(s + 100 + 10 * b + fold))
                    sa += pred
                    itB += it
            accA[s][va_i] = sm / n_bag
            if dual:
                accB[s][va_i] = sa / n_bag
        print(
            f"{name} fold {fold} A={auc(yva, accA[seeds[0]][va_i]):.4f}"
            + (f" B={auc(yva, accB[seeds[0]][va_i]):.4f}" if dual else "")
            + f" it~{itA/n_bag:.0f}",
            flush=True,
        )
        np.savez(ckpt, done=fold, **{f"A{s}": accA[s] for s in seeds}, **{f"B{s}": accB[s] for s in seeds})
    rA = np.mean([rank(accA[s]) for s in seeds], axis=0)
    rec = {"name": name, "A": auc(y, rA), "extra": extra_cats, "dt": time.time() - t0, "n_folds": n_folds, "n_bag": n_bag, "n_seed": len(seeds)}
    mx = rA
    if dual:
        rB = np.mean([rank(accB[s]) for s in seeds], axis=0)
        rec["B"] = auc(y, rB)
        rec["max2"] = auc(y, np.maximum(rA, rB))
        rec["w62"] = auc(y, 0.62 * rA + 0.38 * rB)
        mx = np.maximum(rA, rB)
    else:
        rec["max2"] = rec["A"]
    print(f"=== {name} ===", json.dumps(rec, indent=2), flush=True)
    np.save(OUT / f"exp9i_{name}.npy", mx)
    return rec, mx


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    h = np.load(OUT / "exp8h_max2.npy")
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    report = []
    # 5-fold Ordered-only probes
    for name, extra in [
        ("sra", ["src_reg_age"]),
        ("sra_low", ["src_reg_age", "src_gr_age", "reg_gr_age", "src_reg_gr", "t3_letter"]),
    ]:
        rec, mx = probe(train, y, name, extra, dual=False, n_folds=5)
        rec["max(h,new)"] = auc(y, np.maximum(rank(h), rank(mx)))
        rec["0.85h+0.15"] = auc(y, 0.85 * rank(h) + 0.15 * rank(mx))
        print("blends", rec["max(h,new)"], rec["0.85h+0.15"], flush=True)
        report.append(rec)
        (OUT / "exp9i_3way.json").write_text(json.dumps(report, indent=2))

    winner = max(report, key=lambda r: r["A"])
    print("winner 5-fold", winner["name"], winner["A"])
    if winner["A"] >= 0.689:
        rec, mx = probe(
            train,
            y,
            "sra10",
            winner["extra"],
            dual=True,
            n_folds=10,
            n_bag=3,
            seeds=(2026, 2036, 2046),
        )
        report.append(rec)
        for tag, pred in {
            "sra10_max2": mx,
            "sra10_lgb": 0.8 * rank(mx) + 0.2 * rank(lgb),
            "max(h,sra10)": np.maximum(rank(h), rank(mx)),
            "0.8h+0.2sra": 0.8 * rank(h) + 0.2 * rank(mx),
        }.items():
            a = auc(y, pred)
            print(tag, a, flush=True)
            rec[tag] = a
            if a > auc(y, np.load(OUT / "best_oof.npy")):
                np.save(OUT / "best_oof.npy", pred)
                print("UPDATED", tag, a)
            if a >= 0.70:
                print("HIT 0.70", tag, a)
        (OUT / "exp9i_3way.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
