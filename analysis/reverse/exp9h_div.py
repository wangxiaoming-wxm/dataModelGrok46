#!/usr/bin/env python3
"""Diversity arms: Lossguide Plain, extra-x nums, lower lr. 5-fold val-ES.

Blend with 8-seed Ordered OOF if Spearman < 0.98 and strength is close.
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

from cb_features import CATS, NUM_MAIN, fold_features

OUT = Path("/workspace/analysis/reverse")
THREADS = 1
EXTRA_X = ["x0", "x2", "x3", "x4", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x15", "x16", "livability"]


def auc(y, s):
    return float(roc_auc_score(y, s))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit(trn, val, ytr, yva, nums, cats, params):
    cats = list(cats)
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


def run(train, y, name, params, nums, cats, n_folds=5):
    n = len(y)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        for df in (trn, val):
            df["t3_letter"] = df["t3"].astype(str).str.extract(r"([A-Za-z]+)", expand=False).fillna("NA")
            df["safe_2160"] = ((df["days"] >= 2110) & (df["days"] < 2210)).astype(np.int8)
        p = dict(params)
        p["random_seed"] = 2026 + fold
        pred, it = fit(trn, val, y[tr_i], y[va_i], nums, cats, p)
        oof[va_i] = pred
        print(f"  {name} fold {fold} auc={auc(y[va_i], pred):.4f} it={it}", flush=True)
    a = auc(y, oof)
    print(f"=== {name} OOF {a:.5f} dt={time.time()-t0:.1f}s ===", flush=True)
    np.save(OUT / f"exp9h_{name}.npy", oof)
    return oof, a


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    h = np.load(OUT / "exp8h_max2.npy")
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    best = np.load(OUT / "best_oof.npy")
    report = []
    base_p = dict(
        loss_function="RMSE",
        iterations=800,
        learning_rate=0.03,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=THREADS,
    )
    cfgs = [
        (
            "lossguide",
            dict(base_p, boosting_type="Plain", grow_policy="Lossguide", max_leaves=31, depth=6, l2_leaf_reg=6, rsm=0.3),
            NUM_MAIN,
            CATS,
        ),
        (
            "plain_d8",
            dict(base_p, boosting_type="Plain", depth=8, l2_leaf_reg=8, rsm=0.3),
            NUM_MAIN,
            CATS,
        ),
        (
            "extra_x",
            dict(base_p, boosting_type="Ordered", depth=5, l2_leaf_reg=10, rsm=1.0),
            NUM_MAIN + EXTRA_X + ["safe_2160"],
            list(CATS) + ["t3_letter"],
        ),
        (
            "lr02",
            dict(base_p, boosting_type="Ordered", depth=5, l2_leaf_reg=10, rsm=1.0, learning_rate=0.02, iterations=1200),
            NUM_MAIN,
            CATS,
        ),
    ]
    for name, params, nums, cats in cfgs:
        try:
            oof, a = run(train, y, name, params, nums, cats)
        except Exception as e:
            print("FAIL", name, e, flush=True)
            report.append({"name": name, "error": str(e)})
            continue
        sp = pd.Series(rank(oof)).corr(pd.Series(rank(h)), method="spearman")
        blends = {
            "oof": a,
            "spearman_h": float(sp),
            "max(h,new)": auc(y, np.maximum(rank(h), rank(oof))),
            "0.7h+0.3new": auc(y, 0.7 * rank(h) + 0.3 * rank(oof)),
            "0.85h+0.15new": auc(y, 0.85 * rank(h) + 0.15 * rank(oof)),
            "0.75h+0.15new+0.1lgb": auc(y, 0.75 * rank(h) + 0.15 * rank(oof) + 0.10 * rank(lgb)),
        }
        print("  ", json.dumps(blends), flush=True)
        rec = {"name": name, **blends}
        report.append(rec)
        (OUT / "exp9h_div.json").write_text(json.dumps(report, indent=2))
        ba = max(blends[k] for k in blends if k not in ("oof", "spearman_h"))
        winner = None
        for k, v in blends.items():
            if k in ("oof", "spearman_h"):
                continue
            if v == ba:
                if k == "max(h,new)":
                    winner = np.maximum(rank(h), rank(oof))
                elif k == "0.7h+0.3new":
                    winner = 0.7 * rank(h) + 0.3 * rank(oof)
                elif k == "0.85h+0.15new":
                    winner = 0.85 * rank(h) + 0.15 * rank(oof)
                else:
                    winner = 0.75 * rank(h) + 0.15 * rank(oof) + 0.10 * rank(lgb)
                break
        if ba > auc(y, best) + 1e-5:
            np.save(OUT / "best_oof.npy", winner)
            best = winner
            print("UPDATED best", ba, flush=True)
        if ba >= 0.70:
            print("HIT 0.70", name, ba, flush=True)
    print("DONE", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
