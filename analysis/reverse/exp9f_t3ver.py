#!/usr/bin/env python3
"""Add t3/version as CatBoost cats; Langevin; 5-fold Ordered val-ES ranker.

Winner (if ≥ current 5-fold 0.687) is scaled by the caller. Encoders fold-local.
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
ITERS = 600


def auc(y, s):
    return float(roc_auc_score(y, s))


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def fit(trn, val, ytr, yva, cats, extra=None, seed=0):
    cats = list(cats)
    cols = [c for c in NUM_MAIN if c in trn.columns] + cats
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    p = dict(
        loss_function="RMSE",
        iterations=ITERS,
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
    if extra:
        p.update(extra)
    model = CatBoostRegressor(**p)
    model.fit(
        Pool(Xtr, ytr, cat_features=cats),
        eval_set=Pool(Xva, yva, cat_features=cats),
        use_best_model=True,
        verbose=False,
    )
    return model.predict(Pool(Xva, cat_features=cats)), int(model.best_iteration_ or 0)


def probe(train, y, name, extra_cats, extra_params=None, n_folds=5):
    n = len(y)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    cats_used = None
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        for df in (trn, val):
            df["t3"] = df["t3"].astype(str)
            df["version"] = df["version"].astype(str)
            df["t3_letter"] = df["t3"].str.extract(r"([A-Za-z]+)", expand=False).fillna("NA")
            df["src_t3"] = df["source"].astype(str) + "|" + df["t3"].astype(str)
        cats = list(CATS) + [c for c in extra_cats if c not in CATS]
        cats_used = cats
        pred, it = fit(trn, val, y[tr_i], y[va_i], cats, extra=extra_params, seed=2026 + fold)
        oof[va_i] = pred
        print(f"  {name} fold {fold} auc={auc(y[va_i], pred):.4f} it={it} nc={len(cats)}", flush=True)
    a = auc(y, oof)
    rec = {"name": name, "oof": a, "n_cats": len(cats_used), "extra": extra_cats, "dt": time.time() - t0}
    print(f"=== {name} OOF {a:.5f} dt={rec['dt']:.1f}s ===", flush=True)
    np.save(OUT / f"exp9f_{name}.npy", oof)
    return oof, rec


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    report = []
    grid = [
        ("base", []),
        ("t3", ["t3"]),
        ("ver", ["version"]),
        ("t3ver", ["t3", "version"]),
        ("t3let", ["t3_letter"]),
        ("lang", [], dict(langevin=True, diffusion_temperature=30000)),
    ]
    for item in grid:
        name, extra = item[0], item[1]
        params = item[2] if len(item) > 2 else None
        try:
            oof, rec = probe(train, y, name, extra, extra_params=params)
        except Exception as e:
            print(f"FAIL {name}: {e}")
            report.append({"name": name, "error": str(e)})
            continue
        best = np.load(OUT / "best_oof.npy")
        h = np.load(OUT / "exp8h_max2.npy")
        rec["blend_max"] = auc(y, np.maximum(rank(h), rank(oof)))
        rec["blend_08"] = auc(y, 0.85 * rank(h) + 0.15 * rank(oof))
        print("  blends", rec["blend_max"], rec["blend_08"], flush=True)
        report.append(rec)
        (OUT / "exp9f_t3ver.json").write_text(json.dumps(report, indent=2))
        if rec["oof"] >= 0.690:
            print("PROMISING 5-fold", name, rec["oof"])
        if rec["blend_max"] > auc(y, best) + 0.0002:
            np.save(OUT / "best_oof.npy", np.maximum(rank(h), rank(oof)))
            print("UPDATED best via max blend", rec["blend_max"])
    print("DONE", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
