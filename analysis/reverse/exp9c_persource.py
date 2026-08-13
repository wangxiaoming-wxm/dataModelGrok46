#!/usr/bin/env python3
"""Per-source CatBoost (large cars own model, small cars share global) + Langevin probe.

5-fold val-ES Ordered RMSE. Encoders fit on train fold only.
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
LARGE = ["CAR_1|ENG_591", "CAR_0|ENG_709", "CAR_2|ENG_262", "CAR_5|ENG_062"]
THREADS = 1
ITERS = 500


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def auc(y, s):
    return float(roc_auc_score(y, s))


def params(seed, extra=None):
    p = dict(
        loss_function="RMSE",
        iterations=ITERS,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        random_seed=seed,
        od_type="Iter",
        od_wait=50,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Ordered",
        rsm=1.0,
    )
    if extra:
        p.update(extra)
    return p


def fit_predict(trn, val, ytr, yva, cats, extra=None, seed=0):
    cols = [c for c in NUM_MAIN if c in trn.columns] + cats
    # drop cats missing in this slice
    cats_ok = [c for c in cats if c in trn.columns]
    cols = [c for c in NUM_MAIN if c in trn.columns] + cats_ok
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in cats_ok:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    model = CatBoostRegressor(**params(seed, extra))
    model.fit(
        Pool(Xtr, ytr, cat_features=cats_ok),
        eval_set=Pool(Xva, yva, cat_features=cats_ok),
        use_best_model=True,
        verbose=False,
    )
    return model.predict(Pool(Xva, cat_features=cats_ok)), int(model.best_iteration_ or 0)


def persource(train, y, n_folds=5):
    n = len(y)
    oof_g = np.zeros(n)
    oof_ps = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    slice_aucs = {s: [] for s in LARGE}
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr, yva = y[tr_i], y[va_i]
        pred_g, it = fit_predict(trn, val, ytr, yva, CATS, seed=1000 + fold)
        oof_g[va_i] = pred_g
        pred_ps = pred_g.copy()
        src_va = val["source"].astype(str).to_numpy()
        src_tr = trn["source"].astype(str).to_numpy()
        for s in LARGE:
            mtr = src_tr == s
            mva = src_va == s
            if mtr.sum() < 80 or mva.sum() < 20:
                continue
            # fewer cats in slice: source is constant
            cats_s = [c for c in CATS if c not in ("source", "src_reg", "src_age", "src_cq", "src_dq", "src_dq5", "src_ratioq", "src_rateq", "src_grades")]
            p, _ = fit_predict(trn.loc[mtr], val.loc[mva], ytr[mtr], yva[mva], cats_s, seed=2000 + fold)
            pred_ps[mva] = 0.55 * pred_g[mva] + 0.45 * p
            slice_aucs[s].append(auc(yva[mva], pred_ps[mva]))
        oof_ps[va_i] = pred_ps
        print(f"fold {fold} glob={auc(yva, pred_g):.4f} ps={auc(yva, pred_ps):.4f} it={it}", flush=True)
    rec = {
        "global": auc(y, oof_g),
        "persource": auc(y, oof_ps),
        "dt": time.time() - t0,
        "slice_mean": {s: float(np.mean(v)) if v else None for s, v in slice_aucs.items()},
    }
    print("=== persource", json.dumps(rec, indent=2))
    np.save(OUT / "exp9c_global.npy", oof_g)
    np.save(OUT / "exp9c_persource.npy", oof_ps)
    return rec, oof_ps


def langevin_probe(train, y, n_folds=5):
    n = len(y)
    extras = {
        "base": None,
        "langevin": dict(langevin=True, diffusion_temperature=30000),
        "posterior": dict(posterior_sampling=True),
        "bag_bayes": dict(bootstrap_type="Bayesian", bagging_temperature=1.0),
        "bag_bernoulli": dict(bootstrap_type="Bernoulli", subsample=0.8),
    }
    out = {}
    for name, extra in extras.items():
        oof = np.zeros(n)
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
        t0 = time.time()
        ok = True
        for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
            trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
            try:
                pred, it = fit_predict(trn, val, y[tr_i], y[va_i], CATS, extra=extra, seed=3000 + fold)
            except Exception as e:
                print(f"  {name} failed: {e}")
                ok = False
                break
            oof[va_i] = pred
            print(f"  {name} fold {fold} auc={auc(y[va_i], pred):.4f} it={it}", flush=True)
        if not ok:
            out[name] = {"oof": None, "error": True}
            continue
        a = auc(y, oof)
        out[name] = {"oof": a, "dt": time.time() - t0}
        np.save(OUT / f"exp9d_{name}.npy", oof)
        print(f"=== {name} OOF {a:.5f} ===")
    return out


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    rec_ps, oof_ps = persource(train, y)
    rec_lg = langevin_probe(train, y)
    best = np.load(OUT / "best_oof.npy")
    mx = np.load(OUT / "exp8h_max2.npy")
    lgb = np.load(OUT / "exp8d_lgb_w62.npy")
    blends = {
        "ps": auc(y, oof_ps),
        "max(h,ps)": auc(y, np.maximum(rank(mx), rank(oof_ps))),
        "0.8h+0.2ps": auc(y, 0.8 * rank(mx) + 0.2 * rank(oof_ps)),
        "0.75h+0.15ps+0.1lgb": auc(y, 0.75 * rank(mx) + 0.15 * rank(oof_ps) + 0.10 * rank(lgb)),
    }
    print("blends", blends)
    winner = oof_ps
    wa = blends["ps"]
    for name, a in blends.items():
        if a > wa:
            wa, winner = a, {
                "max(h,ps)": np.maximum(rank(mx), rank(oof_ps)),
                "0.8h+0.2ps": 0.8 * rank(mx) + 0.2 * rank(oof_ps),
                "0.75h+0.15ps+0.1lgb": 0.75 * rank(mx) + 0.15 * rank(oof_ps) + 0.10 * rank(lgb),
            }.get(name, oof_ps)
    if wa > auc(y, best):
        np.save(OUT / "best_oof.npy", winner)
        print("UPDATED best", wa)
    report = {"persource": rec_ps, "tricks": rec_lg, "blends": blends, "best": wa}
    (OUT / "exp9c_persource.json").write_text(json.dumps(report, indent=2, default=str))
    if wa >= 0.70:
        print("HIT 0.70")


if __name__ == "__main__":
    main()
