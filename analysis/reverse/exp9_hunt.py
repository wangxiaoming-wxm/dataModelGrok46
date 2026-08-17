#!/usr/bin/env python3
"""Hunt 0.70: leftover signal after CB, 80-cross TE Ridge, CatBoost cat-set grid.

Protocol: StratifiedKFold, encoders fit on train fold only.
CatBoost probes use val-fold ES (W62 protocol) but 5-fold x 1-bag x 1-seed Ordered arm
so we can rank representations on 1 CPU before scaling a winner.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from cb_features import CATS as CATS_CUR, NUM_MAIN, fold_features
from feat80 import fold_features_80, loo_te, te_many

OUT = Path("/workspace/analysis/reverse")
THREADS = 1
ITERS = 600


def rank(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def auc(y, s):
    return float(roc_auc_score(y, s))


def residual_scan(train: pd.DataFrame, y: np.ndarray) -> dict:
    pred = np.load(OUT / "exp8h_max2.npy")
    # rank-score -> residual in probability space via isotonic would be ideal;
    # use linear calibration of rank to [0,1] mean.
    p = pred.astype(float)
    p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    # rescale to match base rate roughly
    p = p * (y.mean() / p.mean())
    resid = y.astype(float) - p
    rows = []
    num_cols = [c for c in train.columns if c not in ("id", "label") and pd.api.types.is_numeric_dtype(train[c])]
    for c in num_cols:
        x = train[c].to_numpy(float)
        m = np.isfinite(x)
        if m.sum() < 100:
            continue
        rho, _ = spearmanr(resid[m], x[m])
        rows.append((abs(float(rho)), float(rho), c, "num"))
    for c in ["source", "region", "age_range", "grades", "month", "version", "code", "t3"]:
        means = pd.Series(resid, index=train.index).groupby(train[c]).mean()
        gm = train[c].map(means).to_numpy(float)
        rho, _ = spearmanr(resid, gm)
        rows.append((abs(float(rho)), float(rho), c, "cat_gmean"))
    rows.sort(reverse=True)
    top = [{"abs_rho": a, "rho": r, "col": c, "kind": k} for a, r, c, k in rows[:25]]
    print("=== residual after exp8h_max2 ===")
    for t in top[:15]:
        print(f"  {t['abs_rho']:.4f}  {t['col']:16s}  {t['kind']}")
    return {"top": top}


def te_ridge(train: pd.DataFrame, y: np.ndarray, which: str, n_folds: int = 10) -> tuple[np.ndarray, dict]:
    n = len(y)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val, cats = fold_features_80(train.iloc[tr_i], train.iloc[va_i], which=which)
        ytr = y[tr_i]
        te_tr = []
        te_va = []
        for c in cats:
            ktr = trn[c].astype(str).to_numpy()
            kva = val[c].astype(str).to_numpy()
            te_va.append(te_many(ktr, kva, ytr, m=20.0))
            te_tr.append(loo_te(ktr, ytr, m=20.0))
        nums = [c for c in NUM_MAIN if c in trn.columns]
        Xtr = np.column_stack(te_tr + [trn[c].to_numpy(float) for c in nums])
        Xva = np.column_stack(te_va + [val[c].to_numpy(float) for c in nums])
        Xtr = np.nan_to_num(Xtr, nan=0.0)
        Xva = np.nan_to_num(Xva, nan=0.0)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xva_s = sc.transform(Xva)
        best_pred, best_rmse, best_al = None, 1e9, None
        for al in (1.0, 3.0, 10.0, 30.0):
            mdl = Ridge(alpha=al)
            mdl.fit(Xtr_s, ytr.astype(float))
            pr = mdl.predict(Xtr_s)
            rmse = float(np.mean((pr - ytr) ** 2))
            pred = mdl.predict(Xva_s)
            if rmse < best_rmse:
                best_pred, best_rmse, best_al = pred, rmse, al
        oof[va_i] = best_pred
        print(f"  te_ridge {which} fold {fold} auc={auc(y[va_i], oof[va_i]):.4f} a={best_al} n_cats={len(cats)}", flush=True)
    a = auc(y, oof)
    print(f"=== TE-RIDGE {which} OOF {a:.5f} dt={time.time()-t0:.1f}s n_cats={len(cats)} ===")
    return oof, {"which": which, "oof": a, "n_cats": len(cats)}


def fit_cb(trn, val, ytr, yva, nums, cats, params):
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
    pred = model.predict(Pool(Xva, cat_features=cats))
    return pred, int(model.best_iteration_ or 0)


def cb_params(seed, ctr_complexity: int | None, rsm=1.0, depth=5, l2=10, ordered=True):
    p = dict(
        loss_function="RMSE",
        iterations=ITERS,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=l2,
        random_seed=seed,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Ordered" if ordered else "Plain",
        rsm=rsm,
    )
    if ctr_complexity is not None:
        p["max_ctr_complexity"] = ctr_complexity
    return p


def cb_probe(train, y, name, which, ctr, n_folds=5):
    n = len(y)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
    t0 = time.time()
    its = []
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        if which == "current":
            trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
            cats = list(CATS_CUR)
        else:
            trn, val, cats = fold_features_80(train.iloc[tr_i], train.iloc[va_i], which=which)
        ytr, yva = y[tr_i], y[va_i]
        pred, it = fit_cb(trn, val, ytr, yva, NUM_MAIN, cats, cb_params(2026 + fold, ctr))
        oof[va_i] = pred
        its.append(it)
        print(
            f"  cb {name} fold {fold} auc={auc(yva, pred):.4f} it={it} ncats={len(cats)} dt_fold",
            flush=True,
        )
    a = auc(y, oof)
    rec = {"name": name, "which": which, "ctr": ctr, "oof": a, "mean_it": float(np.mean(its)), "n_cats": len(cats), "n_folds": n_folds}
    print(f"=== CB {name} OOF {a:.5f} dt={time.time()-t0:.1f}s ===")
    np.save(OUT / f"exp9_{name}.npy", oof)
    return oof, rec


def blend_with_best(y, oof, tag):
    best = np.load(OUT / "best_oof.npy")
    lgb_p = OUT / "exp8d_lgb_w62.npy"
    lgb = np.load(lgb_p) if lgb_p.exists() else None
    mx = np.load(OUT / "exp8h_max2.npy")
    scores = {}
    scores[tag] = auc(y, oof)
    scores["max(best,new)"] = auc(y, np.maximum(rank(best), rank(oof)))
    scores["0.7best+0.3new"] = auc(y, 0.7 * rank(best) + 0.3 * rank(oof))
    scores["0.85best+0.15new"] = auc(y, 0.85 * rank(best) + 0.15 * rank(oof))
    scores["max(h,new)"] = auc(y, np.maximum(rank(mx), rank(oof)))
    if lgb is not None:
        scores["max(h,new)+lgb"] = auc(y, 0.8 * np.maximum(rank(mx), rank(oof)) + 0.2 * rank(lgb))
        scores["0.75h+0.15new+0.1lgb"] = auc(y, 0.75 * rank(mx) + 0.15 * rank(oof) + 0.10 * rank(lgb))
    print(f"  blends vs current best {auc(y,best):.5f}:")
    for k, v in scores.items():
        print(f"    {v:.5f}  {k}")
    return scores


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    report = {"best_before": auc(y, np.load(OUT / "best_oof.npy"))}

    report["residual"] = residual_scan(train, y)
    (OUT / "exp9_residual.json").write_text(json.dumps(report["residual"], indent=2))

    # --- portable TE ridge ---
    te_report = []
    for which in ("atomic", "sem", "80"):
        oof, rec = te_ridge(train, y, which, n_folds=10)
        np.save(OUT / f"exp9_te_{which}.npy", oof)
        rec["blends"] = blend_with_best(y, oof, f"te_{which}")
        te_report.append(rec)
        (OUT / "exp9_te_ridge.json").write_text(json.dumps(te_report, indent=2))
        if rec["oof"] > auc(y, np.load(OUT / "best_oof.npy")):
            np.save(OUT / "best_oof.npy", oof)
            print("UPDATED best_oof from TE", rec["oof"])
    report["te_ridge"] = te_report

    # --- CatBoost representation grid (5-fold Ordered, val-ES) ---
    grid = [
        ("cur_def", "current", None),
        ("cur_c1", "current", 1),
        ("atom_c4", "atomic", 4),
        ("atom_c2", "atomic", 2),
        ("sem_c1", "sem", 1),
        ("sem_c4", "sem", 4),
        ("c80_c1", "80", 1),
    ]
    cb_report = []
    ckpt = OUT / "exp9_cb_grid.json"
    done = set()
    if ckpt.exists():
        prev = json.loads(ckpt.read_text())
        cb_report = prev
        done = {r["name"] for r in prev}
    for name, which, ctr in grid:
        if name in done:
            print("skip", name)
            continue
        oof, rec = cb_probe(train, y, name, which, ctr, n_folds=5)
        rec["blends"] = blend_with_best(y, oof, name)
        cb_report.append(rec)
        ckpt.write_text(json.dumps(cb_report, indent=2))
        cur_best = np.load(OUT / "best_oof.npy")
        # only replace best_oof if 5-fold probe beats 10-fold ensemble — unlikely;
        # keep as candidate for scaling.
        if rec["oof"] >= 0.70:
            np.save(OUT / "best_oof.npy", oof)
            print("HIT 0.70 on 5-fold probe", name, rec["oof"])
        # rank-blend with 10-fold CB if complementary
        blended = 0.7 * rank(cur_best) + 0.3 * rank(oof)
        ba = auc(y, blended)
        print(f"  10fold-best blend 0.7/0.3 -> {ba:.5f}")
        if ba > auc(y, cur_best) + 0.0003:
            np.save(OUT / "best_oof.npy", blended)
            print("UPDATED best_oof blend", ba)

    report["cb_grid"] = cb_report
    (OUT / "exp9_hunt.json").write_text(json.dumps(report, indent=2, default=str))
    print("=== EXP9 DONE ===", json.dumps({k: report[k] for k in report if k != "residual"}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
