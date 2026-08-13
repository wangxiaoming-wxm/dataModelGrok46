#!/usr/bin/env python3
"""Portable claim scorer: numpy / pandas only.

fit_fold_stats(train_df) -> stats
score(df, stats) -> p in [0, 1]   (closed identity-link arms + rank fuse)

Spark port (see FORMULA.md):
  medians, qcut edges, searchsorted rank, piecewise-linear greatest(x-k,0),
  map-join 2D table with 8-neighbour smoothing, TE maps, Ridge dot product,
  percent_rank fusion.

The ≥0.68 stack adds Spark GBTRegressor (RMSE, fixed 400 trees, no ES)
and is documented in FORMULA.md; this file stays tree-free so it can be
translated line-for-line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms import (
    fill_from_stats,
    fit_power_ridge,
    fit_spline_car,
    fit_table8,
    fit_te_ridge,
    fuse_rank,
    keys_from_edges,
    make_keys,
    predict_power,
    predict_spline,
    predict_table_pack,
    predict_te,
)
from common import (
    OUT,
    auc,
    dump_json,
    fill_condition,
    load_train,
    per_source_rank,
    save_oof,
    skf_splits,
    source_median_map,
)

# Frozen from 5-fold screen (exp6). Closed family, no trees.
CLOSED_WEIGHTS = {"te_ord": 0.45, "table8": 0.25, "spline": 0.20, "power": 0.10}
# A priori portable GBT stack (not used inside score(); Spark recipe).
GBT_WEIGHTS = {"lgb1": 0.40, "te_ord": 0.30, "table8": 0.15, "spline": 0.15}


def fit_fold_stats(train_df: pd.DataFrame, y=None) -> dict:
    """Fit closed-form arms on one train fold. No test labels, no id."""
    trn = train_df.reset_index(drop=True)
    if y is None:
        y = trn["label"].to_numpy(dtype=float)
    else:
        y = np.asarray(y, dtype=float)
    # dummy val = train, we only keep packs
    val = trn
    src_levels = sorted(trn["source"].astype(str).unique())
    reg_levels = sorted(trn["region"].astype(str).unique())
    src_levels_tab = src_levels + ["__UNK__"]
    cond_tr, cond_va = fill_condition(trn, val)
    src_tr = trn["source"].astype(str).to_numpy()
    rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
    rk_va = rk_tr
    km_tr, km_va, dtr, dva, ctr, cva, dbins, cbins, edges = make_keys(trn, val, cond_tr, cond_va, rk_tr, rk_va)
    _, _, pack_sp, _ = fit_spline_car(trn, val, y, src_levels, reg_levels, alpha=20.0)
    t_tr, t_va, pack_tab = fit_table8(trn, val, y, cond_tr, cond_va, dtr, dva, ctr, cva, src_levels_tab, alpha_glob=0.4)
    pack_tab["days_bins"] = dbins.tolist()
    pack_tab["cond_bins"] = cbins.tolist()
    _, _, pack_te = fit_te_ridge(km_tr, km_va, y, use_ordered=True, alpha=18.0, seed=7)
    _, _, pack_pw = fit_power_ridge(trn, val, y, cond_tr, cond_va, rk_tr, rk_va, src_levels, reg_levels, a=1.5, c=0.5, alpha=4.0)
    stats = {
        "med_map": {str(k): float(v) for k, v in source_median_map(src_tr, cond_tr).items()},
        "glob_cond": float(np.median(cond_tr)),
        "cond_ref_src": src_tr.tolist(),
        "cond_ref_val": cond_tr.tolist(),
        "edges": edges,
        "spline": pack_sp,
        "table8": pack_tab,
        "te_ord": pack_te,
        "power": pack_pw,
        "weights": CLOSED_WEIGHTS,
        "n_train": int(len(trn)),
        "prior": float(y.mean()),
    }
    return stats


def score_arms(df: pd.DataFrame, stats: dict) -> dict:
    """Return each closed arm on df. Pure numpy/pandas."""
    df = df.reset_index(drop=True)
    cond, rk, src = fill_from_stats(df, stats)
    km, dbin, cbin = keys_from_edges(df, cond, rk, stats["edges"], stats["med_map"])
    p_sp = predict_spline(df, cond, rk, stats["spline"])
    p_tab = predict_table_pack(src, dbin, cbin, stats["table8"])
    p_te = predict_te(km, stats["te_ord"])
    p_pw = predict_power(df, cond, rk, stats["power"])
    return {"spline": p_sp, "table8": p_tab, "te_ord": p_te, "power": p_pw}


def score(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Rank-fuse closed arms. Ranking is within the scored batch (Spark percent_rank)."""
    parts = score_arms(df, stats)
    fused = fuse_rank(parts, stats.get("weights", CLOSED_WEIGHTS))
    return np.clip(fused, 0.0, 1.0)


def cv_oof(train_df: pd.DataFrame, n_splits: int = 10):
    y = train_df["label"].to_numpy(dtype=float)
    oof = np.zeros(len(y))
    oof_te = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    splits = skf_splits(y.astype(int), n_splits)
    per_fold = []
    last = None
    for fold, (tr_i, va_i) in enumerate(splits):
        stats = fit_fold_stats(train_df.iloc[tr_i], y=y[tr_i])
        parts = score_arms(train_df.iloc[va_i], stats)
        pred = fuse_rank(parts, CLOSED_WEIGHTS)
        oof[va_i] = pred
        oof_te[va_i] = parts["te_ord"]
        oof_tab[va_i] = parts["table8"]
        last = stats
        rec = {"fold": fold, "auc": auc(y[va_i], pred), "te": auc(y[va_i], parts["te_ord"]), "table": auc(y[va_i], parts["table8"])}
        per_fold.append(rec)
        print(f"portable fold {fold} fuse={rec['auc']:.5f} te={rec['te']:.5f} tab={rec['table']:.5f}", flush=True)
    return oof, float(auc(y, oof)), per_fold, last, oof_te, oof_tab


def main():
    df, y = load_train()
    print("=== portable closed 5-fold ===", flush=True)
    oof5 = np.zeros(len(y))
    for tr_i, va_i in skf_splits(y, 5):
        stats = fit_fold_stats(df.iloc[tr_i], y=y[tr_i])
        oof5[va_i] = score(df.iloc[va_i], stats)
    print(f"closed 5fold {auc(y, oof5):.5f}", flush=True)

    print("=== portable closed 10-fold ===", flush=True)
    oof, a10, pf, stats, oof_te, oof_tab = cv_oof(df, n_splits=10)
    save_oof("portable_oof.npy", oof)
    save_oof("portable_te_oof.npy", oof_te)
    compact = {
        "weights": CLOSED_WEIGHTS,
        "gbt_weights_spark": GBT_WEIGHTS,
        "n_train_lastfold": stats["n_train"],
        "prior": stats["prior"],
        "edges": stats["edges"],
        "bmap": stats["power"]["bmap"],
        "power_ac": {"a": stats["power"]["a"], "c": stats["power"]["c"]},
        "table_neigh": stats["table8"]["neigh"],
        "table_alpha_glob": stats["table8"]["alpha_glob"],
        "te_keys": stats["te_ord"]["key_order"],
        "spline_n_coef": len(stats["spline"]["coef"]),
        "days_knots": stats["spline"]["days_knots"],
        "rk_knots": stats["spline"]["rk_knots"],
    }
    dump_json("portable_params_compact.json", compact)
    # full last-fold stats (includes cond_ref; needed for rank)
    (OUT / "portable_stats_lastfold.json").write_text(json.dumps(stats, ensure_ascii=False, default=str))
    dump_json(
        "portable_score.json",
        {
            "protocol": "StratifiedKFold seed=2026, fold-internal fit, identity Ridge + 8-neigh table + ordered-TE maps, rank fuse",
            "closed_weights": CLOSED_WEIGHTS,
            "auc5": auc(y, oof5),
            "auc10": a10,
            "per_fold": pf,
            "spark_gbt_stack": {
                "note": "Not inside score(); implement with Spark GBTRegressor RMSE, 400 trees, no ES",
                "weights": GBT_WEIGHTS,
                "honest_10fold_from_exp6": 0.6822016668537568,
            },
        },
    )
    print("WROTE portable_score.json closed 10fold", a10)


if __name__ == "__main__":
    main()
