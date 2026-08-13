#!/usr/bin/env python3
"""Portable identity-link claim scorer (numpy / pandas only).

Given statistics fit on a training fold, score any dataframe.
Spark port: every operation is median/quantile, searchsorted rank,
piecewise-linear truncated basis, map-join 2D table, and a dot product.

Do not pass test labels or id. Stats must be fit on the train fold only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    OUT,
    auc,
    apply_bins,
    cond_r_of,
    dump_json,
    load_train,
    per_source_rank,
    power_score,
    qcut_apply,
    quantile_knots,
    save_oof,
    skf_splits,
    source_b_map,
    source_median_map,
    standardize_apply,
    standardize_fit,
    te_group,
    te_pair,
)
from exp1_spline import LARGE_N, spline_design
from exp2_car_shapes import car_terms
from exp3_2d_smooth import NEIGH, fill_grid, lookup, make_global, predict_table, smooth_grid, src_index


def _fill_with_medmap(df, med_map: dict, glob: float) -> np.ndarray:
    med = df["source"].map(med_map)
    return df["condition"].fillna(med).fillna(glob).to_numpy(dtype=float)


def windows_from_spec(days: np.ndarray) -> np.ndarray:
    d = np.asarray(days, dtype=float)
    cols = [((d >= lo) & (d < hi)).astype(float) for _, lo, hi in WINDOW_SPEC]
    cols.append((d >= 10000.0).astype(float))
    return np.column_stack(cols)


def fit_fold_stats(train_df: pd.DataFrame, y=None, alpha: float = 25.0) -> dict:
    """Fit generating-process statistics + identity Ridge on one train fold."""
    trn = train_df.reset_index(drop=True)
    if y is None:
        y = trn["label"].to_numpy(dtype=float)
    else:
        y = np.asarray(y, dtype=float)
    cond = trn["condition"].to_numpy(dtype=float)
    src = trn["source"].astype(str).to_numpy()
    med_map = source_median_map(src, np.where(np.isnan(cond), np.nan, cond))
    # source_median_map uses the provided cond including NaN — recompute properly
    cond_f = trn["condition"].fillna(trn.groupby("source")["condition"].transform("median"))
    glob = float(trn["condition"].median())
    cond_f = cond_f.fillna(glob).to_numpy(dtype=float)
    med_map = source_median_map(src, cond_f)
    days = trn["days"].to_numpy(dtype=float)
    rk = per_source_rank(src, cond_f, src, cond_f)
    src_levels = sorted(pd.unique(src))
    reg_levels = sorted(trn["region"].astype(str).unique())
    large = [s for s in src_levels if int((src == s).sum()) >= LARGE_N]
    days_knots = quantile_knots(days, 8).tolist()
    rk_knots = quantile_knots(rk, 6).tolist()
    extra, car_names = car_terms(src, rk)
    X = spline_design(
        days, rk, src, trn["region"].astype(str), trn["age_range"], days_knots, rk_knots, src_levels, reg_levels, large, extra
    )
    bmap = source_b_map(src, days, cond_f, y)
    sc = power_score(days, cond_f, rk, src, 1.5, 1.227, bmap) / 1e5
    # 2D table days_q5 × cond_q10
    dbin, _, days_bins = qcut_apply(days, days, 5)
    cbin, _, cond_bins = qcut_apply(cond_f, cond_f, 10)
    src_levels_tab = src_levels + ["__UNK__"]
    idx = src_index(src, src_levels_tab)
    sm, ct = fill_grid(idx, dbin, cbin, y, len(src_levels_tab), 5, 10)
    prior = float(y.mean())
    neigh = NEIGH[2]
    grid = smooth_grid(sm, ct, prior, 10.0, neigh["w0"], neigh["ws"], neigh["wd"])
    glob_grid = make_global(sm, ct, prior, 10.0, neigh)
    t = predict_table(src, dbin, cbin, src_levels_tab, sm, ct, prior, 10.0, neigh, 0.25, glob_grid)
    # a few TE keys
    cq = dbin  # placeholder unused
    te_src = te_group(src, y)
    te_reg = te_group(trn["region"].astype(str), y)
    key_cqdq = np.asarray(dbin).astype(str) + "|" + np.asarray(cbin).astype(str)
    key_src_cqdq = src.astype(str) + "|" + key_cqdq
    te_cqdq = te_group(key_cqdq, y)
    te_src_cqdq = te_group(key_src_cqdq, y)

    te_src_v, _, _, _ = te_pair(src, src, y, m=10)
    te_reg_v, _, _, _ = te_pair(trn["region"].astype(str), trn["region"].astype(str), y, m=10)
    te_cqdq_v, _, _, _ = te_pair(key_cqdq, key_cqdq, y, m=12)
    te_src_cqdq_v, _, _, _ = te_pair(key_src_cqdq, key_src_cqdq, y, m=10)

    J = np.hstack(
        [
            X,
            t[:, None],
            sc[:, None],
            te_src_cqdq_v[:, None],
            te_cqdq_v[:, None],
            te_src_v[:, None],
            te_reg_v[:, None],
            trn["x20"].to_numpy(float)[:, None],
            trn["V"].to_numpy(float)[:, None],
        ]
    )
    Xs, mu, sd = standardize_fit(J)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xs, y)

    # serialize 2D sums/counts for Spark
    sm_list = sm.tolist()
    ct_list = ct.tolist()
    stats = {
        "med_map": {str(k): float(v) for k, v in med_map.items()},
        "glob_cond": glob,
        "src_levels": src_levels,
        "reg_levels": reg_levels,
        "large_src": large,
        "days_knots": days_knots,
        "rk_knots": rk_knots,
        "bmap": {str(k): float(v) for k, v in bmap.items()},
        "power_a": 1.5,
        "power_c": 1.227,
        "days_bins": days_bins.tolist(),
        "cond_bins": cond_bins.tolist(),
        "src_levels_tab": src_levels_tab,
        "table_sm": sm_list,
        "table_ct": ct_list,
        "table_prior": prior,
        "table_m": 10.0,
        "neigh": {"w0": neigh["w0"], "ws": neigh["ws"], "wd": neigh["wd"]},
        "alpha_glob": 0.25,
        "te_src": {str(k): {"sum": float(r["sum"]), "count": int(r["count"])} for k, r in te_src.iterrows()},
        "te_reg": {str(k): {"sum": float(r["sum"]), "count": int(r["count"])} for k, r in te_reg.iterrows()},
        "te_cqdq": {str(k): {"sum": float(r["sum"]), "count": int(r["count"])} for k, r in te_cqdq.iterrows()},
        "te_src_cqdq": {str(k): {"sum": float(r["sum"]), "count": int(r["count"])} for k, r in te_src_cqdq.iterrows()},
        "ridge_coef": model.coef_.tolist(),
        "ridge_intercept": float(model.intercept_),
        "ridge_mu": mu.tolist(),
        "ridge_sd": sd.tolist(),
        "ridge_alpha": alpha,
        "car_names": car_names,
        "n_train": int(len(trn)),
        "prior": prior,
    }
    return stats


def _te_from_dict(keys, d: dict, prior: float, m: float) -> np.ndarray:
    out = np.empty(len(keys), dtype=float)
    for i, k in enumerate(keys):
        rec = d.get(str(k))
        if rec is None:
            out[i] = prior
        else:
            out[i] = (rec["sum"] + prior * m) / (rec["count"] + m)
    return out


def score(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Score any dataframe with frozen fold statistics. Pure numpy/pandas."""
    src = df["source"].astype(str).to_numpy()
    cond_f = _fill_with_medmap(df, stats["med_map"], stats["glob_cond"])
    days = df["days"].to_numpy(dtype=float)
    # rank vs the training-fold condition lists are not stored raw; use
    # searchsorted against per-source empirical CDF encoded by... we need ref.
    # Reconstruct rank from med_map only is wrong. Store cond refs in stats.
    # Fallback: if cond_ref present use it; else percentile vs knots is wrong.
    # We store cond_ref in stats when available.
    if "cond_ref" in stats:
        rk = per_source_rank(src, cond_f, np.array(stats["cond_ref_src"]), np.array(stats["cond_ref_val"]))
    else:
        # approximate: rank against a uniform grid is bad; use cond vs source median as proxy
        # Better: use stored rk_knots only as g-variable on min-max scaled cond_r
        cr = cond_r_of(src, cond_f, stats["med_map"])
        # map cond_r through a logistic-ish rank proxy in [0,1]
        rk = 1.0 / (1.0 + np.exp(-2.0 * (np.log(np.clip(cr, 1e-6, None)))))
        rk = np.clip(rk, 0.0, 1.0)
    extra, _ = car_terms(src, rk)
    X = spline_design(
        days,
        rk,
        src,
        df["region"].astype(str),
        df["age_range"],
        stats["days_knots"],
        stats["rk_knots"],
        stats["src_levels"],
        stats["reg_levels"],
        stats["large_src"],
        extra,
    )
    sc = power_score(days, cond_f, rk, src, stats["power_a"], stats["power_c"], stats["bmap"]) / 1e5
    dbin = apply_bins(days, stats["days_bins"])
    cbin = apply_bins(cond_f, stats["cond_bins"])
    sm = np.asarray(stats["table_sm"], dtype=float)
    ct = np.asarray(stats["table_ct"], dtype=float)
    neigh = stats["neigh"]
    grid = smooth_grid(sm, ct, stats["table_prior"], stats["table_m"], neigh["w0"], neigh["ws"], neigh["wd"])
    glob = make_global(sm, ct, stats["table_prior"], stats["table_m"], neigh)
    t = predict_table(
        src, dbin, cbin, stats["src_levels_tab"], sm, ct, stats["table_prior"], stats["table_m"], neigh, stats["alpha_glob"], glob
    )
    key_cqdq = np.asarray(dbin).astype(str) + "|" + np.asarray(cbin).astype(str)
    key_src_cqdq = src.astype(str) + "|" + key_cqdq
    te_src_cqdq_v = _te_from_dict(key_src_cqdq, stats["te_src_cqdq"], stats["prior"], 10.0)
    te_cqdq_v = _te_from_dict(key_cqdq, stats["te_cqdq"], stats["prior"], 12.0)
    te_src_v = _te_from_dict(src, stats["te_src"], stats["prior"], 10.0)
    te_reg_v = _te_from_dict(df["region"].astype(str), stats["te_reg"], stats["prior"], 10.0)
    J = np.hstack(
        [
            X,
            t[:, None],
            sc[:, None],
            te_src_cqdq_v[:, None],
            te_cqdq_v[:, None],
            te_src_v[:, None],
            te_reg_v[:, None],
            df["x20"].to_numpy(float)[:, None],
            df["V"].to_numpy(float)[:, None],
        ]
    )
    Xs = standardize_apply(J, np.asarray(stats["ridge_mu"]), np.asarray(stats["ridge_sd"]))
    pred = Xs @ np.asarray(stats["ridge_coef"]) + float(stats["ridge_intercept"])
    return np.clip(pred, 0.0, 1.0)


def fit_fold_stats_with_ref(train_df: pd.DataFrame, y=None, alpha: float = 25.0) -> dict:
    stats = fit_fold_stats(train_df, y=y, alpha=alpha)
    trn = train_df.reset_index(drop=True)
    src = trn["source"].astype(str).to_numpy()
    cond_f = _fill_with_medmap(trn, stats["med_map"], stats["glob_cond"])
    stats["cond_ref_src"] = src.tolist()
    stats["cond_ref_val"] = cond_f.tolist()
    return stats


def cv_oof(train_df: pd.DataFrame, n_splits: int = 10, alpha: float = 25.0):
    y = train_df["label"].to_numpy(dtype=float)
    oof = np.zeros(len(y))
    splits = skf_splits(y.astype(int), n_splits)
    per_fold = []
    last_stats = None
    for fold, (tr_i, va_i) in enumerate(splits):
        stats = fit_fold_stats_with_ref(train_df.iloc[tr_i], y=y[tr_i], alpha=alpha)
        pred = score(train_df.iloc[va_i], stats)
        oof[va_i] = pred
        per_fold.append({"fold": fold, "auc": auc(y[va_i], pred)})
        last_stats = stats
        print(f"portable fold {fold} auc={per_fold[-1]['auc']:.5f}", flush=True)
    return oof, float(auc(y, oof)), per_fold, last_stats


def main():
    df, y = load_train()
    print("=== portable_score 5-fold screen alpha ===", flush=True)
    splits5 = skf_splits(y, 5)
    screen = []
    best = {"auc": -1.0, "alpha": 25.0}
    for alpha in (12.0, 25.0, 40.0):
        oof = np.zeros(len(y))
        for tr_i, va_i in splits5:
            stats = fit_fold_stats_with_ref(df.iloc[tr_i], y=y[tr_i], alpha=alpha)
            oof[va_i] = score(df.iloc[va_i], stats)
        a = auc(y, oof)
        screen.append({"alpha": alpha, "auc5": a})
        print(f"screen alpha={alpha} auc5={a:.5f}", flush=True)
        if a > best["auc"]:
            best = {"auc": a, "alpha": alpha}

    print("=== portable_score 10-fold report ===", flush=True)
    oof, a10, pf, stats = cv_oof(df, n_splits=10, alpha=best["alpha"])
    save_oof("portable_oof.npy", oof)
    # drop bulky cond_ref from the published stats dump? keep it — Spark needs it for rank.
    # Write a compact params file without cond_ref for the formula doc, plus full json.
    compact = {k: v for k, v in stats.items() if k not in ("cond_ref_src", "cond_ref_val", "table_sm", "table_ct")}
    compact["n_table_src"] = len(stats["src_levels_tab"])
    dump_json("portable_params_compact.json", compact)
    # full stats for Spark (may be large)
    (OUT / "portable_stats_lastfold.json").write_text(json.dumps(stats, ensure_ascii=False, default=str))
    dump_json(
        "portable_score.json",
        {
            "protocol": "StratifiedKFold, fold-internal fit, identity Ridge, clip[0,1]",
            "screen_5fold": screen,
            "best_alpha": best,
            "report_10fold": {"auc": a10, "per_fold": pf},
        },
    )
    print("WROTE portable_score.json 10fold", a10)


if __name__ == "__main__":
    main()
