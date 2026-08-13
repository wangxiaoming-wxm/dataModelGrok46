#!/usr/bin/env python3
"""Exp1: per-source piecewise-linear identity Ridge.

  p = clip( a_s + f_s(days) + g_s(condition or rank) + h_region + c*I(age>=8) + windows , 0, 1)

days: 8 segments (quantile knots). condition/rank: 6 segments.
Large sources get own spline deviations; small sources share the global spline + intercept.
5-fold screen Ridge alpha and g-on-rank vs g-on-condition; 10-fold report.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import Ridge

from common import (
    auc,
    dump_json,
    fill_condition,
    fold_report,
    load_train,
    ohe_levels_apply,
    per_source_rank,
    pw_linear_basis,
    quantile_knots,
    save_oof,
    skf_splits,
    standardize_apply,
    standardize_fit,
    window_matrix,
)

LARGE_N = 200
DAYS_SEG = 8
COND_SEG = 6
ALPHAS = [2.0, 8.0, 20.0, 50.0]


def spline_design(
    days,
    gvar,
    src,
    region,
    age,
    days_knots,
    g_knots,
    src_levels,
    reg_levels,
    large_src,
    extra=None,
):
    d = np.asarray(days, dtype=float) / 5000.0
    g = np.asarray(gvar, dtype=float)
    Bd = pw_linear_basis(d, np.asarray(days_knots) / 5000.0)
    Bg = pw_linear_basis(g, g_knots)
    src = np.asarray(src).astype(str)
    age8 = (np.asarray(age, dtype=float) >= 8).astype(float)[:, None]
    W, _ = window_matrix(days)
    X = [Bd, Bg, ohe_levels_apply(src, src_levels), ohe_levels_apply(region, reg_levels), age8, W]
    # hierarchical: large-source deviations (drop the intercept col of each basis)
    for s in large_src:
        ind = (src == s).astype(float)[:, None]
        X.append(ind * Bd[:, 1:])
        X.append(ind * Bg[:, 1:])
    if extra is not None:
        X.append(np.asarray(extra, dtype=float))
    return np.hstack(X)


def knots_for(trn_days, trn_g):
    return quantile_knots(trn_days, DAYS_SEG), quantile_knots(trn_g, COND_SEG)


def oof_for(df, y, splits, alpha, g_mode):
    n = len(y)
    oof = np.zeros(n)
    src_all = df["source"].astype(str).to_numpy()
    src_levels = sorted(src_all)
    # unique levels
    src_levels = sorted(set(src_levels))
    reg_levels = sorted(df["region"].astype(str).unique())
    per_fold = []
    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr = trn["days"].to_numpy(float)
        days_va = val["days"].to_numpy(float)
        if g_mode == "rank":
            g_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
            g_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        else:
            g_tr, g_va = cond_tr, cond_va
        dk, gk = knots_for(days_tr, g_tr)
        counts = pd_count(src_tr)
        large = [s for s in src_levels if counts.get(s, 0) >= LARGE_N]
        Xtr = spline_design(
            days_tr, g_tr, src_tr, trn["region"].astype(str), trn["age_range"], dk, gk, src_levels, reg_levels, large
        )
        Xva = spline_design(
            days_va, g_va, src_va, val["region"].astype(str), val["age_range"], dk, gk, src_levels, reg_levels, large
        )
        Xs, mu, sd = standardize_fit(Xtr)
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(Xs, ytr)
        pred = m.predict(standardize_apply(Xva, mu, sd))
        oof[va_i] = pred
        per_fold.append({"fold": fold, "auc": auc(y[va_i], pred), "n_feat": int(Xtr.shape[1]), "n_large": len(large)})
    return oof, per_fold


def pd_count(src):
    import pandas as pd

    return pd.Series(src).value_counts().to_dict()


def main():
    df, y = load_train()
    src_all = df["source"].astype(str).to_numpy()
    splits5 = skf_splits(y, 5)
    splits10 = skf_splits(y, 10)

    screen = []
    best = {"auc": -1.0}
    for g_mode in ("rank", "cond"):
        for alpha in ALPHAS:
            oof, _ = oof_for(df, y, splits5, alpha, g_mode)
            rec = {"g_mode": g_mode, "alpha": alpha, "auc5": auc(y, oof)}
            screen.append(rec)
            print(f"screen g={g_mode:4s} a={alpha:5.1f} auc5={rec['auc5']:.5f}", flush=True)
            if rec["auc5"] > best["auc"]:
                best = {"auc": rec["auc5"], "g_mode": g_mode, "alpha": alpha}

    reports = {}
    oofs = {}
    for tag, g_mode, alpha in [
        ("winner", best["g_mode"], best["alpha"]),
        ("rank_a8", "rank", 8.0),
        ("cond_a8", "cond", 8.0),
    ]:
        oof, pf = oof_for(df, y, splits10, alpha, g_mode)
        reports[tag] = {
            "g_mode": g_mode,
            "alpha": alpha,
            **fold_report(y, oof, src_all, pf),
        }
        oofs[tag] = oof
        print(f"10fold {tag} g={g_mode} a={alpha} auc={reports[tag]['auc']:.5f}", flush=True)

    save_oof("exp1_oof.npy", oofs["winner"])
    out = {
        "protocol": "StratifiedKFold seed=2026, fold-internal knots/medians/ranks, Ridge identity RMSE",
        "formula": "p=clip(a_s + f_s(days_8pw) + g_s(var_6pw) + h_region + c*I(age>=8) + windows, 0, 1)",
        "screen_5fold": screen,
        "best_screen": best,
        "report_10fold": reports,
    }
    dump_json("exp1_spline.json", out)
    print("WROTE exp1_spline.json", "winner", reports["winner"]["auc"])


if __name__ == "__main__":
    main()
