#!/usr/bin/env python3
"""Exp4: closed-form power grid on the identity link.

  score = days^a / cond^{b_s} * (1-rk)^c
  p = clip( a_s + β * score + h_region + c*I(age>=8) + d·windows , 0, 1)

b_s is chosen fold-internally (max bidirectional AUC of days/cond^b).
5-fold screens (a,c); 10-fold reports the winner.
Records every (a,c) whose 5-fold Ridge OOF AUC > 0.62, plus raw both-AUC of the transform.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import Ridge

from common import (
    auc,
    both_auc,
    dump_json,
    fill_condition,
    fold_report,
    load_train,
    ohe_levels_apply,
    per_source_rank,
    power_score,
    save_oof,
    skf_splits,
    source_b_map,
    standardize_apply,
    standardize_fit,
    window_matrix,
)

AS = [0.5, 0.8, 1.0, 1.2, 1.5]
CS = [0.0, 0.5, 0.8, 1.0, 1.227, 1.5]
ALPHAS = [4.0, 12.0]


def design(days, cond, rk, src, region, age, a, c, bmap, src_levels, reg_levels):
    src = np.asarray(src).astype(str)
    sc = power_score(days, cond, rk, src, a, c, bmap)
    # scale for numerical stability (days^1.5 can be huge)
    sc = sc / 1.0e5
    age8 = (np.asarray(age, dtype=float) >= 8).astype(float)[:, None]
    W, _ = window_matrix(days)
    X = [
        sc[:, None],
        np.log1p(np.clip(sc, 0, None))[:, None],
        ohe_levels_apply(src, src_levels),
        ohe_levels_apply(region, reg_levels),
        age8,
        W,
    ]
    return np.hstack(X), sc


def oof_ridge(df, y, splits, a, c, alpha):
    src_levels = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    oof = np.zeros(len(y))
    raw = np.zeros(len(y))
    per_fold = []
    bmaps = []
    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
        Xtr, _ = design(days_tr, cond_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"], a, c, bmap, src_levels, reg_levels)
        Xva, sc_va = design(days_va, cond_va, rk_va, src_va, val["region"].astype(str), val["age_range"], a, c, bmap, src_levels, reg_levels)
        Xs, mu, sd = standardize_fit(Xtr)
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(Xs, ytr)
        pred = m.predict(standardize_apply(Xva, mu, sd))
        oof[va_i] = pred
        raw[va_i] = sc_va
        per_fold.append({"fold": fold, "auc": auc(y[va_i], pred), "bmap": {k.split("|")[0]: v for k, v in bmap.items()}})
        bmaps.append(bmap)
    return oof, raw, per_fold, bmaps


def main():
    df, y = load_train()
    src_all = df["source"].astype(str).to_numpy()
    splits5 = skf_splits(y, 5)
    splits10 = skf_splits(y, 10)

    # raw both-AUC of the transform on full data is slightly optimistic; still useful for ranking the family.
    # Honest number is the Ridge OOF.
    cond_full, = __import__("common").fill_condition(df)
    rk_full = per_source_rank(src_all, cond_full, src_all, cond_full)
    days_full = df["days"].to_numpy(float)
    b_full = source_b_map(src_all, days_full, cond_full, y)

    screen = []
    best = {"auc": -1.0}
    above = []
    for a in AS:
        for c in CS:
            for alpha in ALPHAS:
                oof, raw, _, _ = oof_ridge(df, y, splits5, a, c, alpha)
                a5 = auc(y, oof)
                raw_a = both_auc(y, power_score(days_full, cond_full, rk_full, src_all, a, c, b_full))
                rec = {"a": a, "c": c, "alpha": alpha, "auc5": a5, "raw_both_auc": raw_a}
                screen.append(rec)
                print(f"screen a={a:.3f} c={c:.3f} al={alpha:.0f} auc5={a5:.5f} raw={raw_a:.5f}", flush=True)
                if a5 > 0.62:
                    above.append(rec)
                if a5 > best["auc"]:
                    best = {**rec, "auc": a5}

    screen.sort(key=lambda z: -z["auc5"])
    above.sort(key=lambda z: -z["auc5"])

    oof, raw, pf, bmaps = oof_ridge(df, y, splits10, best["a"], best["c"], best["alpha"])
    # mean b_s across folds
    keys = set()
    for bm in bmaps:
        keys.update(bm)
    mean_b = {k: float(np.mean([bm.get(k, 0.5) for bm in bmaps])) for k in sorted(keys)}

    report = fold_report(y, oof, src_all, pf)
    save_oof("exp4_oof.npy", oof)
    save_oof("exp4_raw_oof.npy", raw)

    # also 10-fold the historically best (a=1.5, c=1.227)
    oof_h, _, pf_h, _ = oof_ridge(df, y, splits10, 1.5, 1.227, 12.0)
    hist = fold_report(y, oof_h, src_all, pf_h)

    dump_json(
        "exp4_grid_closed.json",
        {
            "protocol": "5-fold screen (a,c,alpha), 10-fold report. b_s fold-internal. Ridge identity.",
            "formula": "p=clip(a_s + β*days^a/cond^{b_s}*(1-rk)^c + h_region + c*I(age>=8) + windows, 0, 1)",
            "best_screen": best,
            "params_auc5_gt_0.62": above,
            "screen_top20": screen[:20],
            "mean_b_s_10fold": mean_b,
            "report_10fold_winner": report,
            "report_10fold_a15_c1227": hist,
        },
    )
    print("WROTE exp4_grid_closed.json winner", report["auc"], "hist", hist["auc"])
    print("AUC>0.62 count", len(above), "best", best)


if __name__ == "__main__":
    main()
