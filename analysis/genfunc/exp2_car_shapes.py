#!/usr/bin/env python3
"""Exp2: insurer-side car-shape additives on the identity Ridge.

CAR_10 inverted-U: I(CAR_10)*(rk-0.5)^2  (+ rk, rk^2 for flexibility)
CAR_1 monotone protection: I(CAR_1)*(1-rk)
CAR_7 worst-condition low risk: I(CAR_7)*rk

Compared against the Exp1 rank-spline backbone (same CV).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import Ridge

from common import (
    auc,
    car_of,
    dump_json,
    fill_condition,
    fold_report,
    load_train,
    per_source_rank,
    save_oof,
    skf_splits,
    standardize_apply,
    standardize_fit,
)
from exp1_spline import LARGE_N, knots_for, pd_count, spline_design


def car_terms(src, rk) -> tuple[np.ndarray, list[str]]:
    car = car_of(src)
    rk = np.asarray(rk, dtype=float)
    u = (rk - 0.5) ** 2
    names = [
        "ushape_car10",
        "rk_car10",
        "rk2_car10",
        "mono_car1",
        "rev_car7",
        "ushape_car0",
        "ushape_car2",
    ]
    X = np.column_stack(
        [
            (car == "CAR_10").astype(float) * u,
            (car == "CAR_10").astype(float) * rk,
            (car == "CAR_10").astype(float) * (rk**2),
            (car == "CAR_1").astype(float) * (1.0 - rk),
            (car == "CAR_7").astype(float) * rk,
            (car == "CAR_0").astype(float) * u,
            (car == "CAR_2").astype(float) * u,
        ]
    )
    return X, names


def oof_for(df, y, splits, alpha, with_car: bool):
    src_levels = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    oof = np.zeros(len(y))
    per_fold = []
    coef_acc = []
    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        dk, gk = knots_for(days_tr, rk_tr)
        large = [s for s in src_levels if pd_count(src_tr).get(s, 0) >= LARGE_N]
        extra_tr = extra_va = None
        if with_car:
            extra_tr, names = car_terms(src_tr, rk_tr)
            extra_va, _ = car_terms(src_va, rk_va)
        Xtr = spline_design(
            days_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"], dk, gk, src_levels, reg_levels, large, extra_tr
        )
        Xva = spline_design(
            days_va, rk_va, src_va, val["region"].astype(str), val["age_range"], dk, gk, src_levels, reg_levels, large, extra_va
        )
        Xs, mu, sd = standardize_fit(Xtr)
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(Xs, ytr)
        pred = m.predict(standardize_apply(Xva, mu, sd))
        oof[va_i] = pred
        rec = {"fold": fold, "auc": auc(y[va_i], pred)}
        if with_car:
            # last 7 columns are the car terms, after standardization
            rec["car_coef_std"] = {names[i]: float(m.coef_[-(7 - i)]) for i in range(7)}
            coef_acc.append(rec["car_coef_std"])
        per_fold.append(rec)
    mean_coef = None
    if coef_acc:
        keys = coef_acc[0].keys()
        mean_coef = {k: float(np.mean([c[k] for c in coef_acc])) for k in keys}
    return oof, per_fold, mean_coef


def main():
    df, y = load_train()
    src_all = df["source"].astype(str).to_numpy()
    splits5 = skf_splits(y, 5)
    splits10 = skf_splits(y, 10)

    screen = []
    best = {"auc": -1.0}
    for with_car in (False, True):
        for alpha in (8.0, 20.0):
            oof, _, _ = oof_for(df, y, splits5, alpha, with_car)
            rec = {"with_car": with_car, "alpha": alpha, "auc5": auc(y, oof)}
            screen.append(rec)
            print(f"screen car={with_car} a={alpha} auc5={rec['auc5']:.5f}", flush=True)
            if rec["auc5"] > best["auc"]:
                best = {"auc": rec["auc5"], **rec}

    reports = {}
    for tag, with_car, alpha in [("base", False, 8.0), ("car", True, 8.0), ("winner", best.get("with_car", True), best.get("alpha", 8.0))]:
        oof, pf, mean_coef = oof_for(df, y, splits10, alpha, with_car)
        reports[tag] = {
            "with_car": with_car,
            "alpha": alpha,
            "mean_car_coef_std": mean_coef,
            **fold_report(y, oof, src_all, pf),
        }
        if tag == "winner":
            save_oof("exp2_oof.npy", oof)
        print(f"10fold {tag} car={with_car} auc={reports[tag]['auc']:.5f} coef={mean_coef}", flush=True)

    dump_json(
        "exp2_car_shapes.json",
        {
            "protocol": "StratifiedKFold seed=2026, fold-internal rank(condition|source)",
            "terms": {
                "CAR_10": "I(CAR_10)*[(rk-0.5)^2, rk, rk^2]  inverted-U (middle quintile 25.7%)",
                "CAR_1": "I(CAR_1)*(1-rk)  monotone protection",
                "CAR_7": "I(CAR_7)*rk  worst condition, lower claim (deductible / moral-hazard screen)",
            },
            "screen_5fold": screen,
            "best_screen": best,
            "report_10fold": reports,
        },
    )
    print("WROTE exp2_car_shapes.json")


if __name__ == "__main__":
    main()
