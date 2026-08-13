#!/usr/bin/env python3
"""Exp1: per-source P(y|days,condition) — poly, inverted-U, isotonic, piecewise, KNN."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from common import (
    auc,
    dump_json,
    fill_condition,
    load_train,
    per_source_rank,
    save_oof,
    skf,
    source_median_map,
    cond_r_of,
)

MIN_SRC = 120  # below this, fall back to global


def feats(days, cond, rk):
    d = np.asarray(days, float)
    c = np.clip(np.asarray(cond, float), 1e-6, None)
    r = np.asarray(rk, float)
    dn = d / 5000.0
    return np.column_stack(
        [
            dn,
            dn**2,
            np.log1p(d),
            np.sqrt(np.clip(d, 0, None)) / 100.0,
            c,
            np.log(c),
            np.sqrt(c),
            1.0 / c,
            r,
            (r - 0.5) ** 2,
            dn * r,
            dn * (r - 0.5) ** 2,
            dn / np.sqrt(c),
            d * (1.0 - r) / 5000.0,
        ]
    )


def fit_predict_src(kind, Xtr, ytr, Xva, days_tr, days_va, cond_tr, cond_va, rk_tr, rk_va):
    if kind == "poly":
        m = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        m.fit(Xtr, ytr)
        return m.predict(Xva)
    if kind == "u":
        Utr = np.column_stack(
            [
                days_tr / 5000.0,
                (days_tr / 5000.0) ** 2,
                rk_tr,
                (rk_tr - 0.5) ** 2,
                (days_tr / 5000.0) * (rk_tr - 0.5) ** 2,
            ]
        )
        Uva = np.column_stack(
            [
                days_va / 5000.0,
                (days_va / 5000.0) ** 2,
                rk_va,
                (rk_va - 0.5) ** 2,
                (days_va / 5000.0) * (rk_va - 0.5) ** 2,
            ]
        )
        m = Ridge(alpha=2.0)
        m.fit(Utr, ytr)
        return m.predict(Uva)
    if kind == "iso_days":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(days_tr, ytr)
        return iso.predict(days_va)
    if kind == "iso_add":
        iso_d = IsotonicRegression(out_of_bounds="clip")
        iso_d.fit(days_tr, ytr)
        r_tr = ytr - iso_d.predict(days_tr)
        iso_c = IsotonicRegression(out_of_bounds="clip")
        # monotone protective: higher condition -> lower risk, so fit on -cond
        iso_c.fit(-cond_tr, r_tr)
        return iso_d.predict(days_va) + iso_c.predict(-cond_va)
    if kind == "iso_u":
        iso_d = IsotonicRegression(out_of_bounds="clip")
        iso_d.fit(days_tr, ytr)
        r_tr = ytr - iso_d.predict(days_tr)
        u_tr = (rk_tr - 0.5) ** 2
        u_va = (rk_va - 0.5) ** 2
        iso_u = IsotonicRegression(out_of_bounds="clip")
        iso_u.fit(u_tr, r_tr)
        return iso_d.predict(days_va) + iso_u.predict(u_va)
    if kind == "piecewise":
        m = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(20, len(ytr) // 25), random_state=0)
        m.fit(np.column_stack([days_tr, cond_tr, rk_tr]), ytr)
        return m.predict(np.column_stack([days_va, cond_va, rk_va]))
    if kind == "knn80":
        Ztr = np.column_stack([days_tr / 4000.0, np.log(np.clip(cond_tr, 1e-6, None))])
        Zva = np.column_stack([days_va / 4000.0, np.log(np.clip(cond_va, 1e-6, None))])
        k = min(80, max(15, len(ytr) // 8))
        m = KNeighborsRegressor(n_neighbors=k, weights="distance")
        m.fit(Ztr, ytr)
        return m.predict(Zva)
    raise ValueError(kind)


def main():
    df, y = load_train()
    kinds = ["poly", "u", "iso_days", "iso_add", "iso_u", "piecewise", "knn80"]
    oof = {k: np.full(len(y), np.nan) for k in kinds}
    oof_blend = np.full(len(y), np.nan)
    src_all = df["source"].astype(str).to_numpy()
    sources = sorted(df["source"].unique())

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i]
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr = trn["days"].to_numpy(float)
        days_va = val["days"].to_numpy(float)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        Xtr_g = feats(days_tr, cond_tr, rk_tr)
        Xva_g = feats(days_va, cond_va, rk_va)
        # global fallbacks
        global_pred = {}
        for k in kinds:
            global_pred[k] = fit_predict_src(
                k, Xtr_g, ytr, Xva_g, days_tr, days_va, cond_tr, cond_va, rk_tr, rk_va
            )
        pred = {k: global_pred[k].copy() for k in kinds}
        for s in sources:
            mtr = src_tr == s
            mva = src_va == s
            if mtr.sum() < MIN_SRC or mva.sum() == 0:
                continue
            Xtr = feats(days_tr[mtr], cond_tr[mtr], rk_tr[mtr])
            Xva = feats(days_va[mva], cond_va[mva], rk_va[mva])
            for k in kinds:
                try:
                    pred[k][mva] = fit_predict_src(
                        k,
                        Xtr,
                        ytr[mtr],
                        Xva,
                        days_tr[mtr],
                        days_va[mva],
                        cond_tr[mtr],
                        cond_va[mva],
                        rk_tr[mtr],
                        rk_va[mva],
                    )
                except Exception:
                    pass
        for k in kinds:
            oof[k][va_i] = pred[k]
        # RMSE-optimal blend of poly + knn + iso_u on train via simple avg of ranks later
        oof_blend[va_i] = 0.45 * pred["poly"] + 0.35 * pred["knn80"] + 0.20 * pred["iso_u"]
        print(
            f"fold {fold} poly={auc(y[va_i], pred['poly']):.4f} knn={auc(y[va_i], pred['knn80']):.4f} "
            f"iso_add={auc(y[va_i], pred['iso_add']):.4f} pw={auc(y[va_i], pred['piecewise']):.4f}",
            flush=True,
        )

    report = {"global_oof": {k: auc(y, oof[k]) for k in kinds}, "blend": auc(y, oof_blend)}
    # per-source holdout AUC using OOF (honest)
    per = {}
    for s in sources:
        m = src_all == s
        per[s] = {
            "n": int(m.sum()),
            "posrate": float(y[m].mean()),
            **{k: auc(y[m], oof[k][m]) for k in kinds},
            "blend": auc(y[m], oof_blend[m]),
        }
    report["per_source"] = per
    print("\n=== EXP1 GLOBAL OOF ===")
    for k, v in report["global_oof"].items():
        print(f"  {k:12s} {v:.5f}")
    print(f"  blend        {report['blend']:.5f}")
    print("\n=== EXP1 PER SOURCE (poly / knn / iso_u / blend) ===")
    for s, r in per.items():
        print(
            f"  {s:18s} n={r['n']:4d} rate={r['posrate']:.3f} "
            f"poly={r['poly']:.3f} knn={r['knn80']:.3f} iso_u={r['iso_u']:.3f} blend={r['blend']:.3f}"
        )
    dump_json("exp1_persource.json", report)
    save_oof("exp1_oof_poly.npy", oof["poly"])
    save_oof("exp1_oof_knn.npy", oof["knn80"])
    save_oof("exp1_oof_blend.npy", oof_blend)
    print("WROTE exp1_persource.json")


if __name__ == "__main__":
    main()
