#!/usr/bin/env python3
"""Exp8a: Spark-portable 10-fold stack — frequency Ridge, HGB, TE-score Ridge, 2D hist, rank fusion.

No CatBoost native cats. Encoders/bins/medians fit on train fold only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor

from common import (
    auc,
    dump_json,
    fill_condition,
    load_train,
    numeric_block,
    per_source_rank,
    qcut_apply,
    rank01,
    save_oof,
    skf,
    te_pair,
    te_val_only,
)
from exp3_freq import additive_design


TE_KEYS = [
    ("src", 10),
    ("reg", 10),
    ("age", 8),
    ("src_reg", 15),
    ("src_age", 12),
    ("reg_age", 12),
    ("src_cq", 15),
    ("src_dq", 15),
    ("reg_cq", 15),
    ("reg_dq", 15),
    ("cq_dq", 12),
    ("src_cq_dq", 20),
    ("src_rq", 15),
    ("src_tq", 15),
    ("src_cq_age", 20),
    ("reg_cq_dq", 25),
]


def make_key_maps(trn, val, cond_tr, cond_va, nd=10, nc=10):
    dtr, dva, _ = qcut_apply(trn["days"], val["days"], nd)
    ctr, cva, _ = qcut_apply(cond_tr, cond_va, nc)
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
    rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
    med = pd.DataFrame({"s": src_tr, "c": cond_tr}).groupby("s")["c"].median()
    cr_tr = cond_tr / pd.Series(src_tr).map(med).astype(float).to_numpy()
    cr_va = cond_va / pd.Series(src_va).map(med).fillna(float(np.median(med))).astype(float).to_numpy()
    rq_tr, rq_va, _ = qcut_apply(trn["days"].to_numpy(float) / np.clip(cr_tr, 1e-6, None), val["days"].to_numpy(float) / np.clip(cr_va, 1e-6, None), 10)
    tq_tr, tq_va, _ = qcut_apply(trn["days"].to_numpy(float) * (1 - rk_tr), val["days"].to_numpy(float) * (1 - rk_va), 10)

    def S(a):
        return np.asarray(a).astype(str)

    tr = {
        "src": S(src_tr),
        "reg": S(trn["region"]),
        "age": S(trn["age_range"].astype(int)),
        "cq": S(ctr),
        "dq": S(dtr),
        "rq": S(rq_tr),
        "tq": S(tq_tr),
    }
    va = {
        "src": S(src_va),
        "reg": S(val["region"]),
        "age": S(val["age_range"].astype(int)),
        "cq": S(cva),
        "dq": S(dva),
        "rq": S(rq_va),
        "tq": S(tq_va),
    }
    for d in (tr, va):
        d["src_reg"] = d["src"] + "|" + d["reg"]
        d["src_age"] = d["src"] + "|" + d["age"]
        d["reg_age"] = d["reg"] + "|" + d["age"]
        d["src_cq"] = d["src"] + "|" + d["cq"]
        d["src_dq"] = d["src"] + "|" + d["dq"]
        d["reg_cq"] = d["reg"] + "|" + d["cq"]
        d["reg_dq"] = d["reg"] + "|" + d["dq"]
        d["cq_dq"] = d["cq"] + "|" + d["dq"]
        d["src_cq_dq"] = d["src"] + "|" + d["cq"] + "|" + d["dq"]
        d["src_rq"] = d["src"] + "|" + d["rq"]
        d["src_tq"] = d["src"] + "|" + d["tq"]
        d["src_cq_age"] = d["src"] + "|" + d["cq"] + "|" + d["age"]
        d["reg_cq_dq"] = d["reg"] + "|" + d["cq"] + "|" + d["dq"]
    return tr, va, rk_tr, rk_va, cr_tr, cr_va


def hgb_fit(Xtr, ytr, Xva, seed, max_depth=5, lr=0.04, min_leaf=80, l2=1.0, cat_idx=None):
    kw = dict(
        loss="squared_error",
        learning_rate=lr,
        max_depth=max_depth,
        max_iter=500,
        min_samples_leaf=min_leaf,
        l2_regularization=l2,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=seed,
        max_bins=255,
    )
    if cat_idx:
        kw["categorical_features"] = cat_idx
    m = HistGradientBoostingRegressor(**kw)
    m.fit(Xtr, ytr)
    return m.predict(Xva)


def main():
    df, y = load_train()
    src_levels = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    n = len(y)
    arms = {
        "hgb_main": np.zeros(n),
        "hgb_alt": np.zeros(n),
        "ridge_freq": np.zeros(n),
        "ridge_te": np.zeros(n),
        "hist2d": np.zeros(n),
        "iso_ratio": np.zeros(n),
        "iso_rate": np.zeros(n),
        "knn_src": np.zeros(n),
        "te_src_cq_dq": np.zeros(n),
        "te_src_cq": np.zeros(n),
        "te_cq_dq": np.zeros(n),
    }

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i].copy(), df.iloc[va_i].copy()
        ytr = y[tr_i].astype(float)
        yva = y[va_i]
        cond_tr, cond_va = fill_condition(trn, val)
        trn["condition_f"] = cond_tr
        val["condition_f"] = cond_va
        km_tr, km_va, rk_tr, rk_va, cr_tr, cr_va = make_key_maps(trn, val, cond_tr, cond_va)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)

        # --- TE scores ---
        te_tr_cols = []
        te_va_cols = []
        te_va_named = {}
        for name, m in TE_KEYS:
            a, b = te_pair(km_tr[name], km_va[name], ytr, m=m)
            te_tr_cols.append(a)
            te_va_cols.append(b)
            te_va_named[name] = b
        TE_tr = np.column_stack(te_tr_cols)
        TE_va = np.column_stack(te_va_cols)
        arms["te_src_cq_dq"][va_i] = te_va_named["src_cq_dq"]
        arms["te_src_cq"][va_i] = te_va_named["src_cq"]
        arms["te_cq_dq"][va_i] = te_va_named["cq_dq"]
        arms["hist2d"][va_i] = te_va_named["src_cq_dq"]

        ridge_te = Ridge(alpha=20.0)
        ridge_te.fit(TE_tr, ytr)
        arms["ridge_te"][va_i] = ridge_te.predict(TE_va)

        # --- frequency additive Ridge ---
        Xtr = additive_design(
            days_tr, cond_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"].to_numpy(float), src_levels, reg_levels
        )
        Xva = additive_design(
            days_va, cond_va, rk_va, src_va, val["region"].astype(str), val["age_range"].to_numpy(float), src_levels, reg_levels
        )
        rf = Ridge(alpha=8.0)
        rf.fit(Xtr, ytr)
        arms["ridge_freq"][va_i] = rf.predict(Xva)

        # --- isotonic ---
        ratio_tr = days_tr / np.clip(cr_tr, 1e-6, None)
        ratio_va = days_va / np.clip(cr_va, 1e-6, None)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(ratio_tr, ytr)
        arms["iso_ratio"][va_i] = iso.predict(ratio_va)
        iso2 = IsotonicRegression(out_of_bounds="clip")
        iso2.fit(days_tr * (1 - rk_tr), ytr)
        arms["iso_rate"][va_i] = iso2.predict(days_va * (1 - rk_va))

        # --- per-source KNN fallback global ---
        knn_pred = np.zeros(len(val))
        for s in src_levels:
            mtr = src_tr == s
            mva = src_va == s
            if mtr.sum() < 80 or mva.sum() == 0:
                continue
            Ztr = np.column_stack([days_tr[mtr] / 4000.0, np.log(np.clip(cond_tr[mtr], 1e-6, None))])
            Zva = np.column_stack([days_va[mva] / 4000.0, np.log(np.clip(cond_va[mva], 1e-6, None))])
            k = min(70, max(20, mtr.sum() // 10))
            knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
            knn.fit(Ztr, ytr[mtr])
            knn_pred[mva] = knn.predict(Zva)
        # fill empty with iso_ratio
        miss = knn_pred == 0
        knn_pred[miss] = arms["iso_ratio"][va_i][miss]
        arms["knn_src"][va_i] = knn_pred

        # --- HGB numeric dual world ---
        Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr)
        Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr)
        src_codes = {s: i for i, s in enumerate(src_levels)}
        reg_codes = {s: i for i, s in enumerate(reg_levels)}
        cat_tr = np.column_stack(
            [
                trn["source"].map(src_codes).fillna(-1).to_numpy(),
                trn["region"].map(reg_codes).fillna(-1).to_numpy(),
                trn["age_range"].to_numpy(),
                pd.Series(km_tr["cq"]).astype(int).to_numpy(),
                pd.Series(km_tr["dq"]).astype(int).to_numpy(),
            ]
        )
        cat_va = np.column_stack(
            [
                val["source"].map(src_codes).fillna(-1).to_numpy(),
                val["region"].map(reg_codes).fillna(-1).to_numpy(),
                val["age_range"].to_numpy(),
                pd.Series(km_va["cq"]).astype(int).to_numpy(),
                pd.Series(km_va["dq"]).astype(int).to_numpy(),
            ]
        )
        # main: cond_r world numerics + low-card cats. Do NOT stack sparse TE.
        Htr = np.hstack([Ntr.to_numpy(float), cat_tr])
        Hva = np.hstack([Nva.to_numpy(float), cat_va])
        nnum = Ntr.shape[1]
        cat_idx = [nnum, nnum + 1, nnum + 2, nnum + 3, nnum + 4]
        arms["hgb_main"][va_i] = hgb_fit(Htr, ytr, Hva, seed=fold, max_depth=5, lr=0.04, min_leaf=90, l2=2.0, cat_idx=cat_idx)
        arms["hgb_alt"][va_i] = hgb_fit(Htr, ytr, Hva, seed=100 + fold, max_depth=6, lr=0.03, min_leaf=70, l2=0.5, cat_idx=cat_idx)

        print(
            f"fold {fold} hgb={auc(yva, arms['hgb_main'][va_i]):.4f} alt={auc(yva, arms['hgb_alt'][va_i]):.4f} "
            f"freq={auc(yva, arms['ridge_freq'][va_i]):.4f} teR={auc(yva, arms['ridge_te'][va_i]):.4f} "
            f"hist={auc(yva, arms['hist2d'][va_i]):.4f} knn={auc(yva, arms['knn_src'][va_i]):.4f}",
            flush=True,
        )

    scores = {k: auc(y, v) for k, v in arms.items()}
    print("\n=== EXP8a ARM OOF ===")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v:.5f}")

    # rank fusion grid (report full-OOF; also nested-ish equal blend)
    def fuse(weights):
        s = np.zeros(n)
        for k, w in weights.items():
            s += w * rank01(arms[k])
        return s

    blends = {
        "w62_hgb": fuse({"hgb_main": 0.62, "hgb_alt": 0.38}),
        "hgb_te": fuse({"hgb_main": 0.55, "ridge_te": 0.25, "hist2d": 0.20}),
        "hgb_freq_te": fuse({"hgb_main": 0.40, "hgb_alt": 0.20, "ridge_freq": 0.15, "ridge_te": 0.15, "hist2d": 0.10}),
        "all_eq": fuse({k: 1.0 for k in arms}),
        "core5": fuse({"hgb_main": 0.30, "hgb_alt": 0.20, "ridge_te": 0.20, "ridge_freq": 0.15, "knn_src": 0.15}),
        "cb_style": fuse({"hgb_main": 0.34, "hgb_alt": 0.22, "ridge_te": 0.18, "hist2d": 0.12, "ridge_freq": 0.08, "iso_ratio": 0.06}),
    }
    blend_scores = {k: auc(y, v) for k, v in blends.items()}
    print("\n=== EXP8a BLENDS ===")
    for k, v in sorted(blend_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v:.5f}")

    best_name = max(blend_scores, key=blend_scores.get)
    best_s = blends[best_name]
    # also consider best single arm
    best_arm = max(scores, key=scores.get)
    if scores[best_arm] > blend_scores[best_name]:
        best_name, best_s = best_arm, arms[best_arm]

    report = {"arms": scores, "blends": blend_scores, "best": best_name, "best_auc": auc(y, best_s)}
    dump_json("exp8a_portable.json", report)
    for k, v in arms.items():
        save_oof(f"exp8a_{k}.npy", v)
    save_oof("exp8a_best.npy", best_s)
    print("BEST", best_name, report["best_auc"])
    print("WROTE exp8a_portable.json")


if __name__ == "__main__":
    main()
