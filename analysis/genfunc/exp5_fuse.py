#!/usr/bin/env python3
"""Exp5: fuse 2D table scores with Ridge additive / rank terms. Target ≥0.67.

Arms (all fold-internal):
  - spline+car Ridge (exp1/2)
  - 8-neigh 2D table (exp3)
  - closed-form power Ridge (exp4)
  - multi-key TE Ridge (src×days_q5×cond_q10 and friends)
  - joint identity Ridge: spline + car + table + power + TE
  - LightGBM RMSE, FIXED rounds (no early stopping) — Spark GBT analogue
  - rank blends

5-fold screens blend weights / LGB rounds; 10-fold reports frozen choices.
Equal-rank blend (no weight tune) is also reported as the leak-free fusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import (
    auc,
    dump_json,
    fill_condition,
    fold_report,
    load_train,
    ohe_levels_apply,
    per_source_rank,
    power_score,
    qcut_apply,
    rank01,
    save_oof,
    skf_splits,
    source_b_map,
    standardize_apply,
    standardize_fit,
    te_pair,
    window_matrix,
)
from exp1_spline import LARGE_N, knots_for, pd_count, spline_design
from exp2_car_shapes import car_terms
from exp3_2d_smooth import NEIGH, fill_grid, make_global, predict_table, src_index

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False


TE_KEYS = [
    ("src", 10),
    ("reg", 10),
    ("src_reg", 15),
    ("src_age", 12),
    ("reg_age", 12),
    ("src_cq", 15),
    ("src_dq", 12),
    ("cq_dq", 12),
    ("src_cq_dq", 10),
    ("src_rq", 15),
    ("reg_dq", 15),
]


def key_maps(trn, val, cond_tr, cond_va, rk_tr, rk_va):
    dtr, dva, _ = qcut_apply(trn["days"], val["days"], 5)
    ctr, cva, _ = qcut_apply(cond_tr, cond_va, 10)
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    med = pd.DataFrame({"s": src_tr, "c": cond_tr}).groupby("s")["c"].median()
    cr_tr = cond_tr / pd.Series(src_tr).map(med).astype(float).to_numpy()
    cr_va = cond_va / pd.Series(src_va).map(med).fillna(float(np.median(med))).astype(float).to_numpy()
    rq_tr, rq_va, _ = qcut_apply(trn["days"].to_numpy(float) / np.clip(cr_tr, 1e-6, None), val["days"].to_numpy(float) / np.clip(cr_va, 1e-6, None), 10)

    def S(a):
        return np.asarray(a).astype(str)

    tr = {
        "src": S(src_tr),
        "reg": S(trn["region"]),
        "age": S(trn["age_range"].astype(int)),
        "cq": S(ctr),
        "dq": S(dtr),
        "rq": S(rq_tr),
    }
    va = {
        "src": S(src_va),
        "reg": S(val["region"]),
        "age": S(val["age_range"].astype(int)),
        "cq": S(cva),
        "dq": S(dva),
        "rq": S(rq_va),
    }
    for d in (tr, va):
        d["src_reg"] = d["src"] + "|" + d["reg"]
        d["src_age"] = d["src"] + "|" + d["age"]
        d["reg_age"] = d["reg"] + "|" + d["age"]
        d["src_cq"] = d["src"] + "|" + d["cq"]
        d["src_dq"] = d["src"] + "|" + d["dq"]
        d["cq_dq"] = d["cq"] + "|" + d["dq"]
        d["src_cq_dq"] = d["src"] + "|" + d["cq"] + "|" + d["dq"]
        d["src_rq"] = d["src"] + "|" + d["rq"]
        d["reg_dq"] = d["reg"] + "|" + d["dq"]
    return tr, va, dtr, dva, ctr, cva


def table_scores(src_tr, src_va, dtr, dva, ctr, cva, ytr, src_levels, neigh, m=10.0, alpha_glob=0.25):
    idx_tr = src_index(src_tr, src_levels)
    sm, ct = fill_grid(idx_tr, dtr, ctr, ytr, len(src_levels), 5, 10)
    prior = float(np.mean(ytr))
    glob = make_global(sm, ct, prior, m, neigh)
    # LOO-ish train scores: use cell mean with this row removed, then 8-neigh (neighbors keep full counts)
    grid = None  # val only needed for OOF; train LOO for joint ridge
    from exp3_2d_smooth import smooth_grid, lookup

    grid = smooth_grid(sm, ct, prior, m, neigh["w0"], neigh["ws"], neigh["wd"])
    va = predict_table(src_va, dva, cva, src_levels, sm, ct, prior, m, neigh, alpha_glob, glob)
    # train: apply same table (slight in-sample); for Ridge features we use LOO cell + neigh
    tr = predict_table(src_tr, dtr, ctr, src_levels, sm, ct, prior, m, neigh, alpha_glob, glob)
    # cheap LOO correction on the cell only
    # p_loo = (sum - y + m*prior) / (cnt-1+m) blended toward tr
    return tr, va


def lgb_predict(Xtr, ytr, Xva, cat_idx, rounds, leaves, seed):
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_idx, free_raw_data=False)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=leaves,
        min_data_in_leaf=80,
        feature_fraction=0.75,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=8.0,
        verbose=-1,
        seed=seed,
        num_threads=1,
    )
    m = lgb.train(params, dtr, num_boost_round=rounds)  # NO early stopping
    return m.predict(Xva)


def run_splits(df, y, splits, lgb_rounds=400, lgb_leaves=31, ridge_alpha=12.0):
    src_levels = sorted(df["source"].astype(str).unique()) + ["__UNK__"]
    src_levels_fit = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    n = len(y)
    arms = {k: np.zeros(n) for k in ["spline_car", "table8", "power", "te_ridge", "joint", "lgb"]}
    neigh = NEIGH[2]  # 8neigh
    per_fold = []

    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        yva = y[va_i]
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        km_tr, km_va, dtr, dva, ctr, cva = key_maps(trn, val, cond_tr, cond_va, rk_tr, rk_va)

        # --- spline + car ---
        dk, gk = knots_for(days_tr, rk_tr)
        large = [s for s in src_levels_fit if pd_count(src_tr).get(s, 0) >= LARGE_N]
        extra_tr, _ = car_terms(src_tr, rk_tr)
        extra_va, _ = car_terms(src_va, rk_va)
        Sptr = spline_design(
            days_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"], dk, gk, src_levels_fit, reg_levels, large, extra_tr
        )
        Spva = spline_design(
            days_va, rk_va, src_va, val["region"].astype(str), val["age_range"], dk, gk, src_levels_fit, reg_levels, large, extra_va
        )
        Xs, mu, sd = standardize_fit(Sptr)
        m = Ridge(alpha=ridge_alpha)
        m.fit(Xs, ytr)
        arms["spline_car"][va_i] = m.predict(standardize_apply(Spva, mu, sd))

        # --- 2D table ---
        t_tr, t_va = table_scores(src_tr, src_va, dtr, dva, ctr, cva, ytr, src_levels, neigh, m=10.0, alpha_glob=0.25)
        arms["table8"][va_i] = t_va

        # --- power closed form ---
        bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
        sc_tr = power_score(days_tr, cond_tr, rk_tr, src_tr, 1.5, 1.227, bmap) / 1e5
        sc_va = power_score(days_va, cond_va, rk_va, src_va, 1.5, 1.227, bmap) / 1e5
        Wtr, _ = window_matrix(days_tr)
        Wva, _ = window_matrix(days_va)
        age8_tr = (trn["age_range"].to_numpy(float) >= 8).astype(float)[:, None]
        age8_va = (val["age_range"].to_numpy(float) >= 8).astype(float)[:, None]
        Ptr = np.hstack(
            [sc_tr[:, None], ohe_levels_apply(src_tr, src_levels_fit), ohe_levels_apply(trn["region"].astype(str), reg_levels), age8_tr, Wtr]
        )
        Pva = np.hstack(
            [sc_va[:, None], ohe_levels_apply(src_va, src_levels_fit), ohe_levels_apply(val["region"].astype(str), reg_levels), age8_va, Wva]
        )
        Xp, mup, sdp = standardize_fit(Ptr)
        mp = Ridge(alpha=12.0)
        mp.fit(Xp, ytr)
        arms["power"][va_i] = mp.predict(standardize_apply(Pva, mup, sdp))

        # --- TE ridge ---
        te_tr, te_va = [], []
        named_va = {}
        for name, mm in TE_KEYS:
            a, b, _, _ = te_pair(km_tr[name], km_va[name], ytr, m=mm)
            te_tr.append(a)
            te_va.append(b)
            named_va[name] = b
        TEtr = np.column_stack(te_tr)
        TEva = np.column_stack(te_va)
        mt = Ridge(alpha=20.0)
        mt.fit(TEtr, ytr)
        arms["te_ridge"][va_i] = mt.predict(TEva)

        # --- joint identity Ridge: spline+car + table + power + a few TE ---
        te_keep = ["src_cq_dq", "cq_dq", "src_rq", "reg_dq", "src_dq"]
        te_idx = [i for i, (n, _) in enumerate(TE_KEYS) if n in te_keep]
        Jtr = np.hstack([Sptr, t_tr[:, None], sc_tr[:, None], TEtr[:, te_idx]])
        Jva = np.hstack([Spva, t_va[:, None], sc_va[:, None], TEva[:, te_idx]])
        # weak continuous residuals (not id)
        for col in ("x20", "V"):
            Jtr = np.hstack([Jtr, trn[col].to_numpy(float)[:, None]])
            Jva = np.hstack([Jva, val[col].to_numpy(float)[:, None]])
        Xj, muj, sdj = standardize_fit(Jtr)
        mj = Ridge(alpha=25.0)
        mj.fit(Xj, ytr)
        arms["joint"][va_i] = mj.predict(standardize_apply(Jva, muj, sdj))

        # --- LGB fixed rounds, medium cats, NO early stopping ---
        if HAS_LGB:
            src_codes = {s: i for i, s in enumerate(src_levels_fit)}
            reg_codes = {s: i for i, s in enumerate(reg_levels)}
            num_tr = np.column_stack(
                [
                    days_tr,
                    np.log1p(days_tr),
                    cond_tr,
                    rk_tr,
                    sc_tr,
                    t_tr,
                    extra_tr,
                    trn["age_range"].to_numpy(float),
                    (trn["age_range"].to_numpy() >= 8).astype(float),
                    trn["x20"].to_numpy(float),
                    trn["V"].to_numpy(float),
                    Wtr,
                ]
            )
            num_va = np.column_stack(
                [
                    days_va,
                    np.log1p(days_va),
                    cond_va,
                    rk_va,
                    sc_va,
                    t_va,
                    extra_va,
                    val["age_range"].to_numpy(float),
                    (val["age_range"].to_numpy() >= 8).astype(float),
                    val["x20"].to_numpy(float),
                    val["V"].to_numpy(float),
                    Wva,
                ]
            )
            # medium-card integer cats — NOT src_cq_dq
            def codes_from_train(tr_keys, va_keys):
                uniq = {k: i for i, k in enumerate(pd.Index(tr_keys).unique())}
                return pd.Series(va_keys).map(uniq).fillna(-1).to_numpy(dtype=float)

            cat_tr = np.column_stack(
                [
                    pd.Series(src_tr).map(src_codes).fillna(-1).to_numpy(),
                    pd.Series(trn["region"].astype(str)).map(reg_codes).fillna(-1).to_numpy(),
                    trn["age_range"].to_numpy(int),
                    pd.Series(km_tr["cq"]).astype(int).to_numpy(),
                    pd.Series(km_tr["dq"]).astype(int).to_numpy(),
                    codes_from_train(km_tr["src_cq"], km_tr["src_cq"]),
                    codes_from_train(km_tr["src_dq"], km_tr["src_dq"]),
                    codes_from_train(km_tr["src_reg"], km_tr["src_reg"]),
                ]
            )
            cat_va = np.column_stack(
                [
                    pd.Series(src_va).map(src_codes).fillna(-1).to_numpy(),
                    pd.Series(val["region"].astype(str)).map(reg_codes).fillna(-1).to_numpy(),
                    val["age_range"].to_numpy(int),
                    pd.Series(km_va["cq"]).astype(int).to_numpy(),
                    pd.Series(km_va["dq"]).astype(int).to_numpy(),
                    codes_from_train(km_tr["src_cq"], km_va["src_cq"]),
                    codes_from_train(km_tr["src_dq"], km_va["src_dq"]),
                    codes_from_train(km_tr["src_reg"], km_va["src_reg"]),
                ]
            )
            nnum = num_tr.shape[1]
            Htr = np.hstack([num_tr, cat_tr])
            Hva = np.hstack([num_va, cat_va])
            cat_idx = list(range(nnum, nnum + cat_tr.shape[1]))
            arms["lgb"][va_i] = lgb_predict(Htr, ytr, Hva, cat_idx, lgb_rounds, lgb_leaves, seed=2026 + fold)
        else:
            arms["lgb"][va_i] = arms["joint"][va_i]

        rec = {k: auc(yva, arms[k][va_i]) for k in arms}
        rec["fold"] = fold
        per_fold.append(rec)
        print(f"  fold {fold} " + " ".join(f"{k}={rec[k]:.4f}" for k in arms), flush=True)

    return arms, per_fold


def blend_grid(arms, y, weights_list):
    best = {"auc": -1.0}
    rows = []
    for name, w in weights_list:
        s = np.zeros(len(y))
        for k, wk in w.items():
            s = s + wk * rank01(arms[k])
        a = auc(y, s)
        rows.append({"name": name, "w": w, "auc": a})
        if a > best["auc"]:
            best = {"name": name, "w": w, "auc": a}
    return rows, best


def main():
    df, y = load_train()
    src_all = df["source"].astype(str).to_numpy()
    splits5 = skf_splits(y, 5)
    splits10 = skf_splits(y, 10)

    print("=== 5-fold screen ===", flush=True)
    screen_lgb = []
    best_lgb = {"auc": -1.0, "rounds": 400, "leaves": 31}
    if HAS_LGB:
        for rounds, leaves in [(400, 31), (600, 31)]:
            arms, _ = run_splits(df, y, splits5, lgb_rounds=rounds, lgb_leaves=leaves)
            a = auc(y, arms["lgb"])
            rec = {"rounds": rounds, "leaves": leaves, "auc5_lgb": a, "auc5_joint": auc(y, arms["joint"])}
            screen_lgb.append(rec)
            print(f"screen LGB {rec}", flush=True)
            if a > best_lgb["auc"]:
                best_lgb = {**rec, "auc": a}
        # reuse last arms for blend screen (from best config if we re-run; use last for speed if last is best)
        if (best_lgb["rounds"], best_lgb["leaves"]) != (rounds, leaves):
            arms, _ = run_splits(df, y, splits5, lgb_rounds=best_lgb["rounds"], lgb_leaves=best_lgb["leaves"])
    else:
        arms, _ = run_splits(df, y, splits5)

    wlist = [
        ("eq_closed", {"spline_car": 0.25, "table8": 0.25, "power": 0.25, "te_ridge": 0.25}),
        ("eq5", {"spline_car": 0.2, "table8": 0.2, "power": 0.2, "te_ridge": 0.2, "joint": 0.2}),
        ("table_ridge", {"table8": 0.45, "spline_car": 0.55}),
        ("table_te", {"table8": 0.4, "te_ridge": 0.6}),
        ("joint_table", {"joint": 0.7, "table8": 0.3}),
        ("closed_plus", {"joint": 0.5, "table8": 0.2, "te_ridge": 0.3}),
        ("w62_style", {"joint": 0.62, "te_ridge": 0.38}),
        ("lgb_joint", {"lgb": 0.62, "joint": 0.38}),
        ("lgb_closed", {"lgb": 0.55, "joint": 0.25, "table8": 0.10, "te_ridge": 0.10}),
        ("lgb_eq", {"lgb": 0.34, "joint": 0.22, "te_ridge": 0.22, "table8": 0.11, "spline_car": 0.11}),
        ("all6", {k: 1 / 6 for k in ["spline_car", "table8", "power", "te_ridge", "joint", "lgb"]}),
    ]
    blend5, best_blend = blend_grid(arms, y, wlist)
    print("5fold blends", sorted(blend5, key=lambda z: -z["auc"])[:6], flush=True)

    print("=== 10-fold report ===", flush=True)
    arms10, pf = run_splits(df, y, splits10, lgb_rounds=best_lgb["rounds"], lgb_leaves=best_lgb["leaves"])
    arm_aucs = {k: auc(y, v) for k, v in arms10.items()}
    print("10fold arms", arm_aucs, flush=True)

    blend10, _ = blend_grid(arms10, y, wlist)
    # frozen winner from 5-fold
    w_star = best_blend["w"]
    fused = np.zeros(len(y))
    for k, wk in w_star.items():
        fused = fused + wk * rank01(arms10[k])
    # leak-free equal rank of closed-form arms (no LGB, no weight tune)
    closed_eq = 0.25 * (rank01(arms10["spline_car"]) + rank01(arms10["table8"]) + rank01(arms10["power"]) + rank01(arms10["te_ridge"]))
    # table + ridge rank (the required fusion)
    tab_ridge = 0.45 * rank01(arms10["table8"]) + 0.55 * rank01(arms10["spline_car"])
    tab_joint = 0.35 * rank01(arms10["table8"]) + 0.65 * rank01(arms10["joint"])

    save_oof("exp5_joint_oof.npy", arms10["joint"])
    save_oof("exp5_lgb_oof.npy", arms10["lgb"])
    save_oof("exp5_fuse_oof.npy", fused)
    save_oof("exp5_table_ridge_oof.npy", tab_ridge)

    report = {
        "protocol": "5-fold screen / 10-fold report. Fold-internal fit. LGB fixed rounds, no ES. Rank fusion weights frozen from 5-fold.",
        "screen_5fold_lgb": screen_lgb,
        "best_lgb_screen": best_lgb,
        "blend_5fold": sorted(blend5, key=lambda z: -z["auc"]),
        "best_blend_5fold": best_blend,
        "arms_10fold": arm_aucs,
        "blend_10fold": sorted(blend10, key=lambda z: -z["auc"]),
        "frozen_blend_10fold": {"w": w_star, "auc": auc(y, fused)},
        "honest_equal_closed_rank_10fold": auc(y, closed_eq),
        "table_plus_ridge_rank_10fold": auc(y, tab_ridge),
        "table_plus_joint_rank_10fold": auc(y, tab_joint),
        "joint_10fold": fold_report(y, arms10["joint"], src_all, [{"fold": r["fold"], "auc": r["joint"]} for r in pf]),
        "lgb_10fold": fold_report(y, arms10["lgb"], src_all, [{"fold": r["fold"], "auc": r["lgb"]} for r in pf]),
        "fuse_10fold": fold_report(y, fused, src_all, []),
    }
    dump_json("exp5_fuse.json", report)
    print("WROTE exp5_fuse.json")
    print("joint", arm_aucs["joint"], "lgb", arm_aucs["lgb"], "frozen", report["frozen_blend_10fold"]["auc"])
    print("equal_closed", report["honest_equal_closed_rank_10fold"], "table+ridge", report["table_plus_ridge_rank_10fold"])


if __name__ == "__main__":
    main()
