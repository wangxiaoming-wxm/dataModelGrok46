#!/usr/bin/env python3
"""Push portable fusion to honest OOF ≥ 0.68.

Drops the collapsing joint Ridge. Adds ordered TE, a richer TE key set,
8-neigh table as a TE column, dual-world LGB (fixed rounds, no ES),
and a 5-fold weight grid. 10-fold reports frozen weights plus equal-rank.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from arms import (
    fit_power_ridge,
    fit_spline_car,
    fit_table8,
    fit_te_ridge,
    fuse_linear,
    fuse_rank,
    lgb_dual,
    make_keys,
    rank01,
)
from common import auc, dump_json, fill_condition, fold_report, load_train, save_oof, skf_splits

try:
    import lightgbm as lgb  # noqa: F401

    HAS_LGB = True
except Exception:
    HAS_LGB = False


def collect(df, y, splits, lgb_rounds=400):
    src_levels = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    src_levels_tab = src_levels + ["__UNK__"]
    n = len(y)
    arms = {k: np.zeros(n) for k in ["spline", "table8", "te", "te_ord", "te_tab", "power", "lgb1", "lgb2", "lgb_max"]}
    per_fold = []
    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        yva = y[va_i]
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        from common import per_source_rank

        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        km_tr, km_va, dtr, dva, ctr, cva, _, _ = make_keys(trn, val, cond_tr, cond_va, rk_tr, rk_va)

        _, p_sp, _, _ = fit_spline_car(trn, val, ytr, src_levels, reg_levels, alpha=20.0)
        arms["spline"][va_i] = p_sp

        _, p_tab, _ = fit_table8(trn, val, ytr, cond_tr, cond_va, dtr, dva, ctr, cva, src_levels_tab, alpha_glob=0.4)
        arms["table8"][va_i] = p_tab

        _, p_te, _ = fit_te_ridge(km_tr, km_va, ytr, use_ordered=False, alpha=18.0, seed=fold)
        arms["te"][va_i] = p_te
        _, p_teo, _ = fit_te_ridge(km_tr, km_va, ytr, use_ordered=True, alpha=18.0, seed=fold * 13)
        arms["te_ord"][va_i] = p_teo
        t_tr, t_va, _ = fit_table8(trn, val, ytr, cond_tr, cond_va, dtr, dva, ctr, cva, src_levels_tab, alpha_glob=0.4)
        _, p_tet, _ = fit_te_ridge(km_tr, km_va, ytr, table_tr=t_tr, table_va=t_va, use_ordered=True, alpha=20.0, seed=fold * 19)
        arms["te_tab"][va_i] = p_tet

        _, p_pw, _ = fit_power_ridge(trn, val, ytr, cond_tr, cond_va, rk_tr, rk_va, src_levels, reg_levels)
        arms["power"][va_i] = p_pw

        if HAS_LGB:
            p1, p2 = lgb_dual(trn, val, ytr, cond_tr, cond_va, rk_tr, rk_va, km_tr, km_va, src_levels, reg_levels, rounds=lgb_rounds, seed=2026 + fold)
            arms["lgb1"][va_i] = p1
            arms["lgb2"][va_i] = p2
            arms["lgb_max"][va_i] = np.maximum(rank01(p1), rank01(p2))
        else:
            arms["lgb1"][va_i] = p_sp
            arms["lgb2"][va_i] = p_te
            arms["lgb_max"][va_i] = rank01(p_sp)

        rec = {k: auc(yva, arms[k][va_i]) for k in arms}
        rec["fold"] = fold
        per_fold.append(rec)
        print("  fold", fold, " ".join(f"{k}={rec[k]:.4f}" for k in ["spline", "table8", "te", "te_ord", "te_tab", "power", "lgb1", "lgb2"]), flush=True)
    return arms, per_fold


def weight_grid(arms, y):
    rows = []
    closed = [
        ("eq_stt", {"spline": 1, "table8": 1, "te_ord": 1}),
        ("eq_st", {"spline": 1, "table8": 1, "te": 1}),
        ("te_tab_only", {"te_tab": 1}),
        ("te_ord_only", {"te_ord": 1}),
        ("w_te55_tab25_sp20", {"te_ord": 0.55, "table8": 0.25, "spline": 0.20}),
        ("w_te50_tab30_sp20", {"te_ord": 0.50, "table8": 0.30, "spline": 0.20}),
        ("w_tet70_sp30", {"te_tab": 0.70, "spline": 0.30}),
        ("w_tet60_tab20_sp20", {"te_tab": 0.60, "table8": 0.20, "spline": 0.20}),
        ("closed_plus_pw", {"te_ord": 0.45, "table8": 0.25, "spline": 0.20, "power": 0.10}),
    ]
    gbt = [
        ("lgb_te", {"lgb1": 0.50, "te_ord": 0.50}),
        ("lgbmax_te", {"lgb_max": 0.50, "te_ord": 0.50}),
        ("w62_lgb_te", {"lgb1": 0.62, "te_ord": 0.38}),
        ("lgb_tet", {"lgb1": 0.45, "te_tab": 0.55}),
        ("lgb_closed", {"lgb1": 0.40, "te_ord": 0.30, "table8": 0.15, "spline": 0.15}),
        ("lgbmax_closed", {"lgb_max": 0.40, "te_ord": 0.30, "table8": 0.15, "spline": 0.15}),
        ("lgb_dual_te_tab", {"lgb1": 0.28, "lgb2": 0.20, "te_ord": 0.28, "table8": 0.12, "spline": 0.12}),
        ("lgb_dual_tet", {"lgb1": 0.30, "lgb2": 0.20, "te_tab": 0.35, "spline": 0.15}),
        ("all_good", {"lgb1": 0.22, "lgb2": 0.16, "te_ord": 0.22, "te_tab": 0.16, "table8": 0.12, "spline": 0.12}),
    ]
    # fine grid on closed triple
    for w_te in (0.40, 0.48, 0.55, 0.62):
        for w_tab in (0.15, 0.22, 0.30):
            w_sp = 1.0 - w_te - w_tab
            if w_sp < 0.08:
                continue
            closed.append((f"g_te{w_te:.2f}_tab{w_tab:.2f}", {"te_ord": w_te, "table8": w_tab, "spline": w_sp}))
    for w_lgb in (0.30, 0.38, 0.46, 0.54):
        rest = 1.0 - w_lgb
        gbt.append((f"g_lgb{w_lgb:.2f}_te", {"lgb1": w_lgb, "te_ord": rest * 0.55, "table8": rest * 0.25, "spline": rest * 0.20}))
        gbt.append((f"g_lgbmax{w_lgb:.2f}", {"lgb_max": w_lgb, "te_ord": rest * 0.55, "table8": rest * 0.25, "spline": rest * 0.20}))

    best_closed = {"auc": -1.0}
    best_all = {"auc": -1.0}
    for name, w in closed + gbt:
        pred = fuse_rank(arms, w)
        a = auc(y, pred)
        rec = {"name": name, "w": w, "auc": a, "family": "gbt" if name in {x[0] for x in gbt} or name.startswith("g_lgb") else "closed"}
        # fix family
        rec["family"] = "gbt" if any(k.startswith("lgb") for k in w) else "closed"
        rows.append(rec)
        if rec["family"] == "closed" and a > best_closed["auc"]:
            best_closed = rec
        if a > best_all["auc"]:
            best_all = rec
    rows.sort(key=lambda z: -z["auc"])
    return rows, best_closed, best_all


def main():
    df, y = load_train()
    src_all = df["source"].astype(str).to_numpy()
    splits5 = skf_splits(y, 5)
    splits10 = skf_splits(y, 10)

    print("=== 5-fold screen ===", flush=True)
    arms5, _ = collect(df, y, splits5, lgb_rounds=400)
    auc5 = {k: auc(y, v) for k, v in arms5.items()}
    print("5fold arms", auc5, flush=True)
    rows5, best_c5, best_a5 = weight_grid(arms5, y)
    print("best closed 5", best_c5, flush=True)
    print("best all 5", best_a5, flush=True)
    print("top8", rows5[:8], flush=True)

    print("=== 10-fold report ===", flush=True)
    arms10, pf = collect(df, y, splits10, lgb_rounds=400)
    auc10 = {k: auc(y, v) for k, v in arms10.items()}
    print("10fold arms", auc10, flush=True)
    rows10, best_c10, best_a10 = weight_grid(arms10, y)

    fused_closed = fuse_rank(arms10, best_c5["w"])
    fused_all = fuse_rank(arms10, best_a5["w"])
    fused_eq = fuse_rank(arms10, {"spline": 1, "table8": 1, "te_ord": 1})
    fused_lgb = fuse_rank(arms10, {"lgb1": 0.40, "te_ord": 0.30, "table8": 0.15, "spline": 0.15})
    fused_lin = fuse_linear(arms10, {"te_ord": 0.55, "table8": 0.25, "spline": 0.20})

    for name, arr in [
        ("exp6_spline_oof.npy", arms10["spline"]),
        ("exp6_table8_oof.npy", arms10["table8"]),
        ("exp6_te_ord_oof.npy", arms10["te_ord"]),
        ("exp6_te_tab_oof.npy", arms10["te_tab"]),
        ("exp6_lgb1_oof.npy", arms10["lgb1"]),
        ("exp6_closed_oof.npy", fused_closed),
        ("exp6_all_oof.npy", fused_all),
    ]:
        save_oof(name, arr)

    report = {
        "protocol": "5-fold screen / 10-fold report. Ordered TE fold-internal. LGB fixed 400 rounds, no ES.",
        "arms_5fold": auc5,
        "arms_10fold": auc10,
        "blend_5fold_top12": rows5[:12],
        "best_closed_5fold": best_c5,
        "best_all_5fold": best_a5,
        "blend_10fold_top12": rows10[:12],
        "frozen_closed_10fold": {"w": best_c5["w"], "auc": auc(y, fused_closed)},
        "frozen_all_10fold": {"w": best_a5["w"], "auc": auc(y, fused_all)},
        "honest_eq_rank_spline_table_teord": auc(y, fused_eq),
        "preset_lgb_closed": auc(y, fused_lgb),
        "linear_prob_te_tab_sp": auc(y, fused_lin),
        "te_ord_10fold": fold_report(y, arms10["te_ord"], src_all, [{"fold": r["fold"], "auc": r["te_ord"]} for r in pf]),
        "lgb1_10fold": fold_report(y, arms10["lgb1"], src_all, [{"fold": r["fold"], "auc": r["lgb1"]} for r in pf]),
        "closed_fuse_10fold": fold_report(y, fused_closed, src_all, []),
        "all_fuse_10fold": fold_report(y, fused_all, src_all, []),
    }
    dump_json("exp6_push.json", report)
    print("WROTE exp6_push.json")
    print("CLOSED frozen", report["frozen_closed_10fold"])
    print("ALL frozen", report["frozen_all_10fold"])
    print("eq rank", report["honest_eq_rank_spline_table_teord"], "lgb_closed", report["preset_lgb_closed"])


if __name__ == "__main__":
    main()
