#!/usr/bin/env python3
"""Exp3: explicit insurance frequency models, RMSE fit (not logloss).

Multiplicative:
  p = clip( a_s * f(days) * g_s(condition) + b_region + c*age + d·windows , 0, 1)
Additive:
  p = clip( f_s(days) + g_s(condition) + h(region) + ... , 0, 1)

Also a linearized GLM/Ridge expansion (Spark-portable).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from common import (
    auc,
    dump_json,
    fill_condition,
    load_train,
    per_source_rank,
    save_oof,
    skf,
    windows_of,
    cond_r_of,
    source_median_map,
)


def additive_design(days, cond, rk, src, region, age, src_levels, reg_levels):
    d = np.asarray(days, float)
    c = np.clip(np.asarray(cond, float), 1e-6, None)
    r = np.asarray(rk, float)
    age = np.asarray(age, float)
    dn = d / 5000.0
    w = windows_of(d)
    # global smooth basis
    base = [
        dn,
        dn**2,
        np.log1p(d),
        np.sqrt(np.clip(d, 0, None)) / 80.0,
        r,
        (r - 0.5) ** 2,
        np.log(c),
        1.0 / np.sqrt(c),
        dn / np.sqrt(c),
        dn * (1.0 - r),
        age / 10.0,
        (age >= 8).astype(float),
        w["w_new50"],
        w["w_safe750"],
        w["w_safe1750"],
        w["w_hot1950"],
        (c < 0.05).astype(float),
    ]
    X = [np.column_stack(base)]
    src = np.asarray(src).astype(str)
    region = np.asarray(region).astype(str)
    # source intercepts + source × key shapes (this is f_s and g_s)
    for s in src_levels:
        ind = (src == s).astype(float)
        X.append(
            np.column_stack(
                [
                    ind,
                    ind * dn,
                    ind * np.log1p(d),
                    ind * r,
                    ind * (r - 0.5) ** 2,
                    ind * (dn / np.sqrt(c)),
                    ind * (dn * (1.0 - r)),
                    ind * w["w_safe750"],
                ]
            )
        )
    for rg in reg_levels:
        X.append((region == rg).astype(float)[:, None])
    return np.hstack(X)


def multiplicative_als(days, cond, rk, src, region, age, y, src_levels, reg_levels, n_iter=12):
    """ALS: p = a_s * f(days) * g_s(rk) + b_r + c*age + d·windows.

    f is 4-basis linear, g_s is quadratic in rk, fit by alternating Ridge.
    """
    d = np.asarray(days, float)
    r = np.asarray(rk, float)
    src = np.asarray(src).astype(str)
    region = np.asarray(region).astype(str)
    age = np.asarray(age, float)
    y = np.asarray(y, float)
    w = windows_of(d)
    F = np.column_stack([np.ones(len(d)), d / 5000.0, np.log1p(d), (d / 5000.0) ** 2])  # f basis
    G = np.column_stack([np.ones(len(d)), r, (r - 0.5) ** 2])  # g basis
    add = np.column_stack(
        [
            age / 10.0,
            (age >= 8).astype(float),
            w["w_new50"],
            w["w_safe750"],
            w["w_safe1750"],
            w["w_hot1950"],
        ]
    )
    # init
    a = {s: 1.0 for s in src_levels}
    bf = np.array([0.08, 0.04, 0.0, 0.0])  # f coeffs
    bg = {s: np.array([1.0, 0.0, 0.0]) for s in src_levels}
    br = {rg: 0.0 for rg in reg_levels}
    cad = np.zeros(add.shape[1])

    def f_val():
        return np.clip(F @ bf, 1e-4, 5.0)

    def g_val():
        out = np.ones(len(d))
        for s in src_levels:
            m = src == s
            out[m] = np.clip(G[m] @ bg[s], 1e-4, 5.0)
        return out

    def a_val():
        out = np.ones(len(d))
        for s in src_levels:
            out[src == s] = a[s]
        return out

    def b_val():
        out = np.zeros(len(d))
        for rg in reg_levels:
            out[region == rg] = br[rg]
        return out

    for _ in range(n_iter):
        fv, gv, av = f_val(), g_val(), a_val()
        # fit additive rest + scale: y ~ av*fv*gv * 1 + region + add  via ridge on residual structure
        # 1) update a_s given others: y - b - add ≈ a_s * (f*g)
        resid_target = y - b_val() - add @ cad
        prod = np.clip(fv * gv, 1e-6, None)
        for s in src_levels:
            m = src == s
            if m.sum() < 20:
                continue
            z = prod[m]
            # scalar least squares with ridge
            a[s] = float(np.clip(np.dot(z, resid_target[m]) / (np.dot(z, z) + 1e-2), -2, 5))
        av = a_val()
        # 2) update f coeffs: y-b-add ≈ (a*g) * (F @ bf)
        scale = np.clip(av * gv, 1e-6, None)
        Z = F * scale[:, None]
        bf = Ridge(alpha=1.0, fit_intercept=False).fit(Z, resid_target).coef_
        fv = f_val()
        # 3) update g_s
        for s in src_levels:
            m = src == s
            if m.sum() < 30:
                continue
            sc = np.clip(av[m] * fv[m], 1e-6, None)
            Z = G[m] * sc[:, None]
            bg[s] = Ridge(alpha=2.0, fit_intercept=False).fit(Z, resid_target[m]).coef_
        gv = g_val()
        av = a_val()
        # 4) update region + additive
        core = av * fv * gv
        rest = y - core
        # region means ridge
        for rg in reg_levels:
            m = region == rg
            if m.sum() < 10:
                br[rg] = 0.0
            else:
                # residual after add
                br[rg] = float(np.clip((rest[m] - add[m] @ cad).mean(), -0.2, 0.2))
        resid2 = rest - b_val()
        cad = Ridge(alpha=5.0, fit_intercept=False).fit(add, resid2).coef_

    fv, gv, av = f_val(), g_val(), a_val()
    p = av * fv * gv + b_val() + add @ cad
    params = {"a": a, "bf": bf.tolist(), "bg": {k: v.tolist() for k, v in bg.items()}, "br": br, "cad": cad.tolist()}
    return p, params


def apply_als(params, days, cond, rk, src, region, age, src_levels, reg_levels):
    d = np.asarray(days, float)
    r = np.asarray(rk, float)
    src = np.asarray(src).astype(str)
    region = np.asarray(region).astype(str)
    age = np.asarray(age, float)
    w = windows_of(d)
    F = np.column_stack([np.ones(len(d)), d / 5000.0, np.log1p(d), (d / 5000.0) ** 2])
    G = np.column_stack([np.ones(len(d)), r, (r - 0.5) ** 2])
    add = np.column_stack(
        [
            age / 10.0,
            (age >= 8).astype(float),
            w["w_new50"],
            w["w_safe750"],
            w["w_safe1750"],
            w["w_hot1950"],
        ]
    )
    fv = np.clip(F @ np.array(params["bf"]), 1e-4, 5.0)
    gv = np.ones(len(d))
    av = np.ones(len(d))
    for s in src_levels:
        m = src == s
        gv[m] = np.clip(G[m] @ np.array(params["bg"].get(s, [1.0, 0.0, 0.0])), 1e-4, 5.0)
        av[m] = params["a"].get(s, 1.0)
    bv = np.zeros(len(d))
    for rg in reg_levels:
        bv[region == rg] = params["br"].get(rg, 0.0)
    return av * fv * gv + bv + add @ np.array(params["cad"])


def main():
    df, y = load_train()
    src_levels = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    n = len(y)
    oof_add = np.zeros(n)
    oof_mul = np.zeros(n)
    oof_lpm = np.zeros(n)

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        age_tr, age_va = trn["age_range"].to_numpy(float), val["age_range"].to_numpy(float)
        reg_tr, reg_va = trn["region"].astype(str).to_numpy(), val["region"].astype(str).to_numpy()

        Xtr = additive_design(days_tr, cond_tr, rk_tr, src_tr, reg_tr, age_tr, src_levels, reg_levels)
        Xva = additive_design(days_va, cond_va, rk_va, src_va, reg_va, age_va, src_levels, reg_levels)
        ridge = Ridge(alpha=8.0)
        ridge.fit(Xtr, ytr)
        oof_add[va_i] = np.clip(ridge.predict(Xva), 0, 1)

        # linearized LPM = additive ridge (same); keep separate alpha
        ridge2 = Ridge(alpha=2.0)
        ridge2.fit(Xtr, ytr)
        oof_lpm[va_i] = ridge2.predict(Xva)

        p_tr, params = multiplicative_als(
            days_tr, cond_tr, rk_tr, src_tr, reg_tr, age_tr, ytr, src_levels, reg_levels
        )
        oof_mul[va_i] = apply_als(params, days_va, cond_va, rk_va, src_va, reg_va, age_va, src_levels, reg_levels)
        print(
            f"fold {fold} add={auc(y[va_i], oof_add[va_i]):.4f} "
            f"lpm={auc(y[va_i], oof_lpm[va_i]):.4f} mul={auc(y[va_i], oof_mul[va_i]):.4f}",
            flush=True,
        )

    report = {
        "additive_ridge": auc(y, oof_add),
        "lpm_ridge": auc(y, oof_lpm),
        "multiplicative_als": auc(y, oof_mul),
        "formula_add": "p = clip( f_s(days)+g_s(cond_rk)+h(region)+c*age+d*windows , 0,1) via Ridge RMSE",
        "formula_mul": "p = clip( a_s * f(days) * g_s(cond_rk) + b_region + c*age + d*windows , 0,1) ALS RMSE",
    }
    print("\n=== EXP3 ===")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:.5f}")
        else:
            print(f"  {k}: {v}")
    dump_json("exp3_freq.json", report)
    save_oof("exp3_oof_add.npy", oof_add)
    save_oof("exp3_oof_mul.npy", oof_mul)
    save_oof("exp3_oof_lpm.npy", oof_lpm)
    print("WROTE exp3_freq.json")


if __name__ == "__main__":
    main()
