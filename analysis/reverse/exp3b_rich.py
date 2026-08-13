#!/usr/bin/env python3
"""Richer probability-scale additive model: source-specific powers, src×region, confirmed windows."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import auc, dump_json, fill_condition, load_train, per_source_rank, save_oof, skf, windows_of


def source_b_map(src, days, cond, y, bs=np.round(np.linspace(0.1, 1.5, 8), 2)):
    """Pick per-source exponent b maximizing |AUC| of days/cond^b on TRAIN only."""
    from sklearn.metrics import roc_auc_score

    out = {}
    cond = np.clip(cond, 1e-6, None)
    for s in np.unique(src):
        m = src == s
        if m.sum() < 40:
            out[s] = 0.5
            continue
        best_b, best_a = 0.5, 0.5
        ys = y[m]
        for b in bs:
            sc = days[m] / np.power(cond[m], b)
            try:
                a = max(roc_auc_score(ys, sc), roc_auc_score(ys, -sc))
            except Exception:
                a = 0.5
            if a > best_a:
                best_a, best_b = a, float(b)
        out[s] = best_b
    return out


def design(days, cond, rk, src, region, age, liv, x20, src_levels, reg_levels, bmap):
    d = np.asarray(days, float)
    c = np.clip(np.asarray(cond, float), 1e-6, None)
    r = np.asarray(rk, float)
    src = np.asarray(src).astype(str)
    region = np.asarray(region).astype(str)
    age = np.asarray(age, float)
    dn = d / 5000.0
    w = windows_of(d)
    # source-specific power ratio
    bvec = np.array([bmap.get(s, 0.5) for s in src])
    pow_ratio = d / np.power(c, bvec)
    pow_ratio = pow_ratio / 8000.0
    cols = [
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
        pow_ratio,
        np.power(np.clip(d, 1, None), 1.5) * np.power(np.clip(1 - r, 1e-6, None), 1.2) / 2e5,
        age / 10.0,
        (age >= 8).astype(float),
        w["w_new50"],
        w["w_new200"],
        w["w_safe750"],
        w["w_safe1750"],
        w["w_hot1950"],
        (c < 0.05).astype(float),
        np.asarray(liv, float),
        np.asarray(x20, float),
    ]
    X = [np.column_stack(cols)]
    car = pd.Series(src).str.split("|").str[0].to_numpy()
    X.append(((car == "CAR_10").astype(float) * (r - 0.5) ** 2)[:, None])
    X.append(((car == "CAR_1").astype(float) * (1.0 - r))[:, None])
    X.append(((car == "CAR_7").astype(float) * r)[:, None])
    X.append(((car == "CAR_5").astype(float) * dn)[:, None])
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
                    ind * pow_ratio,
                    ind * w["w_safe750"],
                    ind * (age >= 8).astype(float),
                ]
            )
        )
    for rg in reg_levels:
        X.append((region == rg).astype(float)[:, None])
    # src x region is 220 cols — keep only large sources
    large = [s for s in src_levels if s.startswith("CAR_0") or s.startswith("CAR_1|") or s.startswith("CAR_2")]
    for s in large:
        for rg in reg_levels:
            X.append(((src == s) & (region == rg)).astype(float)[:, None])
    return np.hstack(X)


def main():
    df, y = load_train()
    src_levels = sorted(df["source"].astype(str).unique())
    reg_levels = sorted(df["region"].astype(str).unique())
    n = len(y)
    oof = np.zeros(n)
    oof2 = np.zeros(n)
    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
        Xtr = design(
            days_tr, cond_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"], trn["livability"], trn["x20"], src_levels, reg_levels, bmap
        )
        Xva = design(
            days_va, cond_va, rk_va, src_va, val["region"].astype(str), val["age_range"], val["livability"], val["x20"], src_levels, reg_levels, bmap
        )
        m = Ridge(alpha=12.0)
        m.fit(Xtr, ytr)
        oof[va_i] = m.predict(Xva)
        m2 = Ridge(alpha=3.0)
        m2.fit(Xtr, ytr)
        oof2[va_i] = m2.predict(Xva)
        print(f"fold {fold} a12={auc(y[va_i], oof[va_i]):.4f} a3={auc(y[va_i], oof2[va_i]):.4f} bmap={ {k.split('|')[0]:v for k,v in bmap.items()} }", flush=True)
    report = {"ridge_a12": auc(y, oof), "ridge_a3": auc(y, oof2)}
    print("=== EXP3b RICH ADDITIVE ===", report)
    dump_json("exp3b_rich.json", report)
    save_oof("exp3b_oof.npy", oof if report["ridge_a12"] >= report["ridge_a3"] else oof2)
    print("WROTE exp3b_rich.json")


if __name__ == "__main__":
    main()
