#!/usr/bin/env python3
"""Exp7: power grid days^a / condition^b and days^a * (1-rank)^c. No PySR -> manual grid."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from common import both_auc, dump_json, fill_condition, load_train, per_source_rank, skf, auc, source_median_map, cond_r_of

AS = np.round(np.linspace(0.3, 1.6, 14), 3)
BS = np.round(np.linspace(0.1, 1.5, 15), 3)
CS = np.round(np.linspace(0.3, 2.0, 12), 3)


def oof_isotonic(x, y, splits):
    oof = np.zeros(len(y))
    for tr, va in splits:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x[tr], y[tr])
        oof[va] = iso.predict(x[va])
    return auc(y, oof)


def main():
    df, y = load_train()
    # For the grid, using full-data both-AUC is slightly optimistic vs OOF isotonic,
    # so we report BOTH raw both-AUC (ranking of the transform) AND 5-fold isotonic OOF for top cells.
    cond = df["condition"].fillna(df.groupby("source")["condition"].transform("median"))
    cond = cond.fillna(df["condition"].median()).to_numpy()
    days = df["days"].to_numpy(float)
    src = df["source"].astype(str).to_numpy()
    rk = per_source_rank(src, cond, src, cond)
    med = source_median_map(src, cond)
    cr = cond_r_of(src, cond, med)
    cond_c = np.clip(cond, 1e-6, None)

    grid_div = []
    best_div = {"auc": -1}
    for a in AS:
        da = np.power(np.clip(days, 1e-6, None), a)
        for b in BS:
            s = da / np.power(cond_c, b)
            u = both_auc(y, s)
            rec = {"a": float(a), "b": float(b), "both_auc": u, "form": "days^a / cond^b"}
            grid_div.append(rec)
            if u > best_div["auc"]:
                best_div = {"auc": u, **rec}

    grid_rank = []
    best_rk = {"auc": -1}
    for a in AS:
        da = np.power(np.clip(days, 1e-6, None), a)
        for c in CS:
            s = da * np.power(np.clip(1.0 - rk, 1e-6, None), c)
            u = both_auc(y, s)
            rec = {"a": float(a), "c": float(c), "both_auc": u, "form": "days^a * (1-rk)^c"}
            grid_rank.append(rec)
            if u > best_rk["auc"]:
                best_rk = {"auc": u, **rec}

    # cond_r in denominator
    grid_cr = []
    best_cr = {"auc": -1}
    for a in AS:
        da = np.power(np.clip(days, 1e-6, None), a)
        for b in BS:
            s = da / np.power(np.clip(cr, 1e-6, None), b)
            u = both_auc(y, s)
            rec = {"a": float(a), "b": float(b), "both_auc": u, "form": "days^a / cond_r^b"}
            grid_cr.append(rec)
            if u > best_cr["auc"]:
                best_cr = {"auc": u, **rec}

    grid_div.sort(key=lambda z: -z["both_auc"])
    grid_rank.sort(key=lambda z: -z["both_auc"])
    grid_cr.sort(key=lambda z: -z["both_auc"])

    # per-source best b for days / cond^b (a=1)
    per_src = {}
    for s in sorted(np.unique(src)):
        m = src == s
        ys = y[m]
        best = {"auc": -1}
        for b in BS:
            sc = days[m] / np.power(cond_c[m], b)
            u = both_auc(ys, sc)
            if u > best["auc"]:
                best = {"auc": u, "b": float(b)}
        # also u-shape vs monotone
        u_mono = both_auc(ys, -cond[m])
        u_u = both_auc(ys, (rk[m] - 0.5) ** 2)
        per_src[str(s)] = {
            "n": int(m.sum()),
            "best_b": best["b"],
            "best_days_over_condb": best["auc"],
            "cond_mono": u_mono,
            "u_shape": u_u,
            "days": both_auc(ys, days[m]),
        }

    # Honest 5-fold isotonic on the winning transforms
    splits = list(skf(y, n=5, seed=2026))

    def make_div(a, b):
        return np.power(np.clip(days, 1e-6, None), a) / np.power(cond_c, b)

    tops = {
        "best_div_raw": best_div,
        "best_rank_raw": best_rk,
        "best_cr_raw": best_cr,
        "iso_oof_best_div": oof_isotonic(make_div(best_div["a"], best_div["b"]), y, splits),
        "iso_oof_sqrt": oof_isotonic(days / np.sqrt(cond_c), y, splits),
        "iso_oof_ratio": oof_isotonic(days / np.clip(cr, 1e-6, None), y, splits),
        "iso_oof_rate": oof_isotonic(days * (1.0 - rk), y, splits),
        "iso_oof_best_rank": oof_isotonic(
            np.power(np.clip(days, 1e-6, None), best_rk["a"]) * np.power(np.clip(1.0 - rk, 1e-6, None), best_rk["c"]),
            y,
            splits,
        ),
        "iso_oof_best_cr": oof_isotonic(
            np.power(np.clip(days, 1e-6, None), best_cr["a"]) / np.power(np.clip(cr, 1e-6, None), best_cr["b"]),
            y,
            splits,
        ),
    }

    report = {
        **tops,
        "top10_div": grid_div[:10],
        "top10_rank": grid_rank[:10],
        "top10_cr": grid_cr[:10],
        "per_source": per_src,
        "note": "Raw both-AUC is a monotone ranking of the transform (no fit). iso_oof is honest 5-fold isotonic.",
    }
    print("=== EXP7 ===")
    print("best days^a/cond^b", best_div)
    print("best days^a*(1-rk)^c", best_rk)
    print("best days^a/cond_r^b", best_cr)
    print("iso oofs", {k: v for k, v in tops.items() if k.startswith("iso")})
    print("per-source best b:")
    for s, r in per_src.items():
        print(f"  {s:18s} b={r['best_b']:.2f} auc={r['best_days_over_condb']:.3f} mono={r['cond_mono']:.3f} U={r['u_shape']:.3f}")
    dump_json("exp7_power.json", report)
    print("WROTE exp7_power.json")


if __name__ == "__main__":
    main()
