#!/usr/bin/env python3
"""Exp2: 2D histogram P = smoothed mean(y | source, days_bin, cond_bin). Honest OOF."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import auc, dump_json, fill_condition, load_train, qcut_apply, save_oof, skf, te_val_only

BINS = [5, 8, 10, 15]
MS = [5, 10, 20, 40, 80]


def keys(src, db, cb):
    return np.asarray(src).astype(str) + "|" + np.asarray(db).astype(str) + "|" + np.asarray(cb).astype(str)


def main():
    df, y = load_train()
    results = []
    best = {"auc": -1, "name": None}
    oofs = {}

    # also try equal-width days bins (generating process may be on raw days, not quantiles)
    configs = []
    for nd in BINS:
        for nc in BINS:
            for m in MS:
                configs.append(("qcut", nd, nc, m))
    for nd in [8, 10, 20, 30]:
        for nc in [5, 8, 10]:
            for m in [10, 20, 40]:
                configs.append(("width", nd, nc, m))

    n = len(y)
    # Pre-allocate only for a subset we keep; compute all AUCs, save best few oofs
    keep_names = set()

    # First pass: compute all, store oof in dict only for promising or representative
    for kind, nd, nc, m in configs:
        oof = np.zeros(n)
        name = f"{kind}_d{nd}_c{nc}_m{m}"
        for tr_i, va_i in skf(y):
            trn, val = df.iloc[tr_i], df.iloc[va_i]
            ytr = y[tr_i]
            cond_tr, cond_va = fill_condition(trn, val)
            src_tr = trn["source"].astype(str).to_numpy()
            src_va = val["source"].astype(str).to_numpy()
            if kind == "qcut":
                dtr, dva, _ = qcut_apply(trn["days"], val["days"], nd)
                ctr, cva, _ = qcut_apply(cond_tr, cond_va, nc)
            else:
                dmin, dmax = float(trn["days"].min()), float(trn["days"].max())
                edges = np.linspace(dmin, dmax, nd + 1)
                edges[0], edges[-1] = -np.inf, np.inf
                dtr = pd.cut(trn["days"], edges, labels=False, include_lowest=True).fillna(0).astype(int).to_numpy()
                dva = pd.cut(val["days"], edges, labels=False, include_lowest=True).fillna(0).astype(int).to_numpy()
                ctr, cva, _ = qcut_apply(cond_tr, cond_va, nc)
            oof[va_i] = te_val_only(keys(src_tr, dtr, ctr), keys(src_va, dva, cva), ytr, m=m)
        a = auc(y, oof)
        rec = {"name": name, "kind": kind, "nd": nd, "nc": nc, "m": m, "auc": a}
        results.append(rec)
        if a > best["auc"]:
            best = {"auc": a, "name": name, "nd": nd, "nc": nc, "m": m, "kind": kind}
            oofs["best"] = oof.copy()
        print(f"{name:22s} {a:.5f}", flush=True)

    results.sort(key=lambda r: -r["auc"])
    print("\n=== EXP2 TOP 15 ===")
    for r in results[:15]:
        print(f"  {r['name']:22s} {r['auc']:.5f}")

    # extra: without source; with region instead of source; 3D src|reg_q|...
    extras = {}
    for tag, keyfun in [
        ("nosrc_q10_q10_m20", lambda src, d, c, reg: np.asarray(d).astype(str) + "|" + np.asarray(c).astype(str)),
        (
            "reg_q10_q10_m20",
            lambda src, d, c, reg: np.asarray(reg).astype(str) + "|" + np.asarray(d).astype(str) + "|" + np.asarray(c).astype(str),
        ),
        (
            "src_reg_dq8_cq8_m40",
            lambda src, d, c, reg: np.asarray(src).astype(str)
            + "|"
            + np.asarray(reg).astype(str)
            + "|"
            + np.asarray(d).astype(str)
            + "|"
            + np.asarray(c).astype(str),
        ),
    ]:
        oof = np.zeros(n)
        for tr_i, va_i in skf(y):
            trn, val = df.iloc[tr_i], df.iloc[va_i]
            cond_tr, cond_va = fill_condition(trn, val)
            nd, nc, m = (8, 8, 40) if "dq8" in tag else (10, 10, 20)
            dtr, dva, _ = qcut_apply(trn["days"], val["days"], nd)
            ctr, cva, _ = qcut_apply(cond_tr, cond_va, nc)
            oof[va_i] = te_val_only(
                keyfun(trn["source"], dtr, ctr, trn["region"]),
                keyfun(val["source"], dva, cva, val["region"]),
                y[tr_i],
                m=m,
            )
        extras[tag] = auc(y, oof)
        oofs[tag] = oof
        print(f"extra {tag:24s} {extras[tag]:.5f}")

    report = {"best": best, "top": results[:20], "all": results, "extras": extras}
    dump_json("exp2_hist2d.json", report)
    save_oof("exp2_best_oof.npy", oofs["best"])
    print("BEST", best)
    print("WROTE exp2_hist2d.json")


if __name__ == "__main__":
    main()
