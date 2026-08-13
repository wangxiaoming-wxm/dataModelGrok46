#!/usr/bin/env python3
"""Exp3: 2D generating surface source × days_q × cond_q.

Known: qcut days=5, cond=10, m=10 → honest OOF 0.644.
This run:
  - 8-neighborhood (and 4-neigh) count-weighted smoothing
  - leave-one-source CV to pick (nd, nc, m, neighborhood)
  - hierarchical blend of per-source table with global table
  - optional Nadaraya–Watson per-source kernel (bandwidth via LOSO)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import (
    auc,
    dump_json,
    fill_condition,
    fold_report,
    load_train,
    qcut_apply,
    save_oof,
    skf_splits,
    te_group,
)

# configs screened by LOSO then 5-fold
BIN_CFGS = [(5, 10), (8, 8), (5, 8), (8, 10)]
MS = [5.0, 10.0, 20.0, 40.0]
NEIGH = [
    {"name": "none", "w0": 1.0, "ws": 0.0, "wd": 0.0},
    {"name": "4neigh", "w0": 4.0, "ws": 1.0, "wd": 0.0},
    {"name": "8neigh", "w0": 4.0, "ws": 1.0, "wd": 0.5},
    {"name": "8equal", "w0": 1.0, "ws": 1.0, "wd": 1.0},
]


def _empty_grid(n_src, nd, nc):
    return np.zeros((n_src, nd, nc)), np.zeros((n_src, nd, nc))


def fill_grid(src_idx, dbin, cbin, y, n_src, nd, nc):
    sm, ct = _empty_grid(n_src, nd, nc)
    # clip bins
    dbin = np.clip(np.asarray(dbin, dtype=int), 0, nd - 1)
    cbin = np.clip(np.asarray(cbin, dtype=int), 0, nc - 1)
    src_idx = np.asarray(src_idx, dtype=int)
    y = np.asarray(y, dtype=float)
    for s, d, c, yi in zip(src_idx, dbin, cbin, y):
        sm[s, d, c] += yi
        ct[s, d, c] += 1.0
    return sm, ct


def smooth_grid(sm, ct, prior, m, w0, ws, wd):
    """Count-weighted 8-neighborhood + m-prior on the (src, days, cond) cube."""
    n_src, nd, nc = sm.shape
    num = np.zeros_like(sm)
    den = np.zeros_like(ct)
    # offsets: center, 4-side, 4-diag
    offs = [(0, 0, w0)]
    for di, dj, w in [(-1, 0, ws), (1, 0, ws), (0, -1, ws), (0, 1, ws)]:
        offs.append((di, dj, w))
    for di, dj, w in [(-1, -1, wd), (-1, 1, wd), (1, -1, wd), (1, 1, wd)]:
        offs.append((di, dj, w))
    for di, dj, w in offs:
        if w <= 0:
            continue
        for i in range(nd):
            ii = i + di
            if ii < 0 or ii >= nd:
                continue
            for j in range(nc):
                jj = j + dj
                if jj < 0 or jj >= nc:
                    continue
                num[:, i, j] += w * sm[:, ii, jj]
                den[:, i, j] += w * ct[:, ii, jj]
    return (num + prior * m) / np.clip(den + m, 1e-9, None)


def lookup(grid, src_idx, dbin, cbin):
    dbin = np.clip(np.asarray(dbin, dtype=int), 0, grid.shape[1] - 1)
    cbin = np.clip(np.asarray(cbin, dtype=int), 0, grid.shape[2] - 1)
    src_idx = np.clip(np.asarray(src_idx, dtype=int), 0, grid.shape[0] - 1)
    return grid[src_idx, dbin, cbin]


def src_index(src, levels):
    mp = {s: i for i, s in enumerate(levels)}
    return np.array([mp.get(s, -1) for s in np.asarray(src).astype(str)], dtype=int)


def hierarchical(src_grid, glob_grid, src_idx, alpha):
    """(1-alpha)*source + alpha*global, global is src_grid averaged over sources with counts."""
    # glob_grid shape (1 or n, nd, nc) — we pass a (nd,nc) array
    g = glob_grid[src_idx] if glob_grid.ndim == 3 else glob_grid[np.clip(src_idx, 0, 0)]
    # actually we'll pass glob as (nd,nc)
    if glob_grid.ndim == 2:
        g = glob_grid[np.clip(np.asarray(src_idx) * 0, 0, 0)]  # placeholder
        # vectorized:
        dbin_shape = src_grid.shape
        # caller should lookup first
        raise ValueError("lookup first")
    return (1 - alpha) * src_grid + alpha * glob_grid


def predict_table(src, dbin, cbin, src_levels, sm, ct, prior, m, neigh, alpha_glob=0.0, glob=None):
    idx = src_index(src, src_levels)
    # unknown source -> last extra slot if present, else 0
    n_src = sm.shape[0]
    idx = np.where(idx < 0, n_src - 1, idx)
    grid = smooth_grid(sm, ct, prior, m, neigh["w0"], neigh["ws"], neigh["wd"])
    p = lookup(grid, idx, dbin, cbin)
    if glob is not None and alpha_glob > 0:
        pg = lookup(glob[None, ...], np.zeros(len(src), dtype=int), dbin, cbin)
        p = (1.0 - alpha_glob) * p + alpha_glob * pg
    return p


def make_global(sm, ct, prior, m, neigh):
    smg = sm.sum(axis=0, keepdims=True)
    ctg = ct.sum(axis=0, keepdims=True)
    return smooth_grid(smg, ctg, prior, m, neigh["w0"], neigh["ws"], neigh["wd"])[0]


def loso_auc(df, y, nd, nc, m, neigh, alpha_glob=0.0):
    """Leave-one-source: table from other sources (GLOBAL surface) scored on held-out source."""
    src = df["source"].astype(str).to_numpy()
    scores = []
    weights = []
    cond = df["condition"].fillna(df.groupby("source")["condition"].transform("median"))
    cond = cond.fillna(df["condition"].median()).to_numpy(float)
    days = df["days"].to_numpy(float)
    for s in np.unique(src):
        va = src == s
        tr = ~va
        if va.sum() < 80 or np.unique(y[va]).size < 2:
            continue
        dtr, dva, _ = qcut_apply(days[tr], days[va], nd)
        ctr, cva, _ = qcut_apply(cond[tr], cond[va], nc)
        # one dummy source
        sm, ct = fill_grid(np.zeros(tr.sum(), dtype=int), dtr, ctr, y[tr], 1, nd, nc)
        prior = float(y[tr].mean())
        grid = smooth_grid(sm, ct, prior, m, neigh["w0"], neigh["ws"], neigh["wd"])
        pred = lookup(grid, np.zeros(va.sum(), dtype=int), dva, cva)
        scores.append(auc(y[va], pred))
        weights.append(int(va.sum()))
    if not scores:
        return 0.5
    return float(np.average(scores, weights=weights))


def oof_table(df, y, splits, nd, nc, m, neigh, alpha_glob=0.0):
    src_levels = sorted(df["source"].astype(str).unique()) + ["__UNK__"]
    oof = np.zeros(len(y))
    per_fold = []
    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        dtr, dva, _ = qcut_apply(trn["days"], val["days"], nd)
        ctr, cva, _ = qcut_apply(cond_tr, cond_va, nc)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        idx_tr = src_index(src_tr, src_levels)
        sm, ct = fill_grid(idx_tr, dtr, ctr, ytr, len(src_levels), nd, nc)
        prior = float(ytr.mean())
        glob = make_global(sm, ct, prior, m, neigh) if alpha_glob > 0 else None
        pred = predict_table(src_va, dva, cva, src_levels, sm, ct, prior, m, neigh, alpha_glob, glob)
        oof[va_i] = pred
        per_fold.append({"fold": fold, "auc": auc(y[va_i], pred)})
    return oof, per_fold


def oof_kernel(df, y, splits, hd, hc):
    """Per-source Nadaraya–Watson on (days, log cond). Small sources fall back to global."""
    oof = np.zeros(len(y))
    per_fold = []
    src_all = df["source"].astype(str)
    for fold, (tr_i, va_i) in enumerate(splits):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        dtr = trn["days"].to_numpy(float)
        dva = val["days"].to_numpy(float)
        ltr = np.log(np.clip(cond_tr, 1e-6, None))
        lva = np.log(np.clip(cond_va, 1e-6, None))
        pred = np.full(len(va_i), float(ytr.mean()))
        # global kernel as fallback
        pred = nw_predict(dva, lva, dtr, ltr, ytr, hd, hc)
        for s in np.unique(src_va):
            mtr = src_tr == s
            mva = src_va == s
            if mtr.sum() < 80 or mva.sum() == 0:
                continue
            pred[mva] = nw_predict(dva[mva], lva[mva], dtr[mtr], ltr[mtr], ytr[mtr], hd, hc)
        oof[va_i] = pred
        per_fold.append({"fold": fold, "auc": auc(y[va_i], pred)})
    return oof, per_fold


def nw_predict(dva, lva, dtr, ltr, ytr, hd, hc):
    # chunk val to keep memory bounded
    out = np.empty(len(dva))
    step = 400
    inv_hd2 = 1.0 / (hd * hd)
    inv_hc2 = 1.0 / (hc * hc)
    ytr = np.asarray(ytr, dtype=float)
    for i0 in range(0, len(dva), step):
        sl = slice(i0, i0 + step)
        dd = ((dva[sl][:, None] - dtr[None, :]) ** 2) * inv_hd2
        dc = ((lva[sl][:, None] - ltr[None, :]) ** 2) * inv_hc2
        K = np.exp(-0.5 * (dd + dc))
        den = K.sum(axis=1)
        den = np.where(den < 1e-12, 1.0, den)
        out[sl] = (K @ ytr) / den
    return out


def main():
    df, y = load_train()
    src_all = df["source"].astype(str).to_numpy()
    splits5 = skf_splits(y, 5)
    splits10 = skf_splits(y, 10)

    # --- LOSO hyperparameter search (global surface; tests shared generating process) ---
    loso_rows = []
    best_loso = {"auc": -1.0}
    for nd, nc in BIN_CFGS:
        for m in MS:
            for neigh in NEIGH:
                a = loso_auc(df, y, nd, nc, m, neigh)
                rec = {"nd": nd, "nc": nc, "m": m, "neigh": neigh["name"], "loso_auc": a}
                loso_rows.append(rec)
                print(f"LOSO d{nd}c{nc} m={m:.0f} {neigh['name']:7s} {a:.5f}", flush=True)
                if a > best_loso["auc"]:
                    best_loso = {**rec, "auc": a, "neigh_obj": neigh}

    loso_rows.sort(key=lambda z: -z["loso_auc"])

    # also 5-fold OOF screen for per-source tables (the actual scoring model)
    screen = []
    best5 = {"auc": -1.0}
    # restrict to a subset: baseline known-best + LOSO winner + 8neigh around (5,10)
    cand = []
    cand.append((5, 10, 10.0, NEIGH[0], 0.0))  # known baseline
    cand.append((5, 10, 10.0, NEIGH[2], 0.0))  # 8neigh
    cand.append((5, 10, 10.0, NEIGH[2], 0.25))
    cand.append((5, 10, 10.0, NEIGH[2], 0.4))
    cand.append((5, 10, 20.0, NEIGH[2], 0.25))
    cand.append((8, 10, 10.0, NEIGH[2], 0.25))
    # LOSO winner
    nw = best_loso["neigh_obj"]
    cand.append((best_loso["nd"], best_loso["nc"], best_loso["m"], nw, 0.0))
    cand.append((best_loso["nd"], best_loso["nc"], best_loso["m"], nw, 0.25))
    # unique
    seen = set()
    uniq = []
    for c in cand:
        key = (c[0], c[1], c[2], c[3]["name"], c[4])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    for nd, nc, m, neigh, ag in uniq:
        oof, _ = oof_table(df, y, splits5, nd, nc, m, neigh, ag)
        rec = {"nd": nd, "nc": nc, "m": m, "neigh": neigh["name"], "alpha_glob": ag, "auc5": auc(y, oof)}
        screen.append(rec)
        print(f"5fold table {rec}", flush=True)
        if rec["auc5"] > best5["auc"]:
            best5 = {**rec, "auc": rec["auc5"], "neigh_obj": neigh}

    # kernel bandwidth 5-fold (small grid)
    kern_screen = []
    bestk = {"auc": -1.0}
    for hd, hc in [(600, 0.35), (800, 0.45), (1000, 0.55), (1200, 0.40)]:
        oof, _ = oof_kernel(df, y, splits5, hd, hc)
        rec = {"hd": hd, "hc": hc, "auc5": auc(y, oof)}
        kern_screen.append(rec)
        print(f"5fold kernel {rec}", flush=True)
        if rec["auc5"] > bestk["auc"]:
            bestk = {**rec, "auc": rec["auc5"]}

    # 10-fold report
    reports = {}
    oof_win, pf = oof_table(df, y, splits10, best5["nd"], best5["nc"], best5["m"], best5["neigh_obj"], best5["alpha_glob"])
    reports["table_winner"] = {"cfg": {k: best5[k] for k in ("nd", "nc", "m", "neigh", "alpha_glob")}, **fold_report(y, oof_win, src_all, pf)}
    save_oof("exp3_table_oof.npy", oof_win)
    print("10fold table_winner", reports["table_winner"]["auc"], flush=True)

    oof_base, pf = oof_table(df, y, splits10, 5, 10, 10.0, NEIGH[0], 0.0)
    reports["table_d5c10_m10_nosmooth"] = fold_report(y, oof_base, src_all, pf)
    print("10fold baseline d5c10 m10", reports["table_d5c10_m10_nosmooth"]["auc"], flush=True)

    oof_8, pf = oof_table(df, y, splits10, 5, 10, 10.0, NEIGH[2], 0.25)
    reports["table_d5c10_m10_8neigh_a25"] = fold_report(y, oof_8, src_all, pf)
    save_oof("exp3_table8_oof.npy", oof_8)
    print("10fold 8neigh", reports["table_d5c10_m10_8neigh_a25"]["auc"], flush=True)

    oof_k, pf = oof_kernel(df, y, splits10, bestk["hd"], bestk["hc"])
    reports["kernel_winner"] = {"cfg": {"hd": bestk["hd"], "hc": bestk["hc"]}, **fold_report(y, oof_k, src_all, pf)}
    save_oof("exp3_kernel_oof.npy", oof_k)
    print("10fold kernel", reports["kernel_winner"]["auc"], flush=True)

    dump_json(
        "exp3_2d_smooth.json",
        {
            "protocol": "LOSO hyperparams on global days×cond surface; 5-fold screen; 10-fold report. Fold-internal qcut.",
            "loso_top15": loso_rows[:15],
            "best_loso": {k: v for k, v in best_loso.items() if k != "neigh_obj"},
            "screen_5fold_table": screen,
            "screen_5fold_kernel": kern_screen,
            "best_5fold_table": {k: v for k, v in best5.items() if k != "neigh_obj"},
            "report_10fold": reports,
        },
    )
    print("WROTE exp3_2d_smooth.json")


if __name__ == "__main__":
    main()
