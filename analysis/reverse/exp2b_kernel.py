#!/usr/bin/env python3
"""Per-source 2D Nadaraya-Watson / fine-grid lookup. Spark-portable as a 2D table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import auc, dump_json, fill_condition, load_train, save_oof, skf


def nw_predict(dtr, ctr, ytr, dva, cva, hd, hc):
    """Vectorized NW for modest n. Uses chunking on val."""
    dtr = np.asarray(dtr, float)
    ctr = np.log(np.clip(np.asarray(ctr, float), 1e-6, None))
    dva = np.asarray(dva, float)
    cva = np.log(np.clip(np.asarray(cva, float), 1e-6, None))
    ytr = np.asarray(ytr, float)
    out = np.empty(len(dva))
    # chunk val to keep memory small
    bs = 512
    inv2d = 1.0 / (2.0 * hd * hd)
    inv2c = 1.0 / (2.0 * hc * hc)
    for i in range(0, len(dva), bs):
        sl = slice(i, i + bs)
        dd = (dva[sl][:, None] - dtr[None, :]) / 1.0
        dc = cva[sl][:, None] - ctr[None, :]
        w = np.exp(-(dd * dd) * inv2d - (dc * dc) * inv2c)
        sw = w.sum(axis=1) + 1e-12
        out[sl] = (w @ ytr) / sw
    return out


def grid_lookup(dtr, ctr, ytr, dva, cva, nd=20, nc=12, m=30.0, prior=0.1):
    """Smoothed 2D histogram on equal-count bins + bilinear-ish neighbor pooling."""
    dtr = np.asarray(dtr, float)
    ctr = np.asarray(ctr, float)
    dva = np.asarray(dva, float)
    cva = np.asarray(cva, float)
    ytr = np.asarray(ytr, float)
    # quantile edges from train
    de = np.unique(np.quantile(dtr, np.linspace(0, 1, nd + 1)))
    ce = np.unique(np.quantile(ctr, np.linspace(0, 1, nc + 1)))
    de[0], de[-1] = -np.inf, np.inf
    ce[0], ce[-1] = -np.inf, np.inf
    id_tr = np.clip(np.searchsorted(de, dtr, side="right") - 1, 0, len(de) - 2)
    ic_tr = np.clip(np.searchsorted(ce, ctr, side="right") - 1, 0, len(ce) - 2)
    ndb, ncb = len(de) - 1, len(ce) - 1
    sm = np.zeros((ndb, ncb))
    ct = np.zeros((ndb, ncb))
    np.add.at(sm, (id_tr, ic_tr), ytr)
    np.add.at(ct, (id_tr, ic_tr), 1.0)
    # 8-neighbor smooth
    sm2 = sm.copy()
    ct2 = ct.copy()
    for di in (-1, 0, 1):
        for cj in (-1, 0, 1):
            if di == 0 and cj == 0:
                continue
            sm2 += np.roll(np.roll(sm, di, 0), cj, 1) * 0.25
            ct2 += np.roll(np.roll(ct, di, 0), cj, 1) * 0.25
    p = (sm2 + prior * m) / (ct2 + m)
    id_va = np.clip(np.searchsorted(de, dva, side="right") - 1, 0, ndb - 1)
    ic_va = np.clip(np.searchsorted(ce, cva, side="right") - 1, 0, ncb - 1)
    return p[id_va, ic_va]


def main():
    df, y = load_train()
    n = len(y)
    oof_nw = np.zeros(n)
    oof_grid = np.zeros(n)
    oof_nw_g = np.zeros(n)  # global NW not per source
    hds = [400, 700, 1100]
    hcs = [0.35, 0.55]

    # pick bandwidth by first fold inner? keep fixed from domain: days std ~ 3000, logcond std ~1
    hd, hc = 800.0, 0.45

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        days_tr = trn["days"].to_numpy(float)
        days_va = val["days"].to_numpy(float)
        pred_nw = np.full(len(va_i), float(ytr.mean()))
        pred_g = np.full(len(va_i), float(ytr.mean()))
        for s in np.unique(src_tr):
            mtr = src_tr == s
            mva = src_va == s
            if mtr.sum() < 60 or mva.sum() == 0:
                continue
            pred_nw[mva] = nw_predict(days_tr[mtr], cond_tr[mtr], ytr[mtr], days_va[mva], cond_va[mva], hd, hc)
            pred_g[mva] = grid_lookup(days_tr[mtr], cond_tr[mtr], ytr[mtr], days_va[mva], cond_va[mva], nd=16, nc=10, m=25.0, prior=float(ytr[mtr].mean()))
        oof_nw[va_i] = pred_nw
        oof_grid[va_i] = pred_g
        oof_nw_g[va_i] = nw_predict(days_tr, cond_tr, ytr, days_va, cond_va, 900.0, 0.5)
        print(
            f"fold {fold} nw_src={auc(y[va_i], pred_nw):.4f} grid={auc(y[va_i], pred_g):.4f} nw_g={auc(y[va_i], oof_nw_g[va_i]):.4f}",
            flush=True,
        )

    report = {
        "nw_per_source": auc(y, oof_nw),
        "grid16x10_per_source": auc(y, oof_grid),
        "nw_global": auc(y, oof_nw_g),
        "hd": hd,
        "hc": hc,
        "spark": "Store per-source 16x10 (or 20x12) smoothed mean table; lookup by train-fold quantile edges + neighbor pooling.",
    }
    print("=== EXP2b KERNEL ===", report)
    dump_json("exp2b_kernel.json", report)
    save_oof("exp2b_oof_nw.npy", oof_nw)
    save_oof("exp2b_oof_grid.npy", oof_grid)
    print("WROTE exp2b_kernel.json")


if __name__ == "__main__":
    main()
