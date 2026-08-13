#!/usr/bin/env python3
"""Tiny factorization machine on medium cats (RMSE). Honest 10-fold.

p = w0 + sum w_f[x_f] + sum_{f<g} <v_f[x_f], v_g[x_g]>
Spark-portable: embeddings are just lookup tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cb_features import fold_features

OUT = Path("/workspace/analysis/reverse")
CATS_FM = ["source", "region", "age_range", "src_cq", "src_dq5", "reg_age", "days_q5", "cond_q"]
K = 8
LR = 0.05
EPOCHS = 12
L2 = 1e-4


def auc(y, s):
    return float(roc_auc_score(y, s))


def encode(tr, va, cols):
    maps = []
    Xtr, Xva = [], []
    for c in cols:
        uniq = pd.Index(tr[c].astype(str).unique())
        mp = {k: i for i, k in enumerate(uniq)}
        maps.append(len(mp))
        Xtr.append(tr[c].astype(str).map(mp).fillna(0).astype(int).to_numpy())
        Xva.append(va[c].astype(str).map(mp).fillna(0).astype(int).to_numpy())
    return np.column_stack(Xtr), np.column_stack(Xva), maps


def sgd_fm(Xtr, ytr, Xva, cards, rng):
    n, f = Xtr.shape
    w0 = 0.1
    w = [np.zeros(c) for c in cards]
    v = [rng.normal(0, 0.01, size=(c, K)) for c in cards]
    idx = np.arange(n)
    for ep in range(EPOCHS):
        rng.shuffle(idx)
        for i in idx:
            xs = Xtr[i]
            # pred
            lin = w0
            vecs = []
            for j in range(f):
                lin += w[j][xs[j]]
                vecs.append(v[j][xs[j]])
            vecs = np.stack(vecs, 0)  # f x k
            inter = 0.5 * ((vecs.sum(0) ** 2) - (vecs ** 2).sum(0)).sum()
            p = lin + inter
            # clip-ish
            g = p - ytr[i]
            w0 -= LR * (g + L2 * w0)
            for j in range(f):
                w[j][xs[j]] -= LR * (g + L2 * w[j][xs[j]])
                others = vecs.sum(0) - vecs[j]
                v[j][xs[j]] -= LR * (g * others + L2 * v[j][xs[j]])
    # predict val
    out = np.empty(len(Xva))
    for i, xs in enumerate(Xva):
        lin = w0
        vecs = []
        for j in range(f):
            lin += w[j][xs[j]]
            vecs.append(v[j][xs[j]])
        vecs = np.stack(vecs, 0)
        inter = 0.5 * ((vecs.sum(0) ** 2) - (vecs ** 2).sum(0)).sum()
        out[i] = lin + inter
    return out


def main():
    train = pd.read_csv("/workspace/data/train.csv")
    y = train["label"].astype(int).to_numpy()
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=2026)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        Xtr, Xva, cards = encode(trn, val, CATS_FM)
        pred = sgd_fm(Xtr, y[tr_i].astype(float), Xva, cards, np.random.RandomState(fold + 7))
        oof[va_i] = pred
        print(f"fold {fold} auc={auc(y[va_i], pred):.4f}", flush=True)
    a = auc(y, oof)
    print("FM OOF", a)
    np.save(OUT / "exp9j_fm.npy", oof)
    (OUT / "exp9j_fm.json").write_text(json.dumps({"oof": a, "k": K, "cats": CATS_FM}))
    h = np.load(OUT / "exp8h_max2.npy")
    r = pd.Series(h).rank(pct=True).to_numpy()
    rf = pd.Series(oof).rank(pct=True).to_numpy()
    print("blend 0.85h+0.15fm", auc(y, 0.85 * r + 0.15 * rf))


if __name__ == "__main__":
    main()
