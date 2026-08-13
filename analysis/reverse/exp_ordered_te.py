#!/usr/bin/env python3
"""Ordered (expanding-mean) target encoding — CatBoost-like, Spark-portable, independent scores + Ridge."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import auc, dump_json, fill_condition, load_train, numeric_block, qcut_apply, save_oof, skf, te_val_only
from exp8a_portable import make_key_maps, TE_KEYS


def ordered_te(keys, y, n_perm=5, m=20.0, seed=0):
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    n = len(y)
    prior = float(y.mean())
    acc = np.zeros(n)
    rng = np.random.RandomState(seed)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        sm = {}
        ct = {}
        enc = np.empty(n)
        for i in perm:
            k = keys[i]
            c = ct.get(k, 0)
            s = sm.get(k, 0.0)
            enc[i] = (s + prior * m) / (c + m)
            sm[k] = s + y[i]
            ct[k] = c + 1
        acc += enc
    return acc / n_perm


def main():
    df, y = load_train()
    n = len(y)
    oof_ridge = np.zeros(n)
    oof_mean = np.zeros(n)
    oof_bestkey = np.zeros(n)
    key_aucs = {name: [] for name, _ in TE_KEYS}

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        km_tr, km_va, *_ = make_key_maps(trn, val, cond_tr, cond_va)
        te_tr = []
        te_va = []
        va_named = {}
        for name, m in TE_KEYS:
            tr_enc = ordered_te(km_tr[name], ytr, n_perm=4, m=m, seed=fold * 17 + hash(name) % 1000)
            va_enc = te_val_only(km_tr[name], km_va[name], ytr, m=m)
            te_tr.append(tr_enc)
            te_va.append(va_enc)
            va_named[name] = va_enc
            key_aucs[name].append(auc(y[va_i], va_enc))
        TE_tr = np.column_stack(te_tr)
        TE_va = np.column_stack(te_va)
        Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr).to_numpy(float)
        Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr).to_numpy(float)
        Xtr = np.hstack([Ntr, TE_tr])
        Xva = np.hstack([Nva, TE_va])
        ridge = Ridge(alpha=25.0)
        ridge.fit(Xtr, ytr)
        oof_ridge[va_i] = ridge.predict(Xva)
        # TE-only ridge
        r2 = Ridge(alpha=15.0)
        r2.fit(TE_tr, ytr)
        oof_mean[va_i] = r2.predict(TE_va)
        oof_bestkey[va_i] = va_named["src_cq_dq"]
        print(
            f"fold {fold} ridge+num={auc(y[va_i], oof_ridge[va_i]):.4f} te_ridge={auc(y[va_i], oof_mean[va_i]):.4f} "
            f"src_cq_dq={auc(y[va_i], va_named['src_cq_dq']):.4f}",
            flush=True,
        )

    report = {
        "ordered_te_ridge_plus_num": auc(y, oof_ridge),
        "ordered_te_ridge_only": auc(y, oof_mean),
        "src_cq_dq": auc(y, oof_bestkey),
        "per_key_mean_fold_auc": {k: float(np.mean(v)) for k, v in key_aucs.items()},
    }
    print("=== ORDERED TE ===", report)
    dump_json("exp_ordered_te.json", report)
    save_oof("exp_ote_ridge.npy", oof_ridge)
    save_oof("exp_ote_teonly.npy", oof_mean)
    print("WROTE exp_ordered_te.json")


if __name__ == "__main__":
    main()
