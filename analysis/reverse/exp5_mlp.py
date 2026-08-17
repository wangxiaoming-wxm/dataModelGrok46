#!/usr/bin/env python3
"""Exp5: Bayes-ceiling probes — MLP / HGB / KNN / CatBoost on source+region+days+condition."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import auc, dump_json, fill_condition, load_train, per_source_rank, save_oof, skf, numeric_block

try:
    from catboost import CatBoostRegressor, Pool

    HAS_CB = True
except Exception:
    HAS_CB = False


def pack_mlp(trn, val, cond_tr, cond_va):
    enc_s = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc_r = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    S_tr = enc_s.fit_transform(trn[["source"]])
    S_va = enc_s.transform(val[["source"]])
    R_tr = enc_r.fit_transform(trn[["region"]])
    R_va = enc_r.transform(val[["region"]])
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
    rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
    num_tr = np.column_stack(
        [
            trn["days"] / 5000.0,
            np.log1p(trn["days"]),
            cond_tr,
            np.log(np.clip(cond_tr, 1e-6, None)),
            rk_tr,
            (rk_tr - 0.5) ** 2,
            trn["age_range"] / 10.0,
            (trn["age_range"] >= 8).astype(float),
        ]
    )
    num_va = np.column_stack(
        [
            val["days"] / 5000.0,
            np.log1p(val["days"]),
            cond_va,
            np.log(np.clip(cond_va, 1e-6, None)),
            rk_va,
            (rk_va - 0.5) ** 2,
            val["age_range"] / 10.0,
            (val["age_range"] >= 8).astype(float),
        ]
    )
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(num_tr)
    num_va = scaler.transform(num_va)
    return np.hstack([S_tr, R_tr, num_tr]), np.hstack([S_va, R_va, num_va])


def main():
    df, y = load_train()
    n = len(y)
    oof_mlp = np.zeros(n)
    oof_mlp2 = np.zeros(n)
    oof_hgb = np.zeros(n)
    oof_knn = np.zeros(n)
    oof_cb = np.zeros(n) if HAS_CB else None

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i].copy(), df.iloc[va_i].copy()
        ytr, yva = y[tr_i].astype(float), y[va_i]
        cond_tr, cond_va = fill_condition(trn, val)
        Xtr, Xva = pack_mlp(trn, val, cond_tr, cond_va)

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=fold,
            batch_size=256,
        )
        mlp.fit(Xtr, ytr)
        oof_mlp[va_i] = mlp.predict(Xva)

        mlp2 = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation="tanh",
            solver="adam",
            alpha=3e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=100 + fold,
        )
        mlp2.fit(Xtr, ytr)
        oof_mlp2[va_i] = mlp2.predict(Xva)

        # HGB on numeric_block + integer cats
        Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr)
        Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr)
        src_codes = {s: i for i, s in enumerate(sorted(trn["source"].unique()))}
        reg_codes = {s: i for i, s in enumerate(sorted(trn["region"].unique()))}
        cat_tr = np.column_stack(
            [
                trn["source"].map(src_codes).fillna(-1).to_numpy(),
                trn["region"].map(reg_codes).fillna(-1).to_numpy(),
                trn["age_range"].to_numpy(),
            ]
        )
        cat_va = np.column_stack(
            [
                val["source"].map(src_codes).fillna(-1).to_numpy(),
                val["region"].map(reg_codes).fillna(-1).to_numpy(),
                val["age_range"].to_numpy(),
            ]
        )
        Htr = np.hstack([Ntr.to_numpy(float), cat_tr])
        Hva = np.hstack([Nva.to_numpy(float), cat_va])
        hgb = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=5,
            max_iter=400,
            min_samples_leaf=80,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=fold,
            categorical_features=[Htr.shape[1] - 3, Htr.shape[1] - 2, Htr.shape[1] - 1],
        )
        hgb.fit(Htr, ytr)
        oof_hgb[va_i] = hgb.predict(Hva)

        # global KNN in (source-oh is too high dim); use scaled days, log cond, src code, region code
        Ztr = np.column_stack(
            [
                trn["days"].to_numpy() / 4000.0,
                np.log(np.clip(cond_tr, 1e-6, None)),
                cat_tr[:, 0] / 5.0,
                cat_tr[:, 1] / 10.0,
                trn["age_range"] / 10.0,
            ]
        )
        Zva = np.column_stack(
            [
                val["days"].to_numpy() / 4000.0,
                np.log(np.clip(cond_va, 1e-6, None)),
                cat_va[:, 0] / 5.0,
                cat_va[:, 1] / 10.0,
                val["age_range"] / 10.0,
            ]
        )
        knn = KNeighborsRegressor(n_neighbors=70, weights="distance")
        knn.fit(Ztr, ytr)
        oof_knn[va_i] = knn.predict(Zva)

        if HAS_CB:
            trn["condition_f"] = cond_tr
            val["condition_f"] = cond_va
            cols = ["days", "condition_f", "age_range", "source", "region"]
            pool_tr = Pool(trn[cols], ytr, cat_features=["source", "region"])
            pool_va = Pool(val[cols], yva, cat_features=["source", "region"])
            cb = CatBoostRegressor(
                loss_function="RMSE",
                iterations=600,
                learning_rate=0.04,
                depth=5,
                l2_leaf_reg=8,
                random_seed=fold,
                od_type="Iter",
                od_wait=50,
                allow_writing_files=False,
                thread_count=4,
                verbose=False,
            )
            cb.fit(pool_tr, eval_set=pool_va, use_best_model=True)
            oof_cb[va_i] = cb.predict(pool_va)

        msg = (
            f"fold {fold} mlp={auc(yva, oof_mlp[va_i]):.4f} mlp2={auc(yva, oof_mlp2[va_i]):.4f} "
            f"hgb={auc(yva, oof_hgb[va_i]):.4f} knn={auc(yva, oof_knn[va_i]):.4f}"
        )
        if HAS_CB:
            msg += f" cb={auc(yva, oof_cb[va_i]):.4f}"
        print(msg, flush=True)

    report = {
        "mlp_64_32": auc(y, oof_mlp),
        "mlp_32_tanh": auc(y, oof_mlp2),
        "hgb_numeric_plus_cats": auc(y, oof_hgb),
        "knn70": auc(y, oof_knn),
        "catboost_core5": auc(y, oof_cb) if HAS_CB else None,
        "note": "If MLP>=0.70, distill embeddings into portable features. Remaining gap to 0.75 is unused signal or irreducible noise.",
    }
    print("\n=== EXP5 BAYES CEILING ===")
    for k, v in report.items():
        if isinstance(v, float) or v is None:
            print(f"  {k:24s} {v}")
    dump_json("exp5_mlp.json", report)
    save_oof("exp5_oof_mlp.npy", oof_mlp)
    save_oof("exp5_oof_hgb.npy", oof_hgb)
    save_oof("exp5_oof_knn.npy", oof_knn)
    if HAS_CB:
        save_oof("exp5_oof_cb_core.npy", oof_cb)
    print("WROTE exp5_mlp.json")


if __name__ == "__main__":
    main()
