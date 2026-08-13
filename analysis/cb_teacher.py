#!/usr/bin/env python3
"""Honest 10-fold CatBoost dual-arm teacher (RMSE). Contrast / rank-fusion scores for Scala.

W62 recipe without sparse 3-way cats:
  Arm1 Ordered depth=5 l2=10
  Arm2 Plain depth=6 l2=6 rsm=0.3
  10-fold x 3-bag, rank blend 0.62/0.38
  src_cq_dq is TE-only, never a CatBoost categorical.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cb_features import CATS, HIGH_TE_KEYS, NUM_ALT, NUM_MAIN, fold_features, te_apply

warnings.filterwarnings("ignore")

DATA = os.environ.get("CLAIM_DATA", "/workspace/data")
OUT = os.environ.get("CLAIM_OUT", "/workspace/submissions")
N_FOLDS = int(os.environ.get("CB_FOLDS", "10"))
N_BAGS = int(os.environ.get("CB_BAGS", "3"))
THREADS = int(os.environ.get("CB_THREADS", "2"))
ITERS = int(os.environ.get("CB_ITERS", "800"))
SEED0 = int(os.environ.get("CB_SEED", "2026"))

os.makedirs(OUT, exist_ok=True)
os.makedirs("/workspace/analysis", exist_ok=True)


def rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(pct=True).to_numpy()


def fit_predict(trn: pd.DataFrame, val: pd.DataFrame, ytr: np.ndarray, nums: list[str], params: dict) -> np.ndarray:
    cols = nums + CATS
    Xtr = trn[cols].copy()
    Xva = val[cols].copy()
    for c in CATS:
        Xtr[c] = Xtr[c].astype(str)
        Xva[c] = Xva[c].astype(str)
    # Honest ES: inner holdout from TRAIN only (never the OOF val fold).
    strat = (ytr > 0.5).astype(int)
    try:
        Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.12, random_state=int(params.get("random_seed", 0)), stratify=strat)
    except ValueError:
        Xa, Xb, ya, yb = Xtr, Xtr.iloc[:0], ytr, ytr[:0]
    pool_tr = Pool(Xa, ya, cat_features=CATS)
    fit_kw = dict(verbose=False)
    if len(yb) >= 30:
        fit_kw["eval_set"] = Pool(Xb, yb, cat_features=CATS)
        fit_kw["use_best_model"] = True
    model = CatBoostRegressor(**params)
    model.fit(pool_tr, **fit_kw)
    return model.predict(Pool(Xva, cat_features=CATS))


def params_main(seed: int) -> dict:
    return dict(
        loss_function="RMSE",
        iterations=ITERS,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=10,
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Ordered",
        rsm=1.0,
    )


def params_alt(seed: int) -> dict:
    return dict(
        loss_function="RMSE",
        iterations=ITERS,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6,
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=THREADS,
        boosting_type="Plain",
        rsm=0.3,
    )


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = train["label"].astype(int).to_numpy()
    ids = train["id"].astype(str).to_numpy()
    test_ids = test["id"].astype(str).to_numpy()

    oof_main = np.zeros(len(y), dtype=np.float64)
    oof_alt = np.zeros(len(y), dtype=np.float64)
    oof_te3 = np.zeros(len(y), dtype=np.float64)
    oof_cqdq = np.zeros(len(y), dtype=np.float64)
    oof_srq = np.zeros(len(y), dtype=np.float64)
    test_main = np.zeros(len(test), dtype=np.float64)
    test_alt = np.zeros(len(test), dtype=np.float64)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED0)
    bag_seeds = [SEED0 + 10 * b for b in range(N_BAGS)]

    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        t_fold = time.time()
        trn, val = fold_features(train.iloc[tr_i], train.iloc[va_i])
        ytr = y[tr_i]
        yva = y[va_i]
        pm = np.zeros(len(va_i))
        pa = np.zeros(len(va_i))
        for b, sd in enumerate(bag_seeds):
            pm += fit_predict(trn, val, ytr, NUM_MAIN, params_main(sd + fold))
            pa += fit_predict(trn, val, ytr, NUM_ALT, params_alt(sd + 100 + fold))
            print(f"  fold {fold} bag {b} done", flush=True)
        pm /= N_BAGS
        pa /= N_BAGS
        oof_main[va_i] = pm
        oof_alt[va_i] = pa
        oof_te3[va_i] = te_apply(trn["src_cq_dq"].to_numpy(), val["src_cq_dq"].to_numpy(), ytr)
        oof_cqdq[va_i] = te_apply(trn["cq_dq"].to_numpy(), val["cq_dq"].to_numpy(), ytr)
        oof_srq[va_i] = te_apply(trn["src_ratioq"].to_numpy(), val["src_ratioq"].to_numpy(), ytr)
        print(
            f"[fold {fold}] main={roc_auc_score(yva, pm):.5f} alt={roc_auc_score(yva, pa):.5f} "
            f"te3={roc_auc_score(yva, oof_te3[va_i]):.5f} dt={time.time()-t_fold:.1f}s",
            flush=True,
        )
        np.savez(
            "/workspace/analysis/cb_teacher_oof_ckpt.npz",
            oof_main=oof_main,
            oof_alt=oof_alt,
            oof_te3=oof_te3,
            y=y,
            ids=ids,
            fold=fold,
        )

    # Full-data fit for test (bags; 12% inner ES from train only).
    trn_full, tes = fold_features(train, test)
    for b, sd in enumerate(bag_seeds):
        test_main += fit_predict(trn_full, tes, y, NUM_MAIN, params_main(sd + 999))
        test_alt += fit_predict(trn_full, tes, y, NUM_ALT, params_alt(sd + 1999))
        print(f"  full bag {b} done", flush=True)
    test_main /= N_BAGS
    test_alt /= N_BAGS

    r_main, r_alt = rank(oof_main), rank(oof_alt)
    w62 = 0.62 * r_main + 0.38 * r_alt
    r_te3, r_cq, r_srq = rank(oof_te3), rank(oof_cqdq), rank(oof_srq)
    blend = 0.82 * w62 + 0.10 * r_te3 + 0.04 * r_cq + 0.04 * r_srq

    auc_main = float(roc_auc_score(y, oof_main))
    auc_alt = float(roc_auc_score(y, oof_alt))
    auc_w62 = float(roc_auc_score(y, w62))
    auc_te3 = float(roc_auc_score(y, oof_te3))
    auc_cq = float(roc_auc_score(y, oof_cqdq))
    auc_srq = float(roc_auc_score(y, oof_srq))
    auc_blend = float(roc_auc_score(y, blend))
    auc_max2 = float(roc_auc_score(y, np.maximum(r_main, r_alt)))

    report = (
        "cb_oof_report\n"
        "backend=python_catboost_1.2.10_teacher\n"
        f"n_folds={N_FOLDS}\n"
        f"n_bags={N_BAGS}\n"
        f"iterations={ITERS}\n"
        "loss=RMSE\n"
        "arm1=Ordered,depth=5,l2=10,rsm=1.0\n"
        "arm2=Plain,depth=6,l2=6,rsm=0.3\n"
        "cats=medium_only_no_src_cq_dq\n"
        f"auc_cb_main={auc_main:.6f}\n"
        f"auc_cb_alt={auc_alt:.6f}\n"
        f"auc_cb_w62={auc_w62:.6f}\n"
        f"auc_cb_max2={auc_max2:.6f}\n"
        f"auc_te3={auc_te3:.6f}\n"
        f"auc_cqdq={auc_cq:.6f}\n"
        f"auc_src_ratioq={auc_srq:.6f}\n"
        f"auc_blend={auc_blend:.6f}\n"
        f"elapsed_sec={time.time()-t0:.1f}\n"
        "note=3-way TE is rank-fusion only; never a CatBoost cat column\n"
    )
    print(report, flush=True)
    with open(os.path.join(OUT, "cb_oof_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, "cb_teacher_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "auc_cb_main": auc_main,
                "auc_cb_alt": auc_alt,
                "auc_cb_w62": auc_w62,
                "auc_blend": auc_blend,
                "auc_te3": auc_te3,
            },
            f,
            indent=2,
        )

    oof_df = pd.DataFrame(
        {
            "id": ids,
            "label": y,
            "pred_cb_main": oof_main,
            "pred_cb_alt": oof_alt,
            "pred_cb_w62": w62,
            "te_src_cq_dq": oof_te3,
            "te_cq_dq": oof_cqdq,
            "te_src_ratioq": oof_srq,
        }
    )
    tes_df = pd.DataFrame(
        {
            "id": test_ids,
            "pred_cb_main": test_main,
            "pred_cb_alt": test_alt,
        }
    )
    oof_path = os.path.join(OUT, "cb_teacher_oof.parquet")
    tes_path = os.path.join(OUT, "cb_teacher_test.parquet")
    oof_df.to_parquet(oof_path, index=False)
    tes_df.to_parquet(tes_path, index=False)
    oof_df.to_csv(os.path.join(OUT, "cb_teacher_oof.csv"), index=False)
    tes_df.to_csv(os.path.join(OUT, "cb_teacher_test.csv"), index=False)
    print(f"wrote {oof_path} {tes_path}", flush=True)


if __name__ == "__main__":
    main()
