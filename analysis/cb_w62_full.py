#!/usr/bin/env python3
"""W62 CatBoost teacher: 10fold × 8seed × Nbag, RMSE, NO inner ES.

Historical W62: Ordered d5 l2=10 + Plain d6 l2=6 rsm=0.3, 800 iter on the
FULL training fold (no 12% early-stop holdout). Blend 0.62*rank(main)+0.38*rank(alt).

Default degrade (time): 8 seed × 1 bag × 10 fold = 160 models.
Override with env:
  CB_SEEDS=2026,2027,...   CB_BAGS=1   CB_THREADS=2   CB_ITERS=800
  CB_NFOLD=10
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cb_features import (  # noqa: E402
    CATS_W62,
    NUM_ALT_W62,
    NUM_MAIN_W62,
    fold_features,
    load_raw,
)

ROOT = Path("/workspace")
SUB = ROOT / "submissions"
CKPT = ROOT / "analysis" / "cb_w62_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)

NFOLD = int(os.environ.get("CB_NFOLD", "10"))
N_BAG = int(os.environ.get("CB_BAGS", "1"))
ITERS = int(os.environ.get("CB_ITERS", "800"))
THREADS = int(os.environ.get("CB_THREADS", "2"))
SEEDS = [int(x) for x in os.environ.get("CB_SEEDS", ",".join(str(s) for s in range(2026, 2034))).split(",") if x.strip()]
BLEND_MAIN = 0.62
BLEND_ALT = 0.38

ARM1 = dict(
    loss_function="RMSE",
    boosting_type="Ordered",
    depth=5,
    l2_leaf_reg=10,
    rsm=1.0,
    learning_rate=0.03,
    iterations=ITERS,
    random_strength=1.0,
    bagging_temperature=0.2,
    border_count=128,
    thread_count=THREADS,
    verbose=False,
    allow_writing_files=False,
)
ARM2 = dict(
    loss_function="RMSE",
    boosting_type="Plain",
    depth=6,
    l2_leaf_reg=6,
    rsm=0.3,
    learning_rate=0.03,
    iterations=ITERS,
    random_strength=1.0,
    bagging_temperature=1.0,
    border_count=64,
    thread_count=THREADS,
    verbose=False,
    allow_writing_files=False,
)


def rank01(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a)
    return s.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def seed_path(seed: int) -> Path:
    return CKPT / f"seed_{seed}_bags{N_BAG}_folds{NFOLD}.npz"


def fit_predict(params: dict, seed: int, cat_idx: list[int], Xtr, ytr, Xva, Xte) -> tuple[np.ndarray, np.ndarray]:
    p = dict(params)
    p["random_seed"] = int(seed)
    model = CatBoostRegressor(**p)
    train = Pool(Xtr, ytr, cat_features=cat_idx)
    # Full training fold. No eval_set / no inner ES — this is the W62 alignment.
    model.fit(train)
    return (
        model.predict(Pool(Xva, cat_features=cat_idx)).astype(np.float64),
        model.predict(Pool(Xte, cat_features=cat_idx)).astype(np.float64),
    )


def run_one_seed(seed: int, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, folds) -> dict:
    n = len(train)
    nt = len(test)
    oof_m = np.zeros(n, dtype=np.float64)
    oof_a = np.zeros(n, dtype=np.float64)
    te_m = np.zeros(nt, dtype=np.float64)
    te_a = np.zeros(nt, dtype=np.float64)
    fold_aucs: list[float] = []
    t0 = time.time()

    for fi, (tr_idx, va_idx) in enumerate(folds):
        trn_f, va_f = fold_features(train.iloc[tr_idx].copy(), train.iloc[va_idx].copy())
        _, te_f = fold_features(train.iloc[tr_idx].copy(), test.copy())
        cols_m = NUM_MAIN_W62 + CATS_W62
        cols_a = NUM_ALT_W62 + CATS_W62
        missing_m = [c for c in cols_m if c not in trn_f.columns]
        missing_a = [c for c in cols_a if c not in trn_f.columns]
        if missing_m or missing_a:
            raise KeyError(f"missing features main={missing_m} alt={missing_a}")
        cat_idx = list(range(len(CATS_W62)))
        ytr = y[tr_idx]
        pm = np.zeros(len(va_idx), dtype=np.float64)
        pa = np.zeros(len(va_idx), dtype=np.float64)
        tm = np.zeros(nt, dtype=np.float64)
        ta = np.zeros(nt, dtype=np.float64)
        for b in range(N_BAG):
            rs = seed + 10007 * b + 17 * fi
            p1, t1 = fit_predict(ARM1, rs, cat_idx, trn_f[cols_m], ytr, va_f[cols_m], te_f[cols_m])
            p2, t2 = fit_predict(ARM2, rs + 7919, cat_idx, trn_f[cols_a], ytr, va_f[cols_a], te_f[cols_a])
            pm += p1
            pa += p2
            tm += t1
            ta += t2
            print(f"  seed={seed} fold={fi} bag={b} done  {time.time()-t0:.0f}s", flush=True)
        pm /= N_BAG
        pa /= N_BAG
        tm /= N_BAG
        ta /= N_BAG
        oof_m[va_idx] = pm
        oof_a[va_idx] = pa
        te_m += tm / NFOLD
        te_a += ta / NFOLD
        blend = BLEND_MAIN * rank01(pm) + BLEND_ALT * rank01(pa)
        auc_m = float(roc_auc_score(y[va_idx], pm))
        auc_a = float(roc_auc_score(y[va_idx], pa))
        auc_b = float(roc_auc_score(y[va_idx], blend))
        fold_aucs.append(auc_b)
        print(
            f"SEED {seed} FOLD {fi}  main={auc_m:.6f} alt={auc_a:.6f} blend={auc_b:.6f}  {time.time()-t0:.0f}s",
            flush=True,
        )
        # per-fold checkpoint so a kill mid-seed is not a total loss
        np.savez_compressed(
            CKPT / f"seed_{seed}_fold{fi}_partial.npz",
            oof_m=oof_m,
            oof_a=oof_a,
            te_m=te_m,
            te_a=te_a,
            fold=fi,
        )

    blend_oof = BLEND_MAIN * rank01(oof_m) + BLEND_ALT * rank01(oof_a)
    auc_m = float(roc_auc_score(y, oof_m))
    auc_a = float(roc_auc_score(y, oof_a))
    auc_b = float(roc_auc_score(y, blend_oof))
    out = {
        "seed": seed,
        "oof_m": oof_m,
        "oof_a": oof_a,
        "te_m": te_m,
        "te_a": te_a,
        "auc_main": auc_m,
        "auc_alt": auc_a,
        "auc_blend": auc_b,
        "fold_aucs": fold_aucs,
        "elapsed_s": time.time() - t0,
    }
    np.savez_compressed(
        seed_path(seed),
        oof_m=oof_m,
        oof_a=oof_a,
        te_m=te_m,
        te_a=te_a,
        auc_main=auc_m,
        auc_alt=auc_a,
        auc_blend=auc_b,
        fold_aucs=np.array(fold_aucs),
        seed=seed,
    )
    print(f"SEED {seed} DONE  main={auc_m:.6f} alt={auc_a:.6f} blend={auc_b:.6f}  {out['elapsed_s']:.0f}s", flush=True)
    return out


def load_seed(seed: int) -> dict | None:
    p = seed_path(seed)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {
        "seed": seed,
        "oof_m": z["oof_m"],
        "oof_a": z["oof_a"],
        "te_m": z["te_m"],
        "te_a": z["te_a"],
        "auc_main": float(z["auc_main"]),
        "auc_alt": float(z["auc_alt"]),
        "auc_blend": float(z["auc_blend"]),
        "fold_aucs": [float(x) for x in z["fold_aucs"]],
        "elapsed_s": 0.0,
    }


def write_outputs(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, results: list[dict]) -> dict:
    oof_m = np.mean([r["oof_m"] for r in results], axis=0)
    oof_a = np.mean([r["oof_a"] for r in results], axis=0)
    te_m = np.mean([r["te_m"] for r in results], axis=0)
    te_a = np.mean([r["te_a"] for r in results], axis=0)
    blend_oof = BLEND_MAIN * rank01(oof_m) + BLEND_ALT * rank01(oof_a)
    blend_te = BLEND_MAIN * rank01(te_m) + BLEND_ALT * rank01(te_a)
    auc_m = float(roc_auc_score(y, oof_m))
    auc_a = float(roc_auc_score(y, oof_a))
    auc_b = float(roc_auc_score(y, blend_oof))
    per_seed = [
        {
            "seed": r["seed"],
            "auc_main": r["auc_main"],
            "auc_alt": r["auc_alt"],
            "auc_blend": r["auc_blend"],
            "fold_aucs": r["fold_aucs"],
        }
        for r in results
    ]
    metrics = {
        "recipe": "W62-aligned CatBoost RMSE, no inner ES, 800 iter full train fold",
        "nfold": NFOLD,
        "n_bag": N_BAG,
        "n_seed": len(results),
        "seeds": [r["seed"] for r in results],
        "iterations": ITERS,
        "threads": THREADS,
        "cats": CATS_W62,
        "n_cats": len(CATS_W62),
        "nums_main": NUM_MAIN_W62,
        "nums_alt": NUM_ALT_W62,
        "blend": f"{BLEND_MAIN}*rank(main)+{BLEND_ALT}*rank(alt)",
        "auc_cb_main": auc_m,
        "auc_cb_alt": auc_a,
        "auc_cb_w62": auc_b,
        "per_seed": per_seed,
        "honest": True,
        "inner_es": False,
        "target_oof": 0.700,
        "historical_w62_oof": 0.70159,
        "historical_w62_online": 0.71503,
    }
    (SUB / "cb_w62_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        {
            "id": train["id"].astype(str).to_numpy(),
            "label": y.astype(np.int32),
            "pred_cb_main": oof_m,
            "pred_cb_alt": oof_a,
            "pred_cb_w62": blend_oof,
        }
    ).to_parquet(SUB / "cb_w62_oof.parquet", index=False)
    pd.DataFrame(
        {
            "id": test["id"].astype(str).to_numpy(),
            "pred_cb_main": te_m,
            "pred_cb_alt": te_a,
            "pred_cb_w62": blend_te,
        }
    ).to_parquet(SUB / "cb_w62_test.parquet", index=False)
    report = SUB / "cb_oof_report.txt"
    extra = (
        f"\n\n=== W62 full (no inner ES) {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        f"nfold={NFOLD} bags={N_BAG} seeds={metrics['seeds']} iters={ITERS}\n"
        f"auc_cb_main={auc_m:.6f}\n"
        f"auc_cb_alt={auc_a:.6f}\n"
        f"auc_cb_w62={auc_b:.6f}   blend={metrics['blend']}\n"
        f"inner_es=False  honest={True}\n"
        + "\n".join(
            f"  seed {p['seed']}: blend={p['auc_blend']:.6f} main={p['auc_main']:.6f} alt={p['auc_alt']:.6f}"
            for p in per_seed
        )
        + "\n"
    )
    with report.open("a", encoding="utf-8") as f:
        f.write(extra)
    print(json.dumps({k: metrics[k] for k in ("auc_cb_main", "auc_cb_alt", "auc_cb_w62", "n_seed", "n_bag", "nfold")}, indent=2), flush=True)
    return metrics


def main() -> None:
    print(
        f"W62 start seeds={SEEDS} bags={N_BAG} folds={NFOLD} iters={ITERS} threads={THREADS} cats={len(CATS_W62)}",
        flush=True,
    )
    print(f"CATS={CATS_W62}", flush=True)
    train, test = load_raw()
    y = train["label"].to_numpy(dtype=np.int32)
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=2026)
    folds = list(skf.split(train, y))
    results: list[dict] = []
    for seed in SEEDS:
        cached = load_seed(seed)
        if cached is not None:
            print(f"SEED {seed} loaded checkpoint auc_blend={cached['auc_blend']:.6f}", flush=True)
            results.append(cached)
        else:
            try:
                results.append(run_one_seed(seed, train, test, y, folds))
            except Exception:
                traceback.print_exc()
                print(f"SEED {seed} FAILED — writing partial outputs from {len(results)} seeds", flush=True)
                break
        if results:
            write_outputs(train, test, y, results)
            last = results[-1]["auc_blend"] if results else 0
            # After averaging, check current ensemble
            oof_m = np.mean([r["oof_m"] for r in results], axis=0)
            oof_a = np.mean([r["oof_a"] for r in results], axis=0)
            ens = float(roc_auc_score(y, BLEND_MAIN * rank01(oof_m) + BLEND_ALT * rank01(oof_a)))
            print(f"ENSEMBLE n_seed={len(results)} auc_w62={ens:.6f} last_seed={last:.6f}", flush=True)
            if ens >= 0.700 and len(results) >= 4:
                print("Hit OOF>=0.700 with >=4 seeds; continuing remaining seeds if time allows.", flush=True)
    if not results:
        raise SystemExit("no seeds completed")
    write_outputs(train, test, y, results)


if __name__ == "__main__":
    main()
