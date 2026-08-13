#!/usr/bin/env python3
"""HONEST 10-fold CatBoostClassifier on opus5 FE.

Protocol (same as merger_ord8 / v2_cat_alt8, only n_splits changes):
  - CatBoostClassifier + Logloss, fixed 800 trees, no eval_set / no ES
  - label-free FE fitted on train+test (opus5 src2)
  - StratifiedKFold(10), seeds 2026..2033
  - rank-pool seeds, then max2(rank(main), rank(alt))

Historical W62 is 10-fold × 8-seed × 3-bag RMSE at 0.70159. opus5 is the same
dual-world idea at 5-fold × 8-seed × 1-bag Classifier, nested 0.69993 / full
0.70023. This run asks whether the missing fold count closes that 0.0014 gap.

Usage:
  python3 -u analysis/opus5/train_honest10.py --run
  python3 -u analysis/opus5/train_honest10.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "src2"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from arms import ARMS, CAT_BASE, altboost_frame, catboost_frame
from features import fit_edges, fit_edges_alt

ROOT = Path("/workspace") if Path("/workspace/data/train.csv").is_file() else HERE.parents[1]
DATA = ROOT / "data"
ART = HERE / "artifacts"
CKPT = HERE / "ckpt_honest10"
ART.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

ALL_SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
N_FOLD = 10
ITERS = 800
W62_OOF = 0.70159
THREADS = int(os.environ.get("CB_THREADS", "4"))
TARGET_FULL = 0.70023  # opus5 5-fold 8-seed max2
# opus5 5-fold per-seed AUCs (merger_ord8 / v2_cat_alt8 artifacts)
FIVE_FOLD_MAIN = {
    2026: 0.690039, 2027: 0.687846, 2028: 0.689450, 2029: 0.686596,
    2030: 0.690014, 2031: 0.687627, 2032: 0.688861, 2033: 0.691640,
}
FIVE_FOLD_ALT = {
    2026: 0.688584, 2027: 0.687396, 2028: 0.688310, 2029: 0.685804,
    2030: 0.689157, 2031: 0.689947, 2032: 0.688518, 2033: 0.691585,
}


def rank01(a: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(a, dtype=float)) / float(len(a))


def nested_auc(oof: np.ndarray, y: np.ndarray, n_blocks: int = 5) -> float:
    out = np.zeros(len(y))
    for block in np.array_split(np.arange(len(y)), n_blocks):
        out[block] = rankdata(oof[block]) / float(len(block))
    return float(roc_auc_score(y, out))


def seed_path(arm: str, seed: int) -> Path:
    return CKPT / f"{arm}_f{N_FOLD}_s{seed}.npz"


def load_xy() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).to_numpy()
    return train, test, y


def fit_one_seed(arm: str, seed: int, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray) -> dict:
    out_p = seed_path(arm, seed)
    if out_p.is_file():
        z = np.load(out_p)
        rec = {
            "arm": arm,
            "seed": seed,
            "auc": float(z["auc"]),
            "elapsed_s": float(z["elapsed_s"]) if "elapsed_s" in z.files else 0.0,
            "skipped": True,
        }
        print(f"[{arm} s{seed}] skip existing OOF={rec['auc']:.6f}", flush=True)
        return rec

    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    stream = ALL_SEEDS.index(seed) + 1
    t0 = time.time()
    if arm == "main":
        edges = fit_edges(raw_all)
        X, cats = catboost_frame(raw_all, edges, stream_offset=stream, n_views=4)
        params = dict(
            loss_function="Logloss",
            learning_rate=0.03,
            l2_leaf_reg=10,
            random_strength=0.7,
            verbose=False,
            thread_count=THREADS,
            allow_writing_files=False,
            depth=5,
            iterations=ITERS,
            boosting_type="Ordered",
        )
    elif arm == "alt":
        edges = fit_edges_alt(raw_all)
        X, cats = altboost_frame(raw_all, edges, stream_offset=stream, n_views=3)
        spec = ARMS["cat_alt"]
        params = dict(CAT_BASE)
        params.update({"l2_leaf_reg": 6, "one_hot_max_size": 12, "thread_count": THREADS})
        params.update({"depth": spec["depth"], "iterations": spec["iterations"]})
    else:
        raise ValueError(arm)

    Xtr = X.iloc[: len(train)].reset_index(drop=True)
    Xte = X.iloc[len(train) :].reset_index(drop=True)
    oof = np.zeros(len(y), dtype=np.float64)
    te = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=seed)
    for fold, (ti, vi) in enumerate(skf.split(Xtr, y)):
        ft = time.time()
        model = CatBoostClassifier(**params, random_seed=seed + fold)
        model.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
        oof[vi] = model.predict_proba(Xtr.iloc[vi])[:, 1]
        te += model.predict_proba(Xte)[:, 1] / float(N_FOLD)
        print(
            f"  [{arm} s{seed} f{fold}] {time.time() - ft:.0f}s  "
            f"fold_rows={len(vi)}",
            flush=True,
        )

    auc = float(roc_auc_score(y, oof))
    elapsed = time.time() - t0
    np.savez(
        out_p,
        oof=oof,
        test_pred=te,
        auc=auc,
        seed=seed,
        arm=arm,
        elapsed_s=elapsed,
        y=y,
        n_fold=N_FOLD,
        iters=ITERS,
    )
    print(f"[{arm} s{seed}] OOF={auc:.6f} ({elapsed:.0f}s) -> {out_p.name}", flush=True)
    return {"arm": arm, "seed": seed, "auc": auc, "elapsed_s": elapsed, "skipped": False}


def pool_arm(arm: str, y: np.ndarray) -> dict | None:
    paths = sorted(CKPT.glob(f"{arm}_f{N_FOLD}_s*.npz"))
    if not paths:
        return None
    oofs, tes, seeds, aucs = [], [], [], []
    for p in paths:
        z = np.load(p)
        oofs.append(rank01(z["oof"]))
        tes.append(rank01(z["test_pred"]))
        seeds.append(int(z["seed"]))
        aucs.append(float(z["auc"]))
    oof = rank01(np.mean(np.vstack(oofs), axis=0))
    te = rank01(np.mean(np.vstack(tes), axis=0))
    return {
        "arm": arm,
        "n_seed": len(seeds),
        "seeds": seeds,
        "per_seed": aucs,
        "oof": oof,
        "test": te,
        "pool_auc": float(roc_auc_score(y, oof)),
        "per_seed_mean": float(np.mean(aucs)),
    }


def write_report(y: np.ndarray) -> dict:
    main = pool_arm("main", y)
    alt = pool_arm("alt", y)
    report: dict = {
        "protocol": f"HONEST Classifier Logloss, {N_FOLD}-fold, no ES, opus5 FE",
        "w62_target": W62_OOF,
        "opus5_5fold_max2_full": TARGET_FULL,
        "n_fold": N_FOLD,
        "iters": ITERS,
        "threads": THREADS,
    }
    if main:
        dlt = []
        for s, a in zip(main["seeds"], main["per_seed"]):
            base = FIVE_FOLD_MAIN.get(int(s))
            dlt.append(None if base is None else round(a - base, 6))
        report["main"] = {
            "n_seed": main["n_seed"],
            "seeds": main["seeds"],
            "per_seed": [round(a, 6) for a in main["per_seed"]],
            "delta_vs_5fold_same_seed": dlt,
            "pool_auc": main["pool_auc"],
            "per_seed_mean": main["per_seed_mean"],
        }
        np.savez(ART / "honest10_main.npz", oof=main["oof"], test_pred=main["test"], y=y,
                 seeds=np.array(main["seeds"]), per_seed=np.array(main["per_seed"]))
    if alt:
        dlt = []
        for s, a in zip(alt["seeds"], alt["per_seed"]):
            base = FIVE_FOLD_ALT.get(int(s))
            dlt.append(None if base is None else round(a - base, 6))
        report["alt"] = {
            "n_seed": alt["n_seed"],
            "seeds": alt["seeds"],
            "per_seed": [round(a, 6) for a in alt["per_seed"]],
            "delta_vs_5fold_same_seed": dlt,
            "pool_auc": alt["pool_auc"],
            "per_seed_mean": alt["per_seed_mean"],
        }
        np.savez(ART / "honest10_alt.npz", oof=alt["oof"], test_pred=alt["test"], y=y,
                 seeds=np.array(alt["seeds"]), per_seed=np.array(alt["per_seed"]))
    if main and alt:
        mx = np.maximum(main["oof"], alt["oof"])
        te = np.maximum(main["test"], alt["test"])
        full = float(roc_auc_score(y, mx))
        nest = nested_auc(mx, y)
        report["max2_full"] = full
        report["max2_nested"] = nest
        n_both = len(set(main["seeds"]) & set(alt["seeds"]))
        report["n_seed_both"] = n_both
        report["comparable_to_opus5_8seed"] = n_both >= 8
        report["beats_w62"] = bool(n_both >= 8 and full >= W62_OOF)
        report["delta_vs_w62"] = full - W62_OOF if n_both >= 8 else None
        report["delta_vs_opus5_5fold"] = full - TARGET_FULL if n_both >= 8 else None
        np.savez(ART / "honest10_max2.npz", oof=mx, test_pred=te, y=y)
        print(
            f"\n[honest10] seeds={sorted(set(main['seeds']) & set(alt['seeds']))} "
            f"max2 full={full:.5f} nested={nest:.5f}  "
            f"vs W62 {W62_OOF:.5f} ({full - W62_OOF:+.5f})  "
            f"vs opus5-5fold {TARGET_FULL:.5f} ({full - TARGET_FULL:+.5f})",
            flush=True,
        )
    out = ART / "honest10_report.json"
    slim = {k: v for k, v in report.items()}
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(json.dumps(slim, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return report


def run(seeds: list[int], arms: list[str]) -> None:
    train, test, y = load_xy()
    print(
        f"[honest10] n={len(y)} pos={int(y.sum())} folds={N_FOLD} "
        f"seeds={seeds} arms={arms} threads={THREADS}",
        flush=True,
    )
    for seed in seeds:
        for arm in arms:
            fit_one_seed(arm, seed, train, test, y)
        write_report(y)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="train missing (arm, seed) checkpoints")
    p.add_argument("--report", action="store_true", help="pool existing checkpoints only")
    p.add_argument("--seeds", default=",".join(str(s) for s in ALL_SEEDS))
    p.add_argument("--arms", default="main,alt")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    if args.run:
        run(seeds, arms)
        return 0
    train, test, y = load_xy()
    write_report(y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
