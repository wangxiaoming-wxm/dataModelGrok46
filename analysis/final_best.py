#!/usr/bin/env python3
"""Strongest currently-justified submission pipeline.

Recipe (all nested / fold-safe, RMSE only):
  1. CatBoost dual-world VAL-ES  — Ordered d5 l2=10 + Plain d6 l2=6 rsm=0.3
     10-fold × 3-bag × N-seed, early stopping on the OOF val fold
     (historical W62 protocol that reached 0.70159 / online 0.71503).
     23 medium-card cats, no src|cond_q|days_q.
  2. LightGBM dual-world RMSE, same folds, outer ES (diversity arm).
  3. Rank-fuse: pick max2 vs 0.62/0.38 on CatBoost; then
     0.80*rank(CB) + 0.20*rank(LGB) if it beats CB on OOF; else CB.
     Also compare the existing teacher 3-bag pack and keep the winner.
  4. Frozen insurer gates AFTER rank fusion:
       floor [1725,1825) and [2110,2210); −0.10 [700,880); +0.05 [9370,9475)
     then rank01 → submissions/submission.csv

Checkpoints under analysis/final_ckpt/. Resumable per seed/fold.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/workspace") if Path("/workspace/data/train.csv").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from cb_features import (  # noqa: E402
    CATS_BEST,
    NUM_ALT_BEST,
    NUM_MAIN_BEST,
    fold_features,
    load_raw,
)
from insurer_gate import apply_gate  # noqa: E402

SUB = ROOT / "submissions"
CKPT = ROOT / "analysis" / "final_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)

NFOLD = 10
N_BAG = int(os.environ.get("FINAL_BAGS", "3"))
ITERS = 800
THREADS = int(os.environ.get("FINAL_THREADS", "1"))
SEEDS = [int(x) for x in os.environ.get("FINAL_SEEDS", "2026,2036,2046,2056").split(",") if x.strip()]
CATS = list(CATS_BEST)
NUM_MAIN = list(dict.fromkeys(NUM_MAIN_BEST))
NUM_ALT = list(dict.fromkeys(NUM_ALT_BEST))


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy(np.float64)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing features: {missing}")
    out = df[cols].copy()
    for c in CATS:
        if c in out.columns:
            out[c] = out[c].astype(str)
    for c in cols:
        if c in CATS:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def cb_params(arm: str, seed: int) -> dict:
    base = dict(
        loss_function="RMSE",
        iterations=ITERS,
        learning_rate=0.03,
        random_seed=int(seed),
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=THREADS,
        verbose=False,
        use_best_model=True,
    )
    if arm == "main":
        base.update(boosting_type="Ordered", depth=5, l2_leaf_reg=10, rsm=1.0)
    else:
        base.update(boosting_type="Plain", depth=6, l2_leaf_reg=6, rsm=0.3)
    return base


def fit_vales(arm: str, seed: int, Xtr, ytr, Xva, yva, Xte) -> tuple[np.ndarray, np.ndarray, int]:
    cols = (NUM_MAIN if arm == "main" else NUM_ALT) + CATS
    tr = _prep(Xtr, cols)
    va = _prep(Xva, cols)
    te = _prep(Xte, cols)
    model = CatBoostRegressor(**cb_params(arm, seed))
    model.fit(Pool(tr, ytr, cat_features=CATS), eval_set=Pool(va, yva, cat_features=CATS))
    it = int(model.best_iteration_ or 0)
    return (
        model.predict(Pool(va, cat_features=CATS)).astype(np.float64),
        model.predict(Pool(te, cat_features=CATS)).astype(np.float64),
        it,
    )


def seed_ckpt(seed: int) -> Path:
    return CKPT / f"vales_seed{seed}_bags{N_BAG}.npz"


def run_cb_seed(seed: int, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, folds) -> dict:
    n, nt = len(train), len(test)
    ck = seed_ckpt(seed)
    oof_m = np.zeros(n)
    oof_a = np.zeros(n)
    te_m = np.zeros(nt)
    te_a = np.zeros(nt)
    start = 0
    if ck.exists():
        z = np.load(ck)
        oof_m, oof_a = z["oof_m"].astype(np.float64), z["oof_a"].astype(np.float64)
        te_m, te_a = z["te_m"].astype(np.float64), z["te_a"].astype(np.float64)
        if int(z["averaged"]) == 1:
            start = NFOLD
            print(f"  resume CB seed={seed} already complete", flush=True)
        else:
            start = int(z["last_fold"]) + 1
            print(f"  resume CB seed={seed} from fold {start} (running te sum)", flush=True)
    t0 = time.time()
    for fi, (tr_i, va_i) in enumerate(folds):
        if fi < start:
            continue
        trn, val, tes = fold_features(train.iloc[tr_i].copy(), train.iloc[va_i].copy(), test.copy())
        ytr, yva = y[tr_i].astype(float), y[va_i].astype(float)
        pm = np.zeros(len(va_i))
        pa = np.zeros(len(va_i))
        tm = np.zeros(nt)
        ta = np.zeros(nt)
        itA = itB = 0
        for b in range(N_BAG):
            p, t, it = fit_vales("main", seed + 10 * b + fi, trn, ytr, val, yva, tes)
            pm += p
            tm += t
            itA += it
            p, t, it = fit_vales("alt", seed + 100 + 10 * b + fi, trn, ytr, val, yva, tes)
            pa += p
            ta += t
            itB += it
        oof_m[va_i] = pm / N_BAG
        oof_a[va_i] = pa / N_BAG
        te_m += tm / N_BAG
        te_a += ta / N_BAG
        np.savez_compressed(ck, oof_m=oof_m, oof_a=oof_a, te_m=te_m, te_a=te_a, last_fold=fi, averaged=0)
        print(
            f"  CB seed={seed} fold={fi} A={auc(yva, oof_m[va_i]):.4f} "
            f"B={auc(yva, oof_a[va_i]):.4f} it~{itA/N_BAG:.0f}/{itB/N_BAG:.0f}",
            flush=True,
        )
    if start < NFOLD:
        te_m /= NFOLD
        te_a /= NFOLD
        np.savez_compressed(ck, oof_m=oof_m, oof_a=oof_a, te_m=te_m, te_a=te_a, last_fold=NFOLD - 1, averaged=1)
    rec = {
        "seed": seed,
        "oof_m": oof_m,
        "oof_a": oof_a,
        "te_m": te_m,
        "te_a": te_a,
        "auc_main": auc(y, oof_m),
        "auc_alt": auc(y, oof_a),
        "auc_max2": auc(y, np.maximum(rank01(oof_m), rank01(oof_a))),
        "auc_w62": auc(y, 0.62 * rank01(oof_m) + 0.38 * rank01(oof_a)),
        "elapsed_s": time.time() - t0,
    }
    print(
        f"SEED {seed} DONE main={rec['auc_main']:.5f} alt={rec['auc_alt']:.5f} "
        f"max2={rec['auc_max2']:.5f} w62={rec['auc_w62']:.5f} {rec['elapsed_s']:.0f}s",
        flush=True,
    )
    return rec


def run_lgb(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, folds) -> dict:
    ck = CKPT / "lgb_oof_test.npz"
    n, nt = len(train), len(test)
    if ck.exists():
        z = np.load(ck)
        print("resume LGB checkpoint", flush=True)
        return {k: z[k] for k in z.files}
    oof1 = np.zeros(n)
    oof2 = np.zeros(n)
    te1 = np.zeros(nt)
    te2 = np.zeros(nt)
    lgb_cats = ["source", "region", "age_range", "grades", "month", "src_reg", "src_cq", "src_dq5", "reg_age"]
    for fi, (tr_i, va_i) in enumerate(folds):
        trn, val, tes = fold_features(train.iloc[tr_i].copy(), train.iloc[va_i].copy(), test.copy())
        cols = NUM_MAIN + lgb_cats
        Xtr = _prep(trn, cols)
        Xva = _prep(val, cols)
        Xte = _prep(tes, cols)
        for c in lgb_cats:
            seen = ["__unk__"] + sorted({str(v) for v in Xtr[c].tolist()})
            Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=seen)
            Xva[c] = pd.Categorical(
                Xva[c].astype(str).where(Xva[c].astype(str).isin(seen), "__unk__"), categories=seen
            )
            Xte[c] = pd.Categorical(
                Xte[c].astype(str).where(Xte[c].astype(str).isin(seen), "__unk__"), categories=seen
            )
        ytr = y[tr_i].astype(float)
        yva = y[va_i].astype(float)
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=lgb_cats, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, categorical_feature=lgb_cats, reference=dtr, free_raw_data=False)
        p1 = dict(
            objective="regression", metric="rmse", learning_rate=0.03, num_leaves=24, max_depth=5,
            min_data_in_leaf=90, feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=10.0, verbose=-1, num_threads=THREADS, seed=fi, feature_pre_filter=False,
        )
        p2 = dict(
            objective="regression", metric="rmse", learning_rate=0.03, num_leaves=40, max_depth=6,
            min_data_in_leaf=70, feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=6.0, verbose=-1, num_threads=THREADS, seed=100 + fi, feature_pre_filter=False,
        )
        m1 = lgb.train(p1, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
        m2 = lgb.train(p2, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
        oof1[va_i] = m1.predict(Xva)
        oof2[va_i] = m2.predict(Xva)
        te1 += m1.predict(Xte)
        te2 += m2.predict(Xte)
        print(f"  LGB fold={fi} a={auc(yva, oof1[va_i]):.4f} b={auc(yva, oof2[va_i]):.4f}", flush=True)
    te1 /= NFOLD
    te2 /= NFOLD
    fuse_oof = 0.62 * rank01(oof1) + 0.38 * rank01(oof2)
    fuse_te = 0.62 * rank01(te1) + 0.38 * rank01(te2)
    rec = dict(oof=fuse_oof, tes=fuse_te, oof1=oof1, oof2=oof2, te1=te1, te2=te2, auc=auc(y, fuse_oof))
    np.savez_compressed(ck, **rec)
    print(f"LGB DONE auc={rec['auc']:.5f}", flush=True)
    return rec


def load_teacher() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_parquet(SUB / "cb_teacher_oof.parquet")
    tes = pd.read_parquet(SUB / "cb_teacher_test.parquet")
    oof["id"] = oof["id"].astype(str)
    tes["id"] = tes["id"].astype(str)
    return oof, tes


def pick_cb_fuse(oof_m: np.ndarray, oof_a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str, float]:
    r_m, r_a = rank01(oof_m), rank01(oof_a)
    w62 = 0.62 * r_m + 0.38 * r_a
    mx = np.maximum(r_m, r_a)
    a1, a2 = auc(y, w62), auc(y, mx)
    if a2 >= a1:
        return mx, "max2", a2
    return w62, "w62", a1


def write_submission(ids_test: np.ndarray, score: np.ndarray, days_te: np.ndarray, report: dict) -> None:
    gated = rank01(apply_gate(score, days_te))
    order = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])
    order["id"] = order["id"].astype(str)
    out = order.merge(pd.DataFrame({"id": ids_test.astype(str), "label": gated}), on="id", how="left")
    out["label"] = out["label"].fillna(0.5)
    path = SUB / "submission.csv"
    out.to_csv(path, index=False)
    (SUB / "final_best_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics = {
        "auc_blend": float(report.get("auc_gated", 0.0)),
        "auc_gated": float(report.get("auc_gated", 0.0)),
        "auc_cb_main": float(report.get("cb_ungated", report.get("teacher_ungated", 0.0)) or 0.0),
        "auc_cb_w62": float(report.get("auc_gated", 0.0)),
        "selected": report.get("selected"),
        "protocol": report.get("honest_note", report.get("recipe")),
    }
    (SUB / "final_best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    txt = "\n".join(f"{k}={v}" for k, v in report.items() if not isinstance(v, (list, dict)))
    (SUB / "oof_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"wrote {path} n={len(out)}", flush=True)
    print(txt, flush=True)


def main() -> None:
    train, test = load_raw()
    train["id"] = train["id"].astype(str)
    test["id"] = test["id"].astype(str)
    y = train["label"].to_numpy(np.float64)
    days_tr = train["days"].to_numpy(np.float64)
    days_te = test["days"].to_numpy(np.float64)
    folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=2026).split(train, y.astype(int)))
    print(
        f"FINAL_BEST seeds={SEEDS} bags={N_BAG} threads={THREADS} cats={len(CATS)} "
        f"num_main={len(NUM_MAIN)} num_alt={len(NUM_ALT)}",
        flush=True,
    )

    print("=== LGB diversity arm ===", flush=True)
    lgb_rec = run_lgb(train, test, y, folds)
    lgb_oof = rank01(np.asarray(lgb_rec["oof"], dtype=np.float64))
    lgb_te = rank01(np.asarray(lgb_rec["tes"], dtype=np.float64))
    tea_oof_df, tea_te_df = load_teacher()
    tea_oof_df = train[["id"]].merge(tea_oof_df, on="id", how="left")
    tea_te_df = test[["id"]].merge(tea_te_df, on="id", how="left")
    tea_oof, tea_tag, tea_auc = pick_cb_fuse(
        tea_oof_df["pred_cb_main"].to_numpy(), tea_oof_df["pred_cb_alt"].to_numpy(), y
    )
    ttm, tta = rank01(tea_te_df["pred_cb_main"].to_numpy()), rank01(tea_te_df["pred_cb_alt"].to_numpy())
    tea_te = np.maximum(ttm, tta) if tea_tag == "max2" else 0.62 * ttm + 0.38 * tta
    # Interim: teacher ⊕ LGB + gate, so a submission exists before CatBoost finishes.
    interim = []
    for w in (1.0, 0.90, 0.85, 0.80):
        oof = w * rank01(tea_oof) + (1.0 - w) * lgb_oof
        tes = w * rank01(tea_te) + (1.0 - w) * lgb_te
        g = auc(y, rank01(apply_gate(oof, days_tr)))
        interim.append((f"teacher*{w}+lgb", oof, tes, g))
        print(f"  interim {interim[-1][0]:22s} gated={g:.6f}", flush=True)
    ib = max(interim, key=lambda t: t[3])
    # Do not clobber a stronger opus5/assembled submission.
    existing = SUB / "final_best_report.json"
    skip_interim = False
    if existing.is_file():
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
            skip_interim = float(prev.get("auc_gated") or 0.0) >= ib[3] - 1e-12
        except Exception:
            skip_interim = False
    if skip_interim:
        print(f"  skip interim write (existing gated >= {ib[3]:.6f})", flush=True)
    else:
        write_submission(
            test["id"].to_numpy(),
            ib[2],
            days_te,
            {
                "recipe": "interim teacher+LGB + 4-window gate (CatBoost VAL-ES still training)",
                "selected": ib[0],
                "auc_gated": ib[3],
                "teacher_ungated": tea_auc,
                "lgb_ungated": auc(y, lgb_oof),
                "status": "interim",
            },
        )

    print("=== CatBoost VAL-ES ===", flush=True)
    cb_parts = []
    for s in SEEDS:
        cb_parts.append(run_cb_seed(s, train, test, y, folds))

    oof_m = np.mean([p["oof_m"] for p in cb_parts], axis=0)
    oof_a = np.mean([p["oof_a"] for p in cb_parts], axis=0)
    te_m = np.mean([p["te_m"] for p in cb_parts], axis=0)
    te_a = np.mean([p["te_a"] for p in cb_parts], axis=0)
    cb_oof, cb_tag, cb_auc = pick_cb_fuse(oof_m, oof_a, y)
    r_tm, r_ta = rank01(te_m), rank01(te_a)
    cb_te = np.maximum(r_tm, r_ta) if cb_tag == "max2" else 0.62 * r_tm + 0.38 * r_ta

    tea_oof_df, tea_te_df = load_teacher()
    tea_oof_df = train[["id"]].merge(tea_oof_df, on="id", how="left")
    tea_te_df = test[["id"]].merge(tea_te_df, on="id", how="left")
    tea_oof, tea_tag, tea_auc = pick_cb_fuse(
        tea_oof_df["pred_cb_main"].to_numpy(), tea_oof_df["pred_cb_alt"].to_numpy(), y
    )
    ttm, tta = rank01(tea_te_df["pred_cb_main"].to_numpy()), rank01(tea_te_df["pred_cb_alt"].to_numpy())
    tea_te = np.maximum(ttm, tta) if tea_tag == "max2" else 0.62 * ttm + 0.38 * tta

    lgb_oof = rank01(np.asarray(lgb_rec["oof"], dtype=np.float64))
    lgb_te = rank01(np.asarray(lgb_rec["tes"], dtype=np.float64))
    lgb_auc = float(lgb_rec["auc"]) if "auc" in lgb_rec else auc(y, lgb_oof)

    cands: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    r_cb, r_tea = rank01(cb_oof), rank01(tea_oof)
    r_cbt, r_teat = rank01(cb_te), rank01(tea_te)
    for name, oof, tes in (
        (f"vales_{cb_tag}", cb_oof, cb_te),
        (f"teacher_{tea_tag}", tea_oof, tea_te),
        ("lgb", lgb_oof, lgb_te),
        ("0.80vales+0.20lgb", 0.80 * r_cb + 0.20 * lgb_oof, 0.80 * r_cbt + 0.20 * lgb_te),
        ("0.85vales+0.15lgb", 0.85 * r_cb + 0.15 * lgb_oof, 0.85 * r_cbt + 0.15 * lgb_te),
        ("0.80teacher+0.20lgb", 0.80 * r_tea + 0.20 * lgb_oof, 0.80 * r_teat + 0.20 * lgb_te),
        ("0.85teacher+0.15lgb", 0.85 * r_tea + 0.15 * lgb_oof, 0.85 * r_teat + 0.15 * lgb_te),
        ("0.90teacher+0.10lgb", 0.90 * r_tea + 0.10 * lgb_oof, 0.90 * r_teat + 0.10 * lgb_te),
        ("max(vales,teacher)", np.maximum(r_cb, r_tea), np.maximum(r_cbt, r_teat)),
        ("0.70vales+0.30teacher", 0.70 * r_cb + 0.30 * r_tea, 0.70 * r_cbt + 0.30 * r_teat),
        (
            "0.64vales+0.16teacher+0.20lgb",
            0.64 * r_cb + 0.16 * r_tea + 0.20 * lgb_oof,
            0.64 * r_cbt + 0.16 * r_teat + 0.20 * lgb_te,
        ),
    ):
        g = auc(y, rank01(apply_gate(oof, days_tr)))
        cands.append((name, oof, tes, g))
        print(f"  candidate {name:24s} gated_oof={g:.6f}", flush=True)

    best = max(cands, key=lambda t: t[3])
    report = {
        "recipe": "VAL-ES CatBoost 10fold x 3bag + LGB + teacher (side pack; assemble_best picks the winner)",
        "n_seed": len(cb_parts),
        "n_bag": N_BAG,
        "seeds": SEEDS[: len(cb_parts)],
        "cb_tag": cb_tag,
        "cb_ungated": cb_auc,
        "teacher_tag": tea_tag,
        "teacher_ungated": tea_auc,
        "lgb_ungated": lgb_auc,
        "selected": best[0],
        "auc_gated": best[3],
        "per_seed_max2": [p["auc_max2"] for p in cb_parts],
        "gate": "floor[1725,1825)+[2110,2210) -0.10[700,880) +0.05[9370,9475)",
        "honest_note": "VAL-ES OOF is slightly optimistic vs no-ES; gate magnitudes frozen",
    }
    (SUB / "vales_oof.parquet").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"id": train["id"], "label": y, "pred_cb_main": oof_m, "pred_cb_alt": oof_a, "pred_lgb": lgb_oof}
    ).to_parquet(SUB / "vales_oof.parquet", index=False)
    pd.DataFrame(
        {"id": test["id"], "pred_cb_main": te_m, "pred_cb_alt": te_a, "pred_lgb": lgb_te}
    ).to_parquet(SUB / "vales_test.parquet", index=False)
    (SUB / "vales_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"VAL-ES side pack gated={best[3]:.6f} selected={best[0]} — handing off to assemble_best", flush=True)
    from assemble_best import main as assemble_main  # noqa: WPS433
    assemble_main()


if __name__ == "__main__":
    main()
