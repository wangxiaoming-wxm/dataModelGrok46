#!/usr/bin/env python3
"""Assemble the strongest currently-available submission without retraining.

Reads (all optional except teacher parquet + test.csv):
  analysis/opus5/artifacts/{merger_ord8,v2_cat_alt8}.npz  # HONEST 8-seed max2 0.69993
  submissions/cb_teacher_{oof,test}.parquet
  analysis/final_ckpt/lgb_oof_test.npz
  analysis/final_ckpt/vales_seed*_bags*.npz  (averaged=1 only)

Picks the max gated OOF among a frozen SAFE candidate list.
Never element-wise max() two OOF vectors from different CV protocols
(that cherry-picks fold luck). Safe to rerun while final_best.py is training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace") if Path("/workspace/data/train.csv").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from insurer_gate import apply_gate  # noqa: E402

SUB = ROOT / "submissions"
CKPT = ROOT / "analysis" / "final_ckpt"
OPUS = ROOT / "analysis" / "opus5" / "artifacts"


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy(np.float64)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def pick_cb_fuse(oof_m: np.ndarray, oof_a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str, float]:
    r_m, r_a = rank01(oof_m), rank01(oof_a)
    w62 = 0.62 * r_m + 0.38 * r_a
    mx = np.maximum(r_m, r_a)
    a1, a2 = auc(y, w62), auc(y, mx)
    if a2 >= a1:
        return mx, "max2", a2
    return w62, "w62", a1


def apply_cb_fuse(te_m: np.ndarray, te_a: np.ndarray, tag: str) -> np.ndarray:
    r_m, r_a = rank01(te_m), rank01(te_a)
    return np.maximum(r_m, r_a) if tag == "max2" else 0.62 * r_m + 0.38 * r_a


def gauss(a: np.ndarray) -> np.ndarray:
    """Gaussian copula of a rank score. Borrowed from best_model_0.706 fusion, not their weights."""
    r = np.clip(rank01(a), 1e-4, 1.0 - 1e-4)
    return norm.ppf(r)


def gauss_blend(parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    s = np.zeros(len(parts[0][1]), dtype=np.float64)
    for w, arr in parts:
        s += w * gauss(arr)
    return rank01(s)


def nested_auc(oof: np.ndarray, y: np.ndarray, n_blocks: int = 5) -> float:
    """Opus5 anti-optimistic metric: re-rank within contiguous index blocks."""
    n = len(y)
    out = np.zeros(n)
    for b in np.array_split(np.arange(n), n_blocks):
        out[b] = pd.Series(oof[b]).rank(method="average", pct=True).to_numpy(np.float64)
    return auc(y, out)


def load_opus5() -> tuple[np.ndarray, np.ndarray, dict] | None:
    mo_p, ca_p = OPUS / "merger_ord8.npz", OPUS / "v2_cat_alt8.npz"
    if not (mo_p.is_file() and ca_p.is_file()):
        return None
    mo, ca = np.load(mo_p, allow_pickle=True), np.load(ca_p, allow_pickle=True)
    mo_oof, mo_te = rank01(np.asarray(mo["oof"], dtype=np.float64)), rank01(np.asarray(mo["test_pred"], dtype=np.float64))
    ca_oof, ca_te = rank01(np.asarray(ca["oof"], dtype=np.float64)), rank01(np.asarray(ca["test_pred"], dtype=np.float64))
    oof, tes = np.maximum(mo_oof, ca_oof), np.maximum(mo_te, ca_te)
    meta = {
        "merger_auc": float(auc(np.asarray(mo["y"], dtype=np.float64), mo_oof)) if "y" in mo.files else None,
        "alt_auc": float(auc(np.asarray(ca["y"], dtype=np.float64), ca_oof)) if "y" in ca.files else None,
    }
    return oof, tes, meta


def load_vales() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
    parts_m, parts_a, tes_m, tes_a, seeds = [], [], [], [], []
    for p in sorted(CKPT.glob("vales_seed*_bags*.npz")):
        z = np.load(p)
        if "averaged" not in z.files or int(z["averaged"]) != 1:
            continue
        if np.std(z["oof_m"]) < 1e-12:
            continue
        parts_m.append(z["oof_m"].astype(np.float64))
        parts_a.append(z["oof_a"].astype(np.float64))
        tes_m.append(z["te_m"].astype(np.float64))
        tes_a.append(z["te_a"].astype(np.float64))
        name = p.stem
        seed = int(name.split("seed")[1].split("_")[0])
        seeds.append(seed)
    if not parts_m:
        return None
    return (
        np.mean(parts_m, axis=0),
        np.mean(parts_a, axis=0),
        np.mean(tes_m, axis=0),
        np.mean(tes_a, axis=0),
        seeds,
    )


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=["id", "label", "days"])
    test = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id", "days"])
    train["id"] = train["id"].astype(str)
    test["id"] = test["id"].astype(str)
    y = train["label"].to_numpy(np.float64)
    days_tr = train["days"].to_numpy(np.float64)
    days_te = test["days"].to_numpy(np.float64)

    tea_oof = pd.read_parquet(SUB / "cb_teacher_oof.parquet")
    tea_te = pd.read_parquet(SUB / "cb_teacher_test.parquet")
    tea_oof["id"] = tea_oof["id"].astype(str)
    tea_te["id"] = tea_te["id"].astype(str)
    tea_oof = train[["id"]].merge(tea_oof, on="id", how="left")
    tea_te = test[["id"]].merge(tea_te, on="id", how="left")
    tea_s, tea_tag, tea_auc = pick_cb_fuse(
        tea_oof["pred_cb_main"].to_numpy(), tea_oof["pred_cb_alt"].to_numpy(), y
    )
    tea_t = apply_cb_fuse(tea_te["pred_cb_main"].to_numpy(), tea_te["pred_cb_alt"].to_numpy(), tea_tag)

    lgb_oof = lgb_te = None
    lgb_auc = None
    lgb_p = CKPT / "lgb_oof_test.npz"
    if lgb_p.is_file():
        z = np.load(lgb_p)
        lgb_oof = rank01(np.asarray(z["oof"], dtype=np.float64))
        lgb_te = rank01(np.asarray(z["tes"], dtype=np.float64))
        lgb_auc = auc(y, lgb_oof)

    vales = load_vales()
    cb_oof = cb_te = None
    cb_tag, cb_auc, seeds = None, None, []
    if vales is not None:
        oof_m, oof_a, te_m, te_a, seeds = vales
        cb_oof, cb_tag, cb_auc = pick_cb_fuse(oof_m, oof_a, y)
        cb_te = apply_cb_fuse(te_m, te_a, cb_tag)

    opus = load_opus5()
    opus_oof = opus_te = None
    opus_meta: dict = {}
    if opus is not None:
        opus_oof, opus_te, opus_meta = opus

    cands: list[tuple[str, np.ndarray, np.ndarray]] = [
        (f"teacher_{tea_tag}", tea_s, tea_t),
    ]
    if lgb_oof is not None:
        cands += [
            ("lgb", lgb_oof, lgb_te),
            ("0.80teacher+0.20lgb", 0.80 * rank01(tea_s) + 0.20 * lgb_oof, 0.80 * rank01(tea_t) + 0.20 * lgb_te),
            ("0.85teacher+0.15lgb", 0.85 * rank01(tea_s) + 0.15 * lgb_oof, 0.85 * rank01(tea_t) + 0.15 * lgb_te),
            ("0.90teacher+0.10lgb", 0.90 * rank01(tea_s) + 0.10 * lgb_oof, 0.90 * rank01(tea_t) + 0.10 * lgb_te),
        ]
    if cb_oof is not None:
        r_cb, r_cbt = rank01(cb_oof), rank01(cb_te)
        r_tea, r_teat = rank01(tea_s), rank01(tea_t)
        cands += [
            (f"vales_{cb_tag}", cb_oof, cb_te),
            ("0.70vales+0.30teacher", 0.70 * r_cb + 0.30 * r_tea, 0.70 * r_cbt + 0.30 * r_teat),
        ]
        if lgb_oof is not None:
            cands += [
                ("0.80vales+0.20lgb", 0.80 * r_cb + 0.20 * lgb_oof, 0.80 * r_cbt + 0.20 * lgb_te),
                ("0.85vales+0.15lgb", 0.85 * r_cb + 0.15 * lgb_oof, 0.85 * r_cbt + 0.15 * lgb_te),
            ]
    if opus_oof is not None:
        r_op, r_opt = rank01(opus_oof), rank01(opus_te)
        cands += [
            ("opus_max2", opus_oof, opus_te),
        ]
        if lgb_oof is not None:
            cands += [
                ("0.90opus+0.10lgb", 0.90 * r_op + 0.10 * lgb_oof, 0.90 * r_opt + 0.10 * lgb_te),
                ("0.85opus+0.15lgb", 0.85 * r_op + 0.15 * lgb_oof, 0.85 * r_opt + 0.15 * lgb_te),
                ("0.80opus+0.20lgb", 0.80 * r_op + 0.20 * lgb_oof, 0.80 * r_opt + 0.20 * lgb_te),
            ]
        # Rank-weighted mix with other CB packs is OK; element-wise max across
        # different CV protocols is not (cherry-picks fold luck).
        if cb_oof is not None:
            cands.append(("0.90opus+0.10vales", 0.90 * r_op + 0.10 * rank01(cb_oof), 0.90 * r_opt + 0.10 * rank01(cb_te)))
            cands.append(
                (
                    "gauss_0.90opus+0.10vales",
                    gauss_blend([(0.90, r_op), (0.10, rank01(cb_oof))]),
                    gauss_blend([(0.90, r_opt), (0.10, rank01(cb_te))]),
                )
            )
        cands.append(("0.90opus+0.10teacher", 0.90 * r_op + 0.10 * rank01(tea_s), 0.90 * r_opt + 0.10 * rank01(tea_t)))
        if lgb_oof is not None:
            cands += [
                (
                    "gauss_0.85opus+0.15lgb",
                    gauss_blend([(0.85, r_op), (0.15, lgb_oof)]),
                    gauss_blend([(0.85, r_opt), (0.15, lgb_te)]),
                ),
                (
                    "gauss_0.90opus+0.10lgb",
                    gauss_blend([(0.90, r_op), (0.10, lgb_oof)]),
                    gauss_blend([(0.90, r_opt), (0.10, lgb_te)]),
                ),
            ]
            if cb_oof is not None:
                cands.append(
                    (
                        "gauss_0.80opus+0.15lgb+0.05vales",
                        gauss_blend([(0.80, r_op), (0.15, lgb_oof), (0.05, rank01(cb_oof))]),
                        gauss_blend([(0.80, r_opt), (0.15, lgb_te), (0.05, rank01(cb_te))]),
                    )
                )

    scored = []
    for name, oof, tes in cands:
        g = auc(y, rank01(apply_gate(oof, days_tr)))
        n = nested_auc(rank01(oof), y)
        scored.append((g, name, oof, tes, n))
        print(f"  {name:32s} gated={g:.6f} nested={n:.6f} ungated={auc(y, oof):.6f}", flush=True)
    best_g, best_name, best_oof, best_te, best_n = max(scored, key=lambda t: t[0])
    gated_te = rank01(apply_gate(best_te, days_te))
    order = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])
    order["id"] = order["id"].astype(str)
    out = order.merge(pd.DataFrame({"id": test["id"], "label": gated_te}), on="id", how="left")
    out["label"] = out["label"].fillna(0.5)
    out.to_csv(SUB / "submission.csv", index=False)

    report = {
        "recipe": "opus5 HONEST 8-seed max2 (Classifier Logloss, no ES) + frozen 4-window gate",
        "selected": best_name,
        "auc_gated": best_g,
        "auc_nested": best_n,
        "opus_merger_auc": opus_meta.get("merger_auc"),
        "opus_alt_auc": opus_meta.get("alt_auc"),
        "teacher_tag": tea_tag,
        "teacher_ungated": tea_auc,
        "lgb_ungated": lgb_auc,
        "cb_tag": cb_tag,
        "cb_ungated": cb_auc,
        "vales_seeds": seeds,
        "n_test": int(len(out)),
        "gate": "floor[1725,1825)+[2110,2210) -0.10[700,880) +0.05[9370,9475)",
        "honest_note": "opus5 is HONEST_NO_ES 5fold x 8seed; VAL-ES mixes are slightly optimistic; no cross-protocol elementwise max",
        "status": "assembled",
    }
    (SUB / "final_best_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (SUB / "final_best_metrics.json").write_text(
        json.dumps(
            {
                "auc_blend": best_g,
                "auc_gated": best_g,
                "auc_cb_main": float(cb_auc or tea_auc),
                "auc_cb_w62": best_g,
                "selected": best_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    txt = "\n".join(f"{k}={v}" for k, v in report.items() if not isinstance(v, (list, dict)))
    (SUB / "oof_report.txt").write_text(txt + "\n", encoding="utf-8")

    fuse_oof = rank01(best_oof)
    fuse_te = rank01(best_te)
    pd.DataFrame(
        {
            "id": train["id"],
            "label": y,
            "pred_cb_main": fuse_oof,
            "pred_cb_alt": fuse_oof,
            "pred_fuse": fuse_oof,
            "pred_lgb": lgb_oof if lgb_oof is not None else fuse_oof,
        }
    ).to_parquet(SUB / "final_best_oof.parquet", index=False)
    pd.DataFrame(
        {
            "id": test["id"],
            "pred_cb_main": fuse_te,
            "pred_cb_alt": fuse_te,
            "pred_fuse": fuse_te,
            "pred_lgb": lgb_te if lgb_te is not None else fuse_te,
        }
    ).to_parquet(SUB / "final_best_test.parquet", index=False)
    print(
        f"wrote {SUB / 'submission.csv'} selected={best_name} gated={best_g:.6f} nested={best_n:.6f} n={len(out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
