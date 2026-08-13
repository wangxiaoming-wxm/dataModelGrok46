#!/usr/bin/env python3
"""Write submissions/submission.csv from the strongest available teacher parquet.

Applies the frozen insurer-vs-customer gate AFTER rank fusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/workspace")
SUB = ROOT / "submissions"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "analysis"))
from insurer_gate import apply_gate  # noqa: E402


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _auc_of(oof: pd.DataFrame) -> float:
    y = oof["label"].to_numpy()
    r_m = rank01(oof["pred_cb_main"].to_numpy())
    r_a = rank01(oof["pred_cb_alt"].to_numpy())
    w62 = 0.62 * r_m + 0.38 * r_a
    mx = np.maximum(r_m, r_a)
    return max(float(roc_auc_score(y, w62)), float(roc_auc_score(y, mx)))


def pick() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Prefer the parquet pack with higher honest rank-blend OOF, not merely 'w62 exists'.

    1-seed × 1-bag W62 is weaker than teacher 3-bag (0.683 vs 0.692). Blindly
    preferring cb_w62_*.parquet would downgrade the submission.
    """
    cands: list[tuple[str, Path, Path]] = []
    w62o, w62t = SUB / "cb_w62_oof.parquet", SUB / "cb_w62_test.parquet"
    if w62o.is_file() and w62t.is_file():
        cands.append(("cb_w62", w62o, w62t))
    cands.append(
        ("cb_teacher", SUB / "cb_teacher_oof.parquet", SUB / "cb_teacher_test.parquet")
    )
    best: tuple[pd.DataFrame, pd.DataFrame, str] | None = None
    best_auc = -1.0
    for tag, op, tp in cands:
        if not op.is_file() or not tp.is_file():
            continue
        oof = pd.read_parquet(op)
        tes = pd.read_parquet(tp)
        a = _auc_of(oof)
        print(f"candidate {tag} ungated_max_blend={a:.6f}", flush=True)
        if a > best_auc:
            best_auc = a
            best = (oof, tes, tag)
    if best is None:
        raise FileNotFoundError("no teacher parquet")
    return best


def main() -> None:
    oof, tes, tag = pick()
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "days"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id", "days"])
    oof = oof.copy()
    tes = tes.copy()
    oof["id"] = oof["id"].astype(str)
    tes["id"] = tes["id"].astype(str)
    train["id"] = train["id"].astype(str)
    test["id"] = test["id"].astype(str)
    oof = oof.merge(train, on="id", how="left")
    tes = tes.merge(test, on="id", how="left")
    y = oof["label"].to_numpy()
    r_m = rank01(oof["pred_cb_main"].to_numpy())
    r_a = rank01(oof["pred_cb_alt"].to_numpy())
    w62 = 0.62 * r_m + 0.38 * r_a
    mx = np.maximum(r_m, r_a)
    auc_w62 = float(roc_auc_score(y, w62))
    auc_mx = float(roc_auc_score(y, mx))
    use_max = auc_mx >= auc_w62
    base = mx if use_max else w62
    gated = apply_gate(base, oof["days"].to_numpy())
    auc_gated = float(roc_auc_score(y, gated))
    print(
        f"teacher={tag} oof_w62={auc_w62:.6f} oof_max2={auc_mx:.6f} "
        f"oof_gated={auc_gated:.6f} selected={'max2' if use_max else 'w62'}+insurer_gate"
    )

    tr_m = rank01(tes["pred_cb_main"].to_numpy())
    tr_a = rank01(tes["pred_cb_alt"].to_numpy())
    pred = np.maximum(tr_m, tr_a) if use_max else 0.62 * tr_m + 0.38 * tr_a
    tes["label"] = rank01(apply_gate(pred, tes["days"].to_numpy()))
    order = pd.read_csv(DATA / "test.csv", usecols=["id"])
    order["id"] = order["id"].astype(str)
    out = order.merge(tes[["id", "label"]], on="id", how="left")
    if out["label"].isna().any():
        out["label"] = out["label"].fillna(0.5)
    SUB.mkdir(parents=True, exist_ok=True)
    path = SUB / "submission.csv"
    out.to_csv(path, index=False)
    report = (
        f"sparkml_teacher_blend\n"
        f"teacher={tag}\n"
        f"n_test={len(out)}\n"
        f"auc_w62={auc_w62:.6f}\n"
        f"auc_max2={auc_mx:.6f}\n"
        f"auc_insurer_gate={auc_gated:.6f}\n"
        f"selected={'cb_max2' if use_max else 'cb_w62'}+insurer_gate\n"
        f"auc_blend={auc_gated:.6f}\n"
        f"gate=zero[1725,1825) rank-0.10[700,880) rank+0.05[9370,9475); submit=rank01(gated)\n"
        f"note=Spark ML Scala pipeline consumes teacher via CatBoostArm.joinTeacher / BlendApp\n"
    )
    (SUB / "oof_report_teacher.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {path} rows={len(out)}")


if __name__ == "__main__":
    main()
