#!/usr/bin/env python3
"""Write submissions/submission.csv from the strongest available teacher parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SUB = Path("/workspace/submissions")
DATA = Path("/workspace/data")


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def pick() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    w62o, w62t = SUB / "cb_w62_oof.parquet", SUB / "cb_w62_test.parquet"
    if w62o.is_file() and w62t.is_file():
        return pd.read_parquet(w62o), pd.read_parquet(w62t), "cb_w62"
    return (
        pd.read_parquet(SUB / "cb_teacher_oof.parquet"),
        pd.read_parquet(SUB / "cb_teacher_test.parquet"),
        "cb_teacher",
    )


def main() -> None:
    oof, tes, tag = pick()
    y = oof["label"].to_numpy()
    r_m = rank01(oof["pred_cb_main"].to_numpy())
    r_a = rank01(oof["pred_cb_alt"].to_numpy())
    w62 = 0.62 * r_m + 0.38 * r_a
    mx = np.maximum(r_m, r_a)
    auc_w62 = float(roc_auc_score(y, w62))
    auc_mx = float(roc_auc_score(y, mx))
    use_max = auc_mx >= auc_w62
    print(f"teacher={tag} oof_w62={auc_w62:.6f} oof_max2={auc_mx:.6f} selected={'max2' if use_max else 'w62'}")

    tr_m = rank01(tes["pred_cb_main"].to_numpy())
    tr_a = rank01(tes["pred_cb_alt"].to_numpy())
    pred = np.maximum(tr_m, tr_a) if use_max else 0.62 * tr_m + 0.38 * tr_a
    tes = tes.copy()
    tes["id"] = tes["id"].astype(str)
    tes["label"] = pred
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
        f"selected={'cb_max2' if use_max else 'cb_w62'}\n"
        f"auc_blend={auc_mx if use_max else auc_w62:.6f}\n"
        f"note=Spark ML Scala pipeline consumes this teacher via CatBoostArm.joinTeacher / BlendApp\n"
    )
    (SUB / "oof_report_teacher.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {path} rows={len(out)}")


if __name__ == "__main__":
    main()
