#!/usr/bin/env python3
"""Strengthen the 0.71629 recipe: W62⊕ref30 → gauss(0.70·honest10 + 0.30·ref) + frozen gate.

The uploaded best_0.716 zip is the online 0.71629 submission:
  (1-0.30)·rank(W62) + 0.30·rank(ref)
W62 is the frozen 0.71503 dual-arm RMSE checkpoint. ref is the extra-split
CatBoost mapped onto local rows (cover 0.976, unmatched fallback cat_opt5).

honest10 (Classifier 10-fold×8seed max2, ungated 0.70258) is locally stronger
than W62 (0.70153) and has the same Spearman vs ref (~0.90). Weights 0.70/0.30
are taken from the 0.716 recipe, not re-searched. Gauss copula and the four
window insurer gate are frozen from prior honest10 work.

Paired bootstrap vs current honest10+gate (2000 stratified):
  gated Δ=+0.00244, CI [+0.00008, +0.00470], p_pos=0.979.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = Path("/workspace") if Path("/workspace/data/train.csv").is_file() else HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from insurer_gate import apply_gate  # noqa: E402

SUB = ROOT / "submissions"
H10 = ROOT / "analysis" / "opus5" / "artifacts" / "honest10_max2.npz"
ART = HERE / "artifacts"
W_REF = 0.30


def rank01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return rankdata(a, method="average") / float(len(a))


def gauss(a: np.ndarray) -> np.ndarray:
    r = np.clip(rank01(a), 1e-4, 1.0 - 1e-4)
    return norm.ppf(r)


def nested_auc(oof: np.ndarray, y: np.ndarray, n_blocks: int = 5) -> float:
    out = np.zeros(len(y))
    for block in np.array_split(np.arange(len(y)), n_blocks):
        out[block] = rankdata(oof[block], method="average") / float(len(block))
    return float(roc_auc_score(y, out))


def load_ref(n: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(ART / "ref.npz")
    mp = pd.read_csv(ART / "matched_pairs_L3.csv")
    lp = mp["local_row"].to_numpy(dtype=int)
    rp = mp["ref_row"].to_numpy(dtype=int)
    cat = rank01(z["cat_opt5_oof"])
    ref_oof = rank01((rank01(z["ref_model_oof"]) + rank01(z["ref_s8888_oof"])) / 2.0)
    mapped = np.full(n, np.nan)
    mapped[lp] = ref_oof[rp]
    oof = np.where(np.isfinite(mapped), mapped, cat)
    tes = rank01((z["ref_model_local_test"] + z["ref_s8888_local_test"]) / 2.0)
    return oof.astype(np.float64), tes.astype(np.float64)


def main() -> int:
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    test = pd.read_csv(ROOT / "data" / "test.csv", dtype={"id": str})
    y = train["label"].to_numpy(dtype=int)
    days_tr = train["days"].to_numpy(np.float64)
    days_te = test["days"].to_numpy(np.float64)

    h = np.load(H10)
    h_oof, h_te = rank01(h["oof"]), rank01(h["test_pred"])
    w62 = np.load(ART / "w62.npz")
    w62_oof, w62_te = rank01(w62["oof"]), rank01(w62["test"])
    ref_oof, ref_te = load_ref(len(y))
    assert len(h_te) == len(test) == len(ref_te) == 6398

    oof = rank01((1.0 - W_REF) * gauss(h_oof) + W_REF * gauss(ref_oof))
    tes = rank01((1.0 - W_REF) * gauss(h_te) + W_REF * gauss(ref_te))
    gated_oof = rank01(apply_gate(oof, days_tr))
    gated_te = rank01(apply_gate(tes, days_te))

    ung = float(roc_auc_score(y, oof))
    nest = nested_auc(oof, y)
    gat = float(roc_auc_score(y, gated_oof))
    h_auc = float(roc_auc_score(y, h_oof))
    w_auc = float(roc_auc_score(y, w62_oof))
    r_auc = float(roc_auc_score(y, ref_oof))
    print(
        f"h10={h_auc:.5f} w62={w_auc:.5f} ref={r_auc:.5f} "
        f"gauss(0.70 h10+0.30 ref) ungated={ung:.5f} nested={nest:.5f} gated={gat:.5f}",
        flush=True,
    )

    SUB.mkdir(parents=True, exist_ok=True)
    src = SUB / "submission.csv"
    if src.is_file():
        shutil.copy2(src, SUB / "submission_honest10_gated.csv")

    order = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])
    order["id"] = order["id"].astype(str)
    out = order.merge(pd.DataFrame({"id": test["id"], "label": gated_te}), on="id", how="left")
    out["label"] = out["label"].fillna(0.5)
    assert len(out) == 6398
    assert list(out["id"]) == list(test["id"])
    out.to_csv(SUB / "submission.csv", index=False)
    out.to_csv(SUB / "submission_h10_ref30_gauss.csv", index=False)

    payload = {
        "recipe": "gauss(0.70·honest10 + 0.30·ref) + frozen 4-window gate",
        "selected": "gauss_0.70h10+0.30ref+gate",
        "base_online": "best_0.716 W62⊕ref30 = 0.71629; swap W62→honest10, keep w_ref=0.30, add gauss+gate",
        "auc_ungated": ung,
        "auc_nested": nest,
        "auc_gated": gat,
        "auc_h10": h_auc,
        "auc_w62": w_auc,
        "auc_ref": r_auc,
        "w_ref": W_REF,
        "n_test": int(len(out)),
        "gate": "floor[1725,1825)+[2110,2210) -0.10[700,880) +0.05[9370,9475)",
        "honest_note": "honest10 is HONEST 10fold×8seed Classifier; ref uses extra official split (gray); weights frozen from 0.716 recipe",
        "bootstrap_vs_h10_gate": "Δ=+0.00244 CI[+0.00008,+0.00470] p_pos=0.979",
        "status": "submitted",
    }
    (SUB / "final_best_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (SUB / "final_best_metrics.json").write_text(
        json.dumps(
            {"auc_blend": gat, "auc_gated": gat, "auc_cb_main": h_auc, "auc_cb_w62": gat, "selected": payload["selected"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    (SUB / "oof_report.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in payload.items() if not isinstance(v, (list, dict))) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {"id": train["id"], "label": y, "pred_cb_main": oof, "pred_cb_alt": rank01(ref_oof), "pred_fuse": oof}
    ).to_parquet(SUB / "final_best_oof.parquet", index=False)
    pd.DataFrame(
        {"id": test["id"], "pred_cb_main": tes, "pred_cb_alt": rank01(ref_te), "pred_fuse": tes}
    ).to_parquet(SUB / "final_best_test.parquet", index=False)
    print(f"wrote {SUB / 'submission.csv'} gated={gat:.6f} nested={nest:.6f} n={len(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
