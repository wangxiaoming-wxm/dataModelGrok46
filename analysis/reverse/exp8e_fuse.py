#!/usr/bin/env python3
"""Fuse teacher CatBoost OOF with generating-process arms. Save best_oof.npy."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import OUT, auc, dump_json, load_train, rank01

def load_arr(p):
    p = Path(p)
    if p.exists():
        if p.suffix == ".npy":
            return np.load(p)
        if p.suffix == ".csv":
            return pd.read_csv(p)
    return None


def main():
    df, y = load_train()
    cands = {}

    tea = load_arr("/workspace/submissions/cb_teacher_oof.csv")
    if tea is not None:
        tea = tea.set_index("id").loc[df["id"].astype(str)]
        for col in ["pred_cb_main", "pred_cb_alt", "pred_cb_w62", "te_src_cq_dq"]:
            if col in tea:
                cands[col] = tea[col].to_numpy()

    mapping = {
        "e1_poly": OUT / "exp1_oof_poly.npy",
        "e1_knn": OUT / "exp1_oof_knn.npy",
        "e2_hist": OUT / "exp2_best_oof.npy",
        "e2_nw": OUT / "exp2b_oof_nw.npy",
        "e3_add": OUT / "exp3_oof_add.npy",
        "e3b": OUT / "exp3b_oof.npy",
        "e5_cb": OUT / "exp5_oof_cb_core.npy",
        "e5_hgb": OUT / "exp5_oof_hgb.npy",
        "e6_third": OUT / "exp6_oof_third.npy",
        "e8a_hgb": OUT / "exp8a_hgb_main.npy",
        "e8a_te": OUT / "exp8a_ridge_te.npy",
        "e8a_freq": OUT / "exp8a_ridge_freq.npy",
        "ote": OUT / "exp_ote_ridge.npy",
        "lgb": OUT / "exp8d_lgb_w62.npy",
        "lgb_alt": OUT / "exp8d_lgb_alt.npy",
        "cb5A": Path("/workspace/analysis/cb_oofA.npy"),
        "cb5B": Path("/workspace/analysis/cb_oofB.npy"),
    }
    for k, p in mapping.items():
        if p.exists():
            cands[k] = np.load(p)

    print("=== ARM AUC ===")
    scores = {}
    for k, a in cands.items():
        if len(a) != len(y) or not np.isfinite(a).all():
            print("skip", k, getattr(a, "shape", None))
            continue
        scores[k] = auc(y, a)
        print(f"  {k:16s} {scores[k]:.5f}")

    # blends
    def R(k):
        return rank01(cands[k])

    blends = {}
    if "pred_cb_w62" in cands:
        blends["teacher_w62"] = cands["pred_cb_w62"]
        blends["teacher_max2"] = np.maximum(R("pred_cb_main"), R("pred_cb_alt"))
    if "pred_cb_w62" in cands and "lgb" in cands:
        blends["cb0.85_lgb0.15"] = 0.85 * R("pred_cb_w62") + 0.15 * R("lgb")
        blends["cb0.75_lgb0.25"] = 0.75 * R("pred_cb_w62") + 0.25 * R("lgb")
        blends["cb0.90_lgb0.10"] = 0.90 * R("pred_cb_w62") + 0.10 * R("lgb")
    if "pred_cb_w62" in cands and "e3b" in cands:
        blends["cb0.85_ridge0.15"] = 0.85 * R("pred_cb_w62") + 0.15 * R("e3b")
        blends["cb0.80_ridge0.20"] = 0.80 * R("pred_cb_w62") + 0.20 * R("e3b")
    if "pred_cb_w62" in cands and "e8a_te" in cands:
        blends["cb0.88_te0.12"] = 0.88 * R("pred_cb_w62") + 0.12 * R("e8a_te")
    if "pred_cb_w62" in cands and "e2_nw" in cands:
        blends["cb0.90_nw0.10"] = 0.90 * R("pred_cb_w62") + 0.10 * R("e2_nw")
    if "pred_cb_w62" in cands and "e2_hist" in cands:
        blends["cb0.88_hist0.12"] = 0.88 * R("pred_cb_w62") + 0.12 * R("e2_hist")
    if "pred_cb_w62" in cands and "lgb" in cands and "e3b" in cands:
        blends["cb0.70_lgb0.15_ridge0.15"] = 0.70 * R("pred_cb_w62") + 0.15 * R("lgb") + 0.15 * R("e3b")
        blends["cb0.78_lgb0.12_ridge0.10"] = 0.78 * R("pred_cb_w62") + 0.12 * R("lgb") + 0.10 * R("e3b")
    if "pred_cb_w62" in cands and "lgb" in cands and "e8a_te" in cands:
        blends["cb0.75_lgb0.15_te0.10"] = 0.75 * R("pred_cb_w62") + 0.15 * R("lgb") + 0.10 * R("e8a_te")
    if "pred_cb_main" in cands and "pred_cb_alt" in cands and "lgb" in cands:
        blends["max2cb_lgb"] = 0.85 * np.maximum(R("pred_cb_main"), R("pred_cb_alt")) + 0.15 * R("lgb")
    if "pred_cb_w62" in cands and "e6_third" in cands:
        blends["cb0.88_third0.12"] = 0.88 * R("pred_cb_w62") + 0.12 * R("e6_third")
    if "pred_cb_w62" in cands and "cb5A" in cands:
        blends["teacher_cb5"] = 0.7 * R("pred_cb_w62") + 0.18 * R("cb5A") + 0.12 * R("cb5B")

    print("\n=== BLENDS ===")
    bscores = {k: auc(y, v) for k, v in blends.items()}
    for k, v in sorted(bscores.items(), key=lambda z: -z[1]):
        print(f"  {k:32s} {v:.5f}")

    best_k = max({**scores, **bscores}, key=lambda k: {**scores, **bscores}[k])
    if best_k in blends:
        best = blends[best_k]
    else:
        best = cands[best_k]
    best_auc = auc(y, best)
    np.save(OUT / "best_oof.npy", best)
    dump_json("exp8e_fuse.json", {"arms": scores, "blends": bscores, "best": best_k, "best_auc": best_auc})
    print("SAVED", best_k, best_auc)


if __name__ == "__main__":
    main()
