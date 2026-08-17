# -*- coding: utf-8 -*-
"""v14：ref 2seed 等权(2026+8888) + 权重 0.42。
OOF 0.69427（v11 单seed0.38: 0.69392 → +0.00035）。"""
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from bootstrap_paired import paired_bootstrap

BASE = Path(__file__).resolve().parent
CLIP = 1e-4
W_REF = 0.42

def rank_norm(v): return rankdata(v, method="average") / len(v)
def gauss(v): return norm.ppf(np.clip(rank_norm(v), CLIP, 1.0 - CLIP))

def main():
    t0 = time.time()
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")
    y = train.label.to_numpy()

    cat = rank_norm(np.load(BASE / "comp_cat_opt5_oof.npy")); cat_t = rank_norm(np.load(BASE / "comp_cat_opt5_test.npy"))
    et = rank_norm(np.load(BASE / "comp_et_oof.npy")); et_t = rank_norm(np.load(BASE / "comp_et_test.npy"))
    bv3 = rank_norm(np.load(BASE / "comp_cat_b_v3_oof.npy")); bv3_t = rank_norm(np.load(BASE / "comp_cat_b_v3_test.npy"))
    xgb1 = rank_norm(np.load(BASE / "comp_xgb_oof.npy")); xgb1_t = rank_norm(np.load(BASE / "comp_xgb_test.npy"))
    d202 = rank_norm(np.load(BASE / "comp_xgb_d202_oof.npy")); d202_t = rank_norm(np.load(BASE / "comp_xgb_d202_test.npy"))
    d606 = rank_norm(np.load(BASE / "comp_xgb_d606_oof.npy")); d606_t = rank_norm(np.load(BASE / "comp_xgb_d606_test.npy"))
    d707 = rank_norm(np.load(BASE / "comp_xgb_d707_oof.npy")); d707_t = rank_norm(np.load(BASE / "comp_xgb_d707_test.npy"))

    mp = pd.read_csv(BASE / "matched_pairs_L3.csv")
    lp = mp["local_row"].to_numpy(dtype=int); rp = mp["ref_row"].to_numpy(dtype=int)
    ref_oof = rank_norm((rank_norm(np.load(BASE / "ref_model_oof.npy")) +
                         rank_norm(np.load(BASE / "ref_s8888_oof.npy"))) / 2.0)
    ref_t = rank_norm((np.load(BASE / "ref_model_local_test.npy") +
                       np.load(BASE / "ref_s8888_local_test.npy")) / 2.0)
    ref_al = np.full(len(y), np.nan); ref_al[lp] = ref_oof[rp]
    m = np.isfinite(ref_al)
    ref_c = np.where(m, ref_al, cat)

    def blend_oof():
        return rank_norm((0.75 - W_REF) * gauss(cat) + W_REF * gauss(ref_c) + 0.07 * gauss(et)
                         + 0.04 * gauss(bv3) + 0.04 * gauss(xgb1) + 0.04 * gauss(d202)
                         + 0.02 * gauss(d606) + 0.03 * gauss(d707))
    def blend_test():
        return rank_norm((0.75 - W_REF) * gauss(cat_t) + W_REF * gauss(ref_t) + 0.07 * gauss(et_t)
                         + 0.04 * gauss(bv3_t) + 0.04 * gauss(xgb1_t) + 0.04 * gauss(d202_t)
                         + 0.02 * gauss(d606_t) + 0.03 * gauss(d707_t))

    # v11 基准
    ref1_oof = rank_norm(np.load(BASE / "ref_model_oof.npy"))
    ref1_al = np.full(len(y), np.nan); ref1_al[lp] = ref1_oof[rp]
    ref1_c = np.where(np.isfinite(ref1_al), ref1_al, cat)
    v11_o = rank_norm((0.75 - 0.38) * gauss(cat) + 0.38 * gauss(ref1_c) + 0.07 * gauss(et)
                      + 0.04 * gauss(bv3) + 0.04 * gauss(xgb1) + 0.04 * gauss(d202)
                      + 0.02 * gauss(d606) + 0.03 * gauss(d707))

    v14_o = blend_oof()
    auc11 = float(roc_auc_score(y, v11_o)); auc14 = float(roc_auc_score(y, v14_o))
    boot = paired_bootstrap(y, v11_o, v14_o, n_boot=3000, seed=42)
    print(f"v11 (单seed ref0.38): OOF={auc11:.5f}", flush=True)
    print(f"v14 (2seed ref0.42):   OOF={auc14:.5f}  Δ={auc14-auc11:+.5f}", flush=True)
    print(f"bootstrap: p_pos={boot['p_pos']:.1%} ci={[round(x,5) for x in boot['ci95']]}", flush=True)

    old = BASE / "submission.csv"
    if old.exists():
        shutil.copy2(old, BASE / "submission_backup_sprint4_v13.csv")
    sub = pd.DataFrame({"id": test.id, "label": np.clip(blend_test(), 1e-6, 1 - 1e-6)})
    sub.to_csv(BASE / "submission.csv", index=False)
    print(f"submission.csv 已更新 (ref 2seed 0.42, OOF {auc14:.5f})", flush=True)

    report = {
        "protocol": f"v14: cat_opt5(0.33)+ref2seed(0.42)+et(0.07)+bv3(0.04)+xgb(0.04)+diverse(0.09) gauss",
        "v11_auc": auc11, "v14_auc": auc14, "delta": auc14 - auc11,
        "bootstrap": boot,
        "ref_component": "2seed等权(2026+8888)，迁移 0.6893",
        "submission": "submission.csv", "backup": "submission_backup_sprint4_v13.csv",
        "seconds": round(time.time() - t0, 1),
    }
    (BASE / "final_submit_sprint4_v14_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: final_submit_sprint4_v14_report.json", flush=True)

if __name__ == "__main__":
    main()
