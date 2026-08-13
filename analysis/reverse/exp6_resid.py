#!/usr/bin/env python3
"""Exp6: residuals after isotonic(ratio) / isotonic(rate). Third arm from leftovers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from common import (
    auc,
    both_auc,
    dump_json,
    fill_condition,
    load_train,
    numeric_block,
    per_source_rank,
    save_oof,
    skf,
    te_val_only,
    cond_r_of,
    source_median_map,
)


def main():
    df, y = load_train()
    n = len(y)
    oof_iso_ratio = np.zeros(n)
    oof_iso_rate = np.zeros(n)
    oof_resid_hgb = np.zeros(n)
    oof_third = np.zeros(n)
    leftover = []

    for fold, (tr_i, va_i) in enumerate(skf(y)):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i].astype(float)
        cond_tr, cond_va = fill_condition(trn, val)
        src_tr = trn["source"].astype(str).to_numpy()
        src_va = val["source"].astype(str).to_numpy()
        med = source_median_map(src_tr, cond_tr)
        ratio_tr = trn["days"].to_numpy(float) / np.clip(cond_r_of(src_tr, cond_tr, med), 1e-6, None)
        ratio_va = val["days"].to_numpy(float) / np.clip(cond_r_of(src_va, cond_va, med), 1e-6, None)
        rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
        rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
        rate_tr = trn["days"].to_numpy(float) * (1.0 - rk_tr)
        rate_va = val["days"].to_numpy(float) * (1.0 - rk_va)

        iso_r = IsotonicRegression(out_of_bounds="clip")
        iso_r.fit(ratio_tr, ytr)
        p_ratio_tr = iso_r.predict(ratio_tr)
        p_ratio_va = iso_r.predict(ratio_va)
        oof_iso_ratio[va_i] = p_ratio_va

        iso_t = IsotonicRegression(out_of_bounds="clip")
        iso_t.fit(rate_tr, ytr)
        p_rate_va = iso_t.predict(rate_va)
        oof_iso_rate[va_i] = p_rate_va

        resid = ytr - p_ratio_tr

        # leftover univariate on residual (train only, report signed corr / residual AUC)
        cand = {
            "region_te": None,
            "age": trn["age_range"].to_numpy(float),
            "x20": trn["x20"].to_numpy(float),
            "V": trn["V"].to_numpy(float),
            "cc": trn["cc"].to_numpy(float),
            "livability": trn["livability"].to_numpy(float),
            "x1": trn["x1"].to_numpy(float),
            "x5": trn["x5"].to_numpy(float),
        }
        # region TE of residual
        reg_va = te_val_only(trn["region"].astype(str), val["region"].astype(str), resid, m=15)
        age_te_va = te_val_only(trn["age_range"].astype(str), val["age_range"].astype(str), resid, m=8)

        fold_left = {"fold": fold}
        for name, arr in cand.items():
            if arr is None:
                continue
            # residual AUC using raw feature (direction-free)
            fold_left[name] = both_auc(resid, arr)
        leftover.append(fold_left)

        # third arm: HGB on residual using region, age, x20, V, windows, source
        Ntr = numeric_block(trn, cond_tr, trn["source"], trn["source"], cond_tr)
        Nva = numeric_block(val, cond_va, val["source"], trn["source"], cond_tr)
        extra_tr = np.column_stack(
            [
                trn["livability"].to_numpy(float),
                trn["x20"].to_numpy(float),
                trn["V"].to_numpy(float),
            ]
        )
        extra_va = np.column_stack(
            [
                val["livability"].to_numpy(float),
                val["x20"].to_numpy(float),
                val["V"].to_numpy(float),
            ]
        )
        src_codes = {s: i for i, s in enumerate(sorted(trn["source"].unique()))}
        reg_codes = {s: i for i, s in enumerate(sorted(trn["region"].unique()))}
        cat_tr = np.column_stack(
            [trn["source"].map(src_codes).fillna(-1), trn["region"].map(reg_codes).fillna(-1), trn["age_range"]]
        )
        cat_va = np.column_stack(
            [val["source"].map(src_codes).fillna(-1), val["region"].map(reg_codes).fillna(-1), val["age_range"]]
        )
        # residual model should NOT re-learn ratio; drop ratio/rate-like cols
        drop = [c for c in Ntr.columns if c in {"ratio", "rate", "ratio_sqrt", "log_ratio", "days", "days_log"}]
        Rtr = Ntr.drop(columns=drop, errors="ignore").to_numpy(float)
        Rva = Nva.drop(columns=drop, errors="ignore").to_numpy(float)
        Xtr = np.hstack([Rtr, extra_tr, cat_tr])
        Xva = np.hstack([Rva, extra_va, cat_va])
        hgb = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_depth=4,
            max_iter=250,
            min_samples_leaf=100,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=fold,
        )
        hgb.fit(Xtr, resid)
        r_va = hgb.predict(Xva)
        oof_resid_hgb[va_i] = r_va
        oof_third[va_i] = p_ratio_va + r_va
        # also mix in region residual TE
        oof_third[va_i] = p_ratio_va + 0.7 * r_va + 0.3 * reg_va

        print(
            f"fold {fold} iso_ratio={auc(y[va_i], p_ratio_va):.4f} iso_rate={auc(y[va_i], p_rate_va):.4f} "
            f"third={auc(y[va_i], oof_third[va_i]):.4f} resid_auc={auc(y[va_i], r_va):.4f}",
            flush=True,
        )

    # aggregate leftover
    keys = [k for k in leftover[0] if k != "fold"]
    leftover_mean = {k: float(np.mean([d[k] for d in leftover])) for k in keys}

    report = {
        "isotonic_ratio_oof": auc(y, oof_iso_ratio),
        "isotonic_rate_oof": auc(y, oof_iso_rate),
        "third_arm_oof": auc(y, oof_third),
        "resid_hgb_as_score": auc(y, oof_resid_hgb),
        "leftover_residual_both_auc_mean": leftover_mean,
        "note": "After isotonic(ratio), leftover both-AUC of region/age/x20/V. Third arm = iso(ratio)+residual HGB+region TE.",
    }
    print("\n=== EXP6 ===")
    print("iso ratio", report["isotonic_ratio_oof"])
    print("iso rate", report["isotonic_rate_oof"])
    print("third", report["third_arm_oof"])
    print("leftover", leftover_mean)
    dump_json("exp6_resid.json", report)
    save_oof("exp6_oof_iso_ratio.npy", oof_iso_ratio)
    save_oof("exp6_oof_iso_rate.npy", oof_iso_rate)
    save_oof("exp6_oof_third.npy", oof_third)
    print("WROTE exp6_resid.json")


if __name__ == "__main__":
    main()
