#!/usr/bin/env python3
"""Write submissions/submission.csv from the HONEST 10-fold opus5 arm + frozen gate.

Does not pick gauss/LGB/VAL-ES mixes. This is the version requested as the
HONEST main arm (Classifier Logloss, 10-fold, no ES, max2, then insurer gate).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from insurer_gate import apply_gate  # noqa: E402
from train_honest10 import (  # noqa: E402
    ART,
    ROOT,
    load_xy,
    nested_auc,
    pool_arm,
    rank01,
    write_report,
)

SUB = ROOT / "submissions"


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-seeds", type=int, default=8)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()

    train, test, y = load_xy()
    report = write_report(y)
    n_both = int(report.get("n_seed_both") or 0)
    if n_both < args.min_seeds and not args.allow_partial:
        print(
            f"refuse: honest10 has {n_both} dual-arm seeds, need {args.min_seeds}. "
            "Pass --allow-partial to override.",
            flush=True,
        )
        return 2

    main_p, alt_p = pool_arm("main", y), pool_arm("alt", y)
    if main_p is None or alt_p is None:
        print("refuse: missing main or alt checkpoints", flush=True)
        return 2
    oof = np.maximum(main_p["oof"], alt_p["oof"])
    tes = np.maximum(main_p["test"], alt_p["test"])
    days_tr = train["days"].to_numpy(np.float64)
    days_te = test["days"].to_numpy(np.float64)
    gated_oof = rank01(apply_gate(oof, days_tr))
    gated_te = rank01(apply_gate(tes, days_te))
    ungated = auc(y, oof)
    gated = auc(y, gated_oof)
    nest = nested_auc(oof, y)

    SUB.mkdir(parents=True, exist_ok=True)
    src = SUB / "submission.csv"
    if src.is_file():
        shutil.copy2(src, SUB / "submission_before_honest10.csv")

    order = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])
    order["id"] = order["id"].astype(str)
    test["id"] = test["id"].astype(str)
    train["id"] = train["id"].astype(str)
    out = order.merge(pd.DataFrame({"id": test["id"].astype(str), "label": gated_te}), on="id", how="left")
    out["label"] = out["label"].fillna(0.5)
    assert len(out) == 6398
    out.to_csv(SUB / "submission.csv", index=False)
    out.to_csv(SUB / "submission_honest10_gated.csv", index=False)

    payload = {
        "recipe": f"honest10 max2 {n_both}-seed 10-fold Classifier Logloss no-ES + frozen 4-window gate",
        "selected": f"honest10_max2_{n_both}seed",
        "auc_ungated": ungated,
        "auc_nested": nest,
        "auc_gated": gated,
        "n_seed": n_both,
        "main_pool": main_p["pool_auc"],
        "alt_pool": alt_p["pool_auc"],
        "main_per_seed": [round(a, 6) for a in main_p["per_seed"]],
        "alt_per_seed": [round(a, 6) for a in alt_p["per_seed"]],
        "n_test": int(len(out)),
        "gate": "floor[1725,1825)+[2110,2210) -0.10[700,880) +0.05[9370,9475)",
        "honest_note": "HONEST_NO_ES 10fold x Nseed; not mixed with LGB/VAL-ES; no cross-protocol max",
        "status": "submitted",
    }
    (SUB / "final_best_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (SUB / "final_best_metrics.json").write_text(
        json.dumps(
            {
                "auc_blend": gated,
                "auc_gated": gated,
                "auc_cb_main": main_p["pool_auc"],
                "auc_cb_w62": gated,
                "selected": payload["selected"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (SUB / "oof_report.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in payload.items() if not isinstance(v, (list, dict))) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "id": train["id"],
            "label": y,
            "pred_cb_main": rank01(main_p["oof"]),
            "pred_cb_alt": rank01(alt_p["oof"]),
            "pred_fuse": rank01(oof),
        }
    ).to_parquet(SUB / "final_best_oof.parquet", index=False)
    pd.DataFrame(
        {
            "id": test["id"],
            "pred_cb_main": rank01(main_p["test"]),
            "pred_cb_alt": rank01(alt_p["test"]),
            "pred_fuse": rank01(tes),
        }
    ).to_parquet(SUB / "final_best_test.parquet", index=False)
    print(
        f"wrote {SUB / 'submission.csv'} honest10_{n_both}seed "
        f"ungated={ungated:.6f} nested={nest:.6f} gated={gated:.6f} n={len(out)}",
        flush=True,
    )
    (ART / "honest10_submit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
