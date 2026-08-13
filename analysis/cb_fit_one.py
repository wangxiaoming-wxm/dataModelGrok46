#!/usr/bin/env python3
"""Train one CatBoost RMSE model and write predictions. Invoked by Scala CatBoostArm."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from catboost import CatBoostRegressor, Pool


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--apply", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--id-col", default="id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--cats", default="")
    p.add_argument("--params", default="{}")
    p.add_argument("--sep", default="\t")
    args = p.parse_args()

    params = json.loads(args.params)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("allow_writing_files", False)
    params.setdefault("thread_count", 4)
    params.setdefault("verbose", False)

    tr = pd.read_csv(args.train, sep=args.sep)
    ap = pd.read_csv(args.apply, sep=args.sep)
    cats = [c for c in args.cats.split(",") if c]
    idc, ycol = args.id_col, args.label_col
    feat_cols = [c for c in tr.columns if c not in (idc, ycol)]
    for c in cats:
        if c in tr.columns:
            tr[c] = tr[c].astype(str).fillna("NA")
            ap[c] = ap[c].astype(str).fillna("NA")
    Xtr = tr[feat_cols]
    ytr = tr[ycol].astype(float).to_numpy()
    Xap = ap[feat_cols]
    cat_idx = [feat_cols.index(c) for c in cats if c in feat_cols]
    pool_tr = Pool(Xtr, ytr, cat_features=cat_idx)
    model = CatBoostRegressor(**params)
    model.fit(pool_tr, verbose=False)
    pred = model.predict(Pool(Xap, cat_features=cat_idx))
    out = pd.DataFrame({idc: ap[idc].astype(str), "pred": pred})
    out.to_csv(args.pred, sep=args.sep, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
