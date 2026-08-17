#!/usr/bin/env python3
"""Fuse all reverse-engineering OOF arms; save best_oof.npy if >= previous best."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import OUT, auc, dump_json, load_train, rank01, save_oof


def maybe(name):
    p = OUT / name
    if p.exists():
        return np.load(p)
    p2 = Path("/workspace/analysis") / name
    if p2.exists():
        return np.load(p2)
    return None


def main():
    _, y = load_train()
    cands = {}
    files = {
        "cb5_A": "/workspace/analysis/cb_oofA.npy",
        "cb5_B": "/workspace/analysis/cb_oofB.npy",
        "e1_poly": "exp1_oof_poly.npy",
        "e1_knn": "exp1_oof_knn.npy",
        "e1_blend": "exp1_oof_blend.npy",
        "e2_hist": "exp2_best_oof.npy",
        "e2_nw": "exp2b_oof_nw.npy",
        "e2_grid": "exp2b_oof_grid.npy",
        "e3_add": "exp3_oof_add.npy",
        "e3_mul": "exp3_oof_mul.npy",
        "e3_lpm": "exp3_oof_lpm.npy",
        "e5_mlp": "exp5_oof_mlp.npy",
        "e5_hgb": "exp5_oof_hgb.npy",
        "e5_knn": "exp5_oof_knn.npy",
        "e5_cb": "exp5_oof_cb_core.npy",
        "e6_iso_r": "exp6_oof_iso_ratio.npy",
        "e6_iso_t": "exp6_oof_iso_rate.npy",
        "e6_third": "exp6_oof_third.npy",
        "e8a_hgb": "exp8a_hgb_main.npy",
        "e8a_alt": "exp8a_hgb_alt.npy",
        "e8a_freq": "exp8a_ridge_freq.npy",
        "e8a_te": "exp8a_ridge_te.npy",
        "e8a_hist": "exp8a_hist2d.npy",
        "e8a_knn": "exp8a_knn_src.npy",
        "e8b_A": "exp8b_oofA.npy",
        "e8b_B": "exp8b_oofB.npy",
        "e8b_w62": "exp8b_w62.npy",
    }
    for k, fn in files.items():
        arr = maybe(fn)
        if arr is not None and len(arr) == len(y) and np.isfinite(arr).mean() > 0.99:
            a = auc(y, arr)
            cands[k] = (arr, a)
            print(f"  {k:12s} {a:.5f}")

    if not cands:
        print("no oofs yet")
        return

    # greedy rank blends of top arms
    ranked = sorted(cands.items(), key=lambda kv: -kv[1][1])
    print("\nTOP ARMS")
    for k, (_, a) in ranked[:12]:
        print(f"  {k:12s} {a:.5f}")

    blends = {}
    # pairwise among top 6
    topk = [k for k, _ in ranked[:8]]
    for i, a in enumerate(topk):
        for b in topk[i + 1 :]:
            s = 0.62 * rank01(cands[a][0]) + 0.38 * rank01(cands[b][0])
            blends[f"w62:{a}+{b}"] = (s, auc(y, s))
            s2 = 0.5 * rank01(cands[a][0]) + 0.5 * rank01(cands[b][0])
            blends[f"eq:{a}+{b}"] = (s2, auc(y, s2))
            s3 = np.maximum(rank01(cands[a][0]), rank01(cands[b][0]))
            blends[f"max:{a}+{b}"] = (s3, auc(y, s3))

    # 3-way among top 4
    t4 = topk[:4]
    if len(t4) >= 3:
        s = sum(rank01(cands[k][0]) for k in t4[:3]) / 3.0
        blends["eq3:" + "+".join(t4[:3])] = (s, auc(y, s))
        s = 0.5 * rank01(cands[t4[0]][0]) + 0.3 * rank01(cands[t4[1]][0]) + 0.2 * rank01(cands[t4[2]][0])
        blends["532:" + "+".join(t4[:3])] = (s, auc(y, s))
    if len(t4) >= 4:
        s = 0.4 * rank01(cands[t4[0]][0]) + 0.25 * rank01(cands[t4[1]][0]) + 0.2 * rank01(cands[t4[2]][0]) + 0.15 * rank01(cands[t4[3]][0])
        blends["4way"] = (s, auc(y, s))

    best_single = ranked[0]
    best_blend = max(blends.items(), key=lambda kv: kv[1][1]) if blends else None
    print("\nBEST SINGLE", best_single[0], best_single[1][1])
    if best_blend:
        print("BEST BLEND", best_blend[0], best_blend[1][1])

    if best_blend and best_blend[1][1] >= best_single[1][1]:
        name, (arr, a) = best_blend[0], best_blend[1]
    else:
        name, (arr, a) = best_single[0], best_single[1]

    np.save("/workspace/analysis/reverse/best_oof.npy", arr)
    dump_json(
        "exp8c_fuse.json",
        {
            "arms": {k: v[1] for k, v in cands.items()},
            "best_name": name,
            "best_auc": a,
            "top_blends": sorted({k: v[1] for k, v in blends.items()}.items(), key=lambda z: -z[1])[:15],
        },
    )
    print("SAVED best_oof.npy", name, a)


if __name__ == "__main__":
    main()
