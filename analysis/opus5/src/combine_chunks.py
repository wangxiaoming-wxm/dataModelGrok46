"""Combine v2fe_ord chunk npz files into a single rank-pooled 8-seed arm.

Each chunk npz holds its own 2-seed rank-pooled oof & test_pred (already
rank-normalized within chunk). To combine across chunks at rank level, we
average the chunk-level rank vectors and re-rank — equivalent to a balanced
rank pool across all seeds.

Usage: python3 src/combine_chunks.py <OUT_TAG> <chunk0> <chunk1> [...]
  e.g. python3 src/combine_chunks.py merger_ord8 v2fe_ord_c0 v2fe_ord_c1 v2fe_ord_c2 v2fe_ord_c3
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ART = Path(__file__).resolve().parent.parent / "artifacts"
out_tag = sys.argv[1]
chunks = sys.argv[2:]


def main():
    oofs, tes, all_per_seed, all_seeds = [], [], [], []
    y = None
    for c in chunks:
        d = np.load(ART / f"{c}.npz", allow_pickle=True)
        oofs.append(np.asarray(d["oof"], dtype=float))
        tes.append(np.asarray(d["test_pred"], dtype=float))
        all_per_seed.extend(np.asarray(d["per_seed"]).tolist())
        all_seeds.extend(np.asarray(d["seeds"]).tolist())
        y = np.asarray(d["y"], dtype=int)
    # average chunk-level rank vectors, then re-rank for the combined pool
    oof = np.mean(np.vstack(oofs), axis=0)
    te = np.mean(np.vstack(tes), axis=0)
    oof_r = rankdata(oof) / len(oof)
    te_r = rankdata(te) / len(te)
    auc = roc_auc_score(y, oof_r)
    print(f"[combine] {out_tag}: {len(all_seeds)} seeds combined OOF AUC = {auc:.6f}")
    print(f"  per-seed mean = {np.mean(all_per_seed):.6f} ± {np.std(all_per_seed):.6f}")
    print(f"  per-seed vals = {[f'{x:.5f}' for x in all_per_seed]}")
    np.savez(ART / f"{out_tag}.npz", oof=oof_r, test_pred=te_r,
             per_seed=np.array(all_per_seed), seeds=np.array(all_seeds), y=y)
    print(f"  wrote artifacts/{out_tag}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
