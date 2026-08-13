"""Chunked 8-seed bag of v2's 'alt' encoding world, HONEST (v2 fit_predict is already honest).

cat_alt config: depth6, iter800, l2=6, one_hot_max_size=12, Plain boosting.
v2's strongest diverse encoding world (build_alt: rate=days*(1-rank_pct)).
Run in 4 chunks of 2 seeds. Per-seed jitter stream for diversity.

Usage: python3 src/v2_cat_alt_chunk.py <CHUNK_IDX>

NOTE: V2_SRC now points to the local src2/ (inlined dependency, self-contained).
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
V2_SRC = str(Path(__file__).resolve().parent / "src2")
sys.path.insert(0, V2_SRC)
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from features import fit_edges_alt
from arms import altboost_frame, ARMS, CAT_BASE

DATA = Path("/workspace/data")
ART = Path(__file__).resolve().parent.parent / "artifacts"
ALL_SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
CHUNK_SIZE = 2
CHUNK_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEEDS = ALL_SEEDS[CHUNK_IDX * CHUNK_SIZE:(CHUNK_IDX + 1) * CHUNK_SIZE]
TAG = f"v2_cat_alt_c{CHUNK_IDX}"
SPEC = ARMS["cat_alt"]


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges_alt(raw_all)
    base = dict(CAT_BASE)
    base.update({"l2_leaf_reg": 6, "one_hot_max_size": 12, "thread_count": -1})
    pooled_oof = np.zeros(len(y))
    pooled_te = np.zeros(len(test))
    per_seed = []
    t_all = time.time()
    for si, seed in enumerate(SEEDS):
        stream = ALL_SEEDS.index(seed) + 1
        X, cats = altboost_frame(raw_all, edges, stream_offset=stream, n_views=3)
        Xtr = X.iloc[:len(train)].reset_index(drop=True)
        Xte = X.iloc[len(train):].reset_index(drop=True)
        oof_s = np.zeros(len(y))
        te_s = np.zeros(len(test))
        t0 = time.time()
        for fold, (ti, vi) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y)
        ):
            m = CatBoostClassifier(**base, depth=SPEC["depth"],
                                   iterations=SPEC["iterations"],
                                   random_seed=seed + fold)
            m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)  # HONEST
            oof_s[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
            te_s += m.predict_proba(Xte)[:, 1] / 5
        a = roc_auc_score(y, oof_s)
        per_seed.append(a)
        print(f"[{TAG}] seed {seed} (stream {stream}): OOF={a:.6f} ({time.time()-t0:.0f}s)", flush=True)
        pooled_oof += rankdata(oof_s) / len(oof_s)
        pooled_te += rankdata(te_s) / len(te_s)
    pooled_oof /= len(SEEDS)
    pooled_te /= len(SEEDS)
    pool_auc = roc_auc_score(y, pooled_oof)
    print(f"\n[{TAG}] {len(SEEDS)}-seed rank-pooled OOF = {pool_auc:.6f}  "
          f"(per-seed {np.mean(per_seed):.6f} ± {np.std(per_seed):.6f})  "
          f"total {time.time()-t_all:.0f}s", flush=True)
    np.savez(ART / f"{TAG}.npz", oof=pooled_oof, test_pred=pooled_te,
             per_seed=np.array(per_seed), seeds=np.array(SEEDS), y=y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
