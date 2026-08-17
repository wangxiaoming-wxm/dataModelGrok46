"""Chunked 8-seed Ordered bag on v2 main FE, depth=5 (the sweep winner).

Run in 4 chunks of 2 seeds each (each chunk ~12-14 min, fits one bash window).
Each chunk saves its own npz; combine with combine_chunks.py at the end.

Usage: python3 src/v2fe_ord_chunk.py <CHUNK_IDX>
  CHUNK 0 -> seeds 2026,2027  -> v2fe_ord_c0.npz
  CHUNK 1 -> seeds 2028,2029  -> v2fe_ord_c1.npz
  CHUNK 2 -> seeds 2030,2031  -> v2fe_ord_c2.npz
  CHUNK 3 -> seeds 2032,2033  -> v2fe_ord_c3.npz

Honest: fixed 800 trees, no ES, FE label-free, per-seed jitter stream.

NOTE: V2_SRC now points to the local src2/ (inlined dependency, self-contained).
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
V2_SRC = str(Path(__file__).resolve().parent / "src2")
sys.path.insert(0, V2_SRC)
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from features import fit_edges
from arms import catboost_frame

DATA = Path("/workspace/data")
ART = Path(__file__).resolve().parent.parent / "artifacts"
DEPTH = 5
ITER = 800
ALL_SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
CHUNK_SIZE = 2
CHUNK_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEEDS = ALL_SEEDS[CHUNK_IDX * CHUNK_SIZE:(CHUNK_IDX + 1) * CHUNK_SIZE]
TAG = f"v2fe_ord_c{CHUNK_IDX}"


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].astype(int).values
    raw_all = pd.concat([train.drop(columns=["label"]), test], ignore_index=True)
    edges = fit_edges(raw_all)
    pooled_oof = np.zeros(len(y)); pooled_te = np.zeros(len(test))
    per_seed = []; t_all = time.time()
    for si, seed in enumerate(SEEDS):
        stream = ALL_SEEDS.index(seed) + 1  # global stream index for diversity
        X, cats = catboost_frame(raw_all, edges, stream_offset=stream, n_views=4)
        Xtr = X.iloc[:len(train)].reset_index(drop=True)
        Xte = X.iloc[len(train):].reset_index(drop=True)
        oof_s = np.zeros(len(y)); te_s = np.zeros(len(test)); t0 = time.time()
        for fold, (ti, vi) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y)
        ):
            m = CatBoostClassifier(
                loss_function="Logloss", learning_rate=0.03, l2_leaf_reg=10,
                random_strength=0.7, verbose=False, thread_count=-1,
                allow_writing_files=False, depth=DEPTH, iterations=ITER,
                boosting_type="Ordered", random_seed=seed + fold)
            m.fit(Xtr.iloc[ti], y[ti], cat_features=cats, verbose=False)
            oof_s[vi] = m.predict_proba(Xtr.iloc[vi])[:, 1]
            te_s += m.predict_proba(Xte)[:, 1] / 5
        a = roc_auc_score(y, oof_s)
        per_seed.append(a)
        print(f"[{TAG}] seed {seed} (stream {stream}): OOF={a:.6f} ({time.time()-t0:.0f}s)", flush=True)
        pooled_oof += rankdata(oof_s) / len(oof_s)
        pooled_te += rankdata(te_s) / len(te_s)
    pooled_oof /= len(SEEDS); pooled_te /= len(SEEDS)
    pool_auc = roc_auc_score(y, pooled_oof)
    print(f"\n[{TAG}] {len(SEEDS)}-seed rank-pooled OOF = {pool_auc:.6f}  "
          f"(per-seed {np.mean(per_seed):.6f} ± {np.std(per_seed):.6f})  "
          f"total {time.time()-t_all:.0f}s", flush=True)
    np.savez(ART / f"{TAG}.npz", oof=pooled_oof, test_pred=pooled_te,
             per_seed=np.array(per_seed), seeds=np.array(SEEDS), depth=DEPTH, y=y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
