"""Phase4 honest max2 fusion: max(merger_ord8 + v2_cat_alt8) = nested 0.69993.

This is the PURE HONEST baseline (no early-stopping arms). Both arms use fixed
tree counts (800), no use_best_model, label-free FE. The 5-block nested OOF is
an unbiased estimate of generalization.

Recipe:
  - merger_ord8: v2 main FE + Ordered boosting, depth=5, 8 seeds (2026..2033)
  - v2_cat_alt8: v2 'alt' encoding world, depth=6, 8 seeds (2026..2033)
  - fusion rule: element-wise max of rank-normalized predictions
  - metric: 5-block nested OOF AUC (re-rank within each block, then AUC)

Output:
  - submissions/submission_v4_honest.csv  (6398 rows, label in [0.001, 0.999])
  - prints: nested OOF AUC (should be 0.69993) and full OOF AUC
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ART = Path(__file__).resolve().parent.parent / "artifacts"
SUB = Path(__file__).resolve().parent.parent / "submissions"
DATA = Path("/workspace/data")
N_BLOCKS = 5


def load(name):
    """Load an arm's oof + test_pred, return rank-normalized in [0,1]."""
    d = np.load(ART / f"{name}.npz", allow_pickle=True)
    oof = np.asarray(d["oof"], dtype=float)
    te = np.asarray(d["test_pred"], dtype=float)
    return (rankdata(oof) / len(oof), rankdata(te) / len(te))


def nested_auc(oof, y, n_blocks=N_BLOCKS):
    """5-block nested OOF AUC: re-rank within each block, then AUC.

    This is the conservative (anti-optimistic) metric. Full OOF AUC is ~+0.0003
    higher because global ranking absorbs some between-block variance.
    """
    n = len(y)
    out = np.zeros(n)
    for b in np.array_split(np.arange(n), n_blocks):
        out[b] = rankdata(oof[b]) / len(b)
    return roc_auc_score(y, out)


def main():
    y = pd.read_csv(DATA / "train.csv")["label"].astype(int).values
    test = pd.read_csv(DATA / "test.csv")

    mo_oof, mo_te = load("merger_ord8")
    ca_oof, ca_te = load("v2_cat_alt8")

    print(f"  [honest] merger_ord8   oof_auc={roc_auc_score(y, mo_oof):.5f}")
    print(f"  [honest] v2_cat_alt8   oof_auc={roc_auc_score(y, ca_oof):.5f}")

    # max2 fusion (element-wise max of rank-normalized predictions)
    f_oof = np.maximum(mo_oof, ca_oof)
    f_te = np.maximum(mo_te, ca_te)

    nest = nested_auc(f_oof, y)
    full = roc_auc_score(y, f_oof)
    print(f"\nmax2(merger_ord8 + v2_cat_alt8)")
    print(f"  nested={nest:.5f}  full={full:.5f}")
    print(f"  (nested is the unbiased metric; target = 0.69993)")

    # write submission
    sub = pd.DataFrame({"id": test["id"], "label": f_te.clip(0.001, 0.999)})
    out = SUB / "submission_v4_honest.csv"
    sub.to_csv(out, index=False)
    assert len(sub) == 6398
    assert sub["label"].between(0, 1).all()
    print(f"\nwrote {out}  rows={len(sub)}  label_range=[{sub.label.min():.4f},{sub.label.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
