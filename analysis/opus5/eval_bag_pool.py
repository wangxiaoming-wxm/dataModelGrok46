#!/usr/bin/env python3
"""Pool honest10 checkpoints by bag, then gauss(0.70 h10 + 0.30 ref) + frozen gate.

Does not touch the running trainer. Dry-run by default: never overwrites
submissions/submission.csv unless bag1 is complete AND gated+nested both beat
the locked 0.70742 recipe, and --write is passed.

Usage:
  python3 -u analysis/opus5/eval_bag_pool.py
  python3 -u analysis/opus5/eval_bag_pool.py --write
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = Path("/workspace") if Path("/workspace/data/train.csv").is_file() else HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from insurer_gate import apply_gate  # noqa: E402

CKPT = ROOT / "analysis" / "opus5" / "ckpt_honest10"
ART = ROOT / "analysis" / "opus5" / "artifacts"
BEST = ROOT / "analysis" / "best716" / "artifacts"
SUB = ROOT / "submissions"
ALL_SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
W_REF = 0.30
LOCKED_GATED = 0.7074169896957119
LOCKED_NESTED = 0.7048286274191353
BAG0_RE = re.compile(r"^(main|alt)_f10_s(\d+)\.npz$")
BAGN_RE = re.compile(r"^(main|alt)_f10_s(\d+)_b(\d+)\.npz$")


def rank01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return rankdata(a, method="average") / float(len(a))


def gauss(a: np.ndarray) -> np.ndarray:
    r = np.clip(rank01(a), 1e-4, 1.0 - 1e-4)
    return norm.ppf(r)


def nested_auc(oof: np.ndarray, y: np.ndarray, n_blocks: int = 5) -> float:
    out = np.zeros(len(y))
    for block in np.array_split(np.arange(len(y)), n_blocks):
        out[block] = rankdata(oof[block], method="average") / float(len(block))
    return float(roc_auc_score(y, out))


def list_ckpt(arm: str, bag: int | None) -> list[Path]:
    out = []
    for p in sorted(CKPT.glob(f"{arm}_f10_s*.npz")):
        m0 = BAG0_RE.match(p.name)
        mn = BAGN_RE.match(p.name)
        if bag == 0:
            if m0:
                out.append(p)
        elif bag is None:
            if m0 or mn:
                out.append(p)
        else:
            if mn and int(mn.group(3)) == bag:
                out.append(p)
    return out


def pool_files(paths: list[Path], y: np.ndarray) -> dict | None:
    if not paths:
        return None
    oofs, tes, seeds, aucs, names = [], [], [], [], []
    for p in paths:
        z = np.load(p)
        oofs.append(rank01(z["oof"]))
        tes.append(rank01(z["test_pred"]))
        seeds.append(int(z["seed"]))
        aucs.append(float(z["auc"]))
        names.append(p.name)
    oof = rank01(np.mean(np.vstack(oofs), axis=0))
    te = rank01(np.mean(np.vstack(tes), axis=0))
    return {
        "n_file": len(paths),
        "seeds": seeds,
        "files": names,
        "per_seed": aucs,
        "oof": oof,
        "test": te,
        "pool_auc": float(roc_auc_score(y, oof)),
    }


def load_ref(n: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(BEST / "ref.npz")
    mp = pd.read_csv(BEST / "matched_pairs_L3.csv")
    lp = mp["local_row"].to_numpy(dtype=int)
    rp = mp["ref_row"].to_numpy(dtype=int)
    cat = rank01(z["cat_opt5_oof"])
    ref_oof = rank01((rank01(z["ref_model_oof"]) + rank01(z["ref_s8888_oof"])) / 2.0)
    mapped = np.full(n, np.nan)
    mapped[lp] = ref_oof[rp]
    oof = np.where(np.isfinite(mapped), mapped, cat)
    tes = rank01((z["ref_model_local_test"] + z["ref_s8888_local_test"]) / 2.0)
    return oof.astype(np.float64), tes.astype(np.float64)


def score_pack(name: str, h_oof: np.ndarray, h_te: np.ndarray, y: np.ndarray, ref_oof: np.ndarray, ref_te: np.ndarray, days_tr: np.ndarray, days_te: np.ndarray) -> dict:
    h_oof, h_te = rank01(h_oof), rank01(h_te)
    solo_u = float(roc_auc_score(y, h_oof))
    solo_n = nested_auc(h_oof, y)
    oof = rank01((1.0 - W_REF) * gauss(h_oof) + W_REF * gauss(ref_oof))
    tes = rank01((1.0 - W_REF) * gauss(h_te) + W_REF * gauss(ref_te))
    gated_oof = rank01(apply_gate(oof, days_tr))
    gated_te = rank01(apply_gate(tes, days_te))
    return {
        "name": name,
        "h10_ungated": solo_u,
        "h10_nested": solo_n,
        "blend_ungated": float(roc_auc_score(y, oof)),
        "blend_nested": nested_auc(oof, y),
        "blend_gated": float(roc_auc_score(y, gated_oof)),
        "oof": oof,
        "tes": tes,
        "gated_oof": gated_oof,
        "gated_te": gated_te,
    }


def bootstrap_delta(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        pi = rng.choice(pos, size=len(pos), replace=True)
        ni = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([pi, ni])
        diffs[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "mean": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_pos": float((diffs > 0).mean()),
    }


def bag_complete(bag: int) -> bool:
    for arm in ("main", "alt"):
        seeds = set()
        for p in list_ckpt(arm, bag):
            m = BAG0_RE.match(p.name) if bag == 0 else BAGN_RE.match(p.name)
            if m:
                seeds.add(int(m.group(2)))
        if seeds != set(ALL_SEEDS):
            return False
    return True


def slim(pack: dict) -> dict:
    return {k: pack[k] for k in ("name", "h10_ungated", "h10_nested", "blend_ungated", "blend_nested", "blend_gated")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="overwrite submission.csv if criteria met")
    p.add_argument("--force", action="store_true", help="write even if gated/nested do not beat lock")
    args = p.parse_args()

    train = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    test = pd.read_csv(ROOT / "data" / "test.csv", dtype={"id": str})
    y = train["label"].to_numpy(dtype=int)
    days_tr = train["days"].to_numpy(np.float64)
    days_te = test["days"].to_numpy(np.float64)
    ref_oof, ref_te = load_ref(len(y))

    specs = {
        "bag0": (0,),
        "bag1": (1,),
        "bag0+bag1": (0, 1),
    }
    packs = {}
    file_counts = {}
    for name, bags in specs.items():
        mains, alts = [], []
        for b in bags:
            mains.extend(list_ckpt("main", b))
            alts.extend(list_ckpt("alt", b))
        file_counts[name] = {"main": [x.name for x in mains], "alt": [x.name for x in alts]}
        if not mains or not alts:
            continue
        pm, pa = pool_files(mains, y), pool_files(alts, y)
        mx, te = np.maximum(pm["oof"], pa["oof"]), np.maximum(pm["test"], pa["test"])
        packs[name] = score_pack(name, mx, te, y, ref_oof, ref_te, days_tr, days_te)
        packs[name]["n_main"] = pm["n_file"]
        packs[name]["n_alt"] = pa["n_file"]
        packs[name]["main_pool"] = pm["pool_auc"]
        packs[name]["alt_pool"] = pa["pool_auc"]

    b0_ok = bag_complete(0)
    b1_ok = bag_complete(1)
    print(f"bag0_complete={b0_ok} bag1_complete={b1_ok}", flush=True)
    for name, pack in packs.items():
        print(
            f"{name:10s} files={pack['n_main']}+{pack['n_alt']} "
            f"h10={pack['h10_ungated']:.5f}/{pack['h10_nested']:.5f} "
            f"gauss+gate ung={pack['blend_ungated']:.5f} nest={pack['blend_nested']:.5f} "
            f"gated={pack['blend_gated']:.5f}",
            flush=True,
        )

    cand = packs.get("bag0+bag1") if b1_ok else None
    locked = packs.get("bag0")
    boot = None
    if cand is not None and locked is not None:
        boot = bootstrap_delta(cand["gated_oof"], locked["gated_oof"], y)
        print(
            f"bootstrap gated bag01 vs bag0: Δ={boot['mean']:+.5f} "
            f"CI[{boot['ci_lo']:+.5f},{boot['ci_hi']:+.5f}] p_pos={boot['p_pos']:.3f}",
            flush=True,
        )

    should = False
    reason = "bag1 incomplete or missing pool"
    if cand is not None and locked is not None:
        dg = cand["blend_gated"] - LOCKED_GATED
        dn = cand["blend_nested"] - LOCKED_NESTED
        should = dg > 0.0 and dn > 0.0
        reason = f"gated {cand['blend_gated']:.5f} ({dg:+.5f} vs lock) nested {cand['blend_nested']:.5f} ({dn:+.5f} vs lock)"
        if should and boot is not None and boot["ci_lo"] <= 0:
            # still allow write if both metrics beat lock; CI is extra evidence
            reason += f"; bootstrap CI includes 0 (p_pos={boot['p_pos']:.3f})"

    payload = {
        "bag0_complete": b0_ok,
        "bag1_complete": b1_ok,
        "locked_gated": LOCKED_GATED,
        "locked_nested": LOCKED_NESTED,
        "packs": {k: slim(v) | {"n_main": v["n_main"], "n_alt": v["n_alt"]} for k, v in packs.items()},
        "bootstrap_gated_bag01_vs_bag0": boot,
        "should_write": should,
        "reason": reason,
        "recipe": "gauss(0.70·honest10_pool + 0.30·ref) + frozen 4-window gate",
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "bag_pool_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "packs"}, indent=2), flush=True)
    print(f"wrote {ART / 'bag_pool_report.json'}", flush=True)

    if cand is not None and b1_ok:
        np.savez(ART / "honest10_max2_bag01.npz", oof=cand["oof"], test_pred=cand["tes"], y=y)

    if not args.write:
        print("dry-run: pass --write to update submission.csv", flush=True)
        return 0
    if not b1_ok:
        print("refuse write: bag1 not complete (8 dual-arm seeds)", flush=True)
        return 2
    if cand is None:
        print("refuse write: missing bag0+bag1 pack", flush=True)
        return 2
    if not should and not args.force:
        print(f"refuse write: {reason}", flush=True)
        return 3

    SUB.mkdir(parents=True, exist_ok=True)
    src = SUB / "submission.csv"
    if src.is_file():
        shutil.copy2(src, SUB / "submission_before_bag01.csv")
    order = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])
    order["id"] = order["id"].astype(str)
    out = order.merge(pd.DataFrame({"id": test["id"], "label": cand["gated_te"]}), on="id", how="left")
    out["label"] = out["label"].fillna(0.5)
    assert len(out) == 6398
    out.to_csv(SUB / "submission.csv", index=False)
    out.to_csv(SUB / "submission_h10_bag01_ref30_gauss.csv", index=False)
    print(f"wrote {SUB / 'submission.csv'} gated={cand['blend_gated']:.6f} nested={cand['blend_nested']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
