#!/usr/bin/env python3
"""Actuarial residual hunt on teacher OOF, nested 10-fold.

Hypotheses (insurance, not fishing):
1. days is vehicle/policy age; claim pits cluster on 365-day anniversaries
   (renewal / inspection / warranty). Frozen gates already hit 2y and 5y.
2. Binary flags (w1/w2, r1, code) look like coverage / underwriting switches
   that CatBoost cats may not have as explicit product columns.
3. Teacher (inner-ES 3bag) and HONEST 8seed×2bag / val-ES best_oof may still
   have rank diversity worth nested blending.
4. Extra window [2110,2210) ≈ 6*365 was found in raw rates but not gated.

Do not retune the three frozen gate magnitudes. RMSE/AUC only. No logloss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, "/tmp/testsb/analysis")
from insurer_gate import apply_gate  # noqa: E402

OUT = ROOT / "analysis" / "reverse" / "exp10_actuarial.json"
ANNIV = np.array([365.0 * k for k in range(1, 33)], dtype=np.float64)
FROZEN = [(700.0, 880.0), (1725.0, 1825.0), (9370.0, 9475.0)]
NEW2110 = (2110.0, 2210.0)


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy(np.float64)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def in_win(days: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (days >= lo) & (days < hi)


def overlaps_frozen(lo: float, hi: float) -> bool:
    for a, b in FROZEN:
        if lo < b and hi > a:
            return True
    return False


def load_oof() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    tr["id"] = tr["id"].astype(str)
    tea = pd.read_parquet(ROOT / "submissions" / "cb_teacher_oof.parquet")
    tea["id"] = tea["id"].astype(str)
    w62 = pd.read_parquet(ROOT / "submissions" / "cb_w62_oof.parquet")
    w62["id"] = w62["id"].astype(str)
    df = tr.merge(tea[["id", "pred_cb_main", "pred_cb_alt", "label"]], on="id", how="left", suffixes=("", "_y"))
    if "label_y" in df.columns:
        df["label"] = df["label"].fillna(df["label_y"])
        df = df.drop(columns=["label_y"])
    df = df.merge(w62[["id", "pred_cb_main", "pred_cb_alt"]].rename(
        columns={"pred_cb_main": "w62_main", "pred_cb_alt": "w62_alt"}
    ), on="id", how="left")
    y = df["label"].to_numpy(np.float64)
    r_m = rank01(df["pred_cb_main"].to_numpy())
    r_a = rank01(df["pred_cb_alt"].to_numpy())
    tea_w62 = 0.62 * r_m + 0.38 * r_a
    tea_max2 = np.maximum(r_m, r_a)
    w_m = rank01(df["w62_main"].to_numpy())
    w_a = rank01(df["w62_alt"].to_numpy())
    hon_w62 = 0.62 * w_m + 0.38 * w_a
    hon_max2 = np.maximum(w_m, w_a)
    best = np.load(ROOT / "analysis" / "reverse" / "best_oof.npy")
    assert len(best) == len(df)
    scores = {
        "y": y,
        "tea_w62": tea_w62,
        "tea_max2": tea_max2,
        "hon_w62": hon_w62,
        "hon_max2": hon_max2,
        "best": rank01(best),
        "days": df["days"].to_numpy(np.float64),
    }
    scores["tea_gated"] = rank01(apply_gate(tea_max2, scores["days"]))
    scores["hon_gated"] = rank01(apply_gate(hon_max2, scores["days"]))
    scores["best_gated"] = rank01(apply_gate(scores["best"], scores["days"]))
    return df, scores


def residual(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return y - p


def slice_table(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, col: str) -> list[dict]:
    resid = residual(y, p)
    rows = []
    for k, g in df.groupby(col, dropna=False):
        idx = g.index.to_numpy()
        if len(idx) < 30:
            continue
        rows.append({
            "key": str(k),
            "n": int(len(idx)),
            "rate": float(y[idx].mean()),
            "p_mean": float(p[idx].mean()),
            "resid": float(resid[idx].mean()),
        })
    rows.sort(key=lambda r: abs(r["resid"]), reverse=True)
    return rows[:12]


def anniversary_raw(days: np.ndarray, y: np.ndarray, half: float = 50.0) -> list[dict]:
    rows = []
    base = float(y.mean())
    for a in ANNIV:
        m = np.abs(days - a) < half
        n = int(m.sum())
        if n < 40:
            continue
        rate = float(y[m].mean())
        rows.append({
            "anniv_days": float(a),
            "years": round(a / 365.0, 2),
            "n": n,
            "rate": rate,
            "lift": rate / base,
            "frozen": bool(overlaps_frozen(a - half, a + half)),
        })
    rows.sort(key=lambda r: r["lift"])
    return rows


def leftover_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    d = df
    days = d["days"].to_numpy(np.float64)
    cond = d["condition"].to_numpy(np.float64)
    cond = np.where(np.isfinite(cond), cond, np.nanmedian(cond))
    src = d["source"].astype(str).to_numpy()
    # fold-unsafe rank would leak; caller must pass fold-safe pieces for CV.
    bits = {
        "t1": d["t1"].to_numpy(np.float64),
        "t2": d["t2"].to_numpy(np.float64),
        "r1": d["r1"].to_numpy(np.float64),
        "r2": d["r2"].to_numpy(np.float64),
        "c1": d["c1"].to_numpy(np.float64),
        "c2": d["c2"].to_numpy(np.float64),
        "w1": d["w1"].to_numpy(np.float64),
        "w2": d["w2"].to_numpy(np.float64),
        "w1_eq_not_w2": (d["w1"].to_numpy() == (1 - d["w2"].to_numpy())).astype(np.float64),
        "codeA": (d["code"].astype(str) == "A").astype(np.float64),
        "codeD": (d["code"].astype(str) == "D").astype(np.float64),
        "t3M": d["t3"].astype(str).str[-1].eq("M").astype(np.float64),
        "w2110": in_win(days, *NEW2110).astype(np.float64),
        "age8": (d["age_range"].to_numpy() >= 8).astype(np.float64),
        "cond_log": np.log1p(cond),
        "days_mod365": np.mod(days, 365.0) / 365.0,
        "anniv_dist": np.min(np.abs(days[:, None] - ANNIV[None, :]), axis=1) / 40.0,
        "near_anniv": (np.min(np.abs(days[:, None] - ANNIV[None, :]), axis=1) < 40.0).astype(np.float64),
        "y1": (np.abs(days - 365.0) < 50).astype(np.float64),
        "y3": (np.abs(days - 1095.0) < 50).astype(np.float64),
        "y4": (np.abs(days - 1460.0) < 50).astype(np.float64),
        "y6": (np.abs(days - 2190.0) < 50).astype(np.float64),
        "y7": (np.abs(days - 2555.0) < 50).astype(np.float64),
    }
    names = list(bits)
    X = np.column_stack([bits[k] for k in names])
    return X, names


def nested_residual_ridge(
    X: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    days: np.ndarray,
) -> dict:
    """Fit Ridge on (y - base) in-fold; blend rank(base) with rank(pred)."""
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    oof_resid = np.zeros(len(y))
    oof_blend = np.zeros(len(y))
    oof_gate2110 = np.zeros(len(y))
    oof_anniv = np.zeros(len(y))
    for tr, va in kf.split(X, y.astype(int)):
        r = y[tr] - base[tr]
        mdl = Ridge(alpha=2.0)
        mdl.fit(X[tr], r)
        pred = mdl.predict(X[va])
        oof_resid[va] = pred
        b = rank01(base[va])
        extra = rank01(pred)
        oof_blend[va] = 0.92 * b + 0.08 * extra
        g = apply_gate(base[va], days[va])
        # extra 6y pit: only if train-fold rate is clearly low
        m_tr = in_win(days[tr], *NEW2110)
        m_va = in_win(days[va], *NEW2110)
        rate = float(y[tr][m_tr].mean()) if m_tr.any() else 1.0
        g2 = g.copy()
        if rate < 0.06 and m_tr.sum() >= 40:
            g2[m_va] = g2[m_va] - 0.08
        oof_gate2110[va] = rank01(g2)
        # anniversary pits discovered on train fold (exclude frozen)
        g3 = g.copy()
        base_rate = float(y[tr].mean())
        for a in ANNIV:
            lo, hi = a - 50.0, a + 50.0
            if overlaps_frozen(lo, hi):
                continue
            mt = (np.abs(days[tr] - a) < 50.0)
            n = int(mt.sum())
            if n < 80:
                continue
            rt = float(y[tr][mt].mean())
            mv = (np.abs(days[va] - a) < 50.0)
            if rt < 0.45 * base_rate:
                floor = float(np.min(g3) - 1.0)
                g3[mv] = floor
            elif rt < 0.60 * base_rate:
                g3[mv] = g3[mv] - 0.08
            elif rt > 1.70 * base_rate:
                g3[mv] = g3[mv] + 0.05
        oof_anniv[va] = rank01(g3)
    return {
        "auc_resid_raw": auc(y, oof_resid),
        "auc_blend_08": auc(y, oof_blend),
        "auc_gate2110": auc(y, oof_gate2110),
        "auc_anniv_nested": auc(y, oof_anniv),
        "oof_resid": oof_resid,
        "oof_blend": oof_blend,
        "oof_gate2110": oof_gate2110,
        "oof_anniv": oof_anniv,
    }


def nested_pack_blend(y: np.ndarray, packs: dict[str, np.ndarray]) -> dict:
    names = list(packs)
    mat = np.column_stack([rank01(packs[k]) for k in names])
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    oof = np.zeros(len(y))
    chosen = []
    grid = []
    # coarse simplex over 2 or 3 packs
    if len(names) == 2:
        ws = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.0, 1.0)]
        grid = [dict(zip(names, w)) for w in ws]
    else:
        for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            for b in (0.0, 0.1, 0.2, 0.3):
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                grid.append({names[0]: a, names[1]: b, names[2]: max(c, 0.0)})
    for tr, va in kf.split(mat, y.astype(int)):
        best_w, best_a = grid[0], -1.0
        for w in grid:
            s = sum(w[n] * mat[tr, i] for i, n in enumerate(names))
            a = auc(y[tr], s)
            if a > best_a:
                best_a, best_w = a, w
        chosen.append(best_w)
        oof[va] = sum(best_w[n] * mat[va, i] for i, n in enumerate(names))
    mean_w = {n: float(np.mean([c[n] for c in chosen])) for n in names}
    return {"auc": auc(y, oof), "mean_weights": mean_w, "oof": oof}


def per_source_resid_te(df: pd.DataFrame, y: np.ndarray, base: np.ndarray) -> dict:
    src = df["source"].astype(str).to_numpy()
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    oof = np.zeros(len(y))
    for tr, va in kf.split(base.reshape(-1, 1), y.astype(int)):
        resid = y[tr] - base[tr]
        mu = pd.Series(resid).groupby(src[tr]).mean()
        gmu = float(resid.mean())
        add = np.array([mu.get(s, gmu) for s in src[va]], dtype=np.float64)
        oof[va] = rank01(base[va] + add)
    return {"auc": auc(y, oof), "oof": oof}


def main() -> None:
    df, sc = load_oof()
    y, days = sc["y"], sc["days"]
    report: dict = {"baselines": {}}
    for k in ("tea_w62", "tea_max2", "tea_gated", "hon_w62", "hon_max2", "hon_gated", "best", "best_gated"):
        report["baselines"][k] = auc(y, sc[k])
        print(f"baseline {k:12s} {report['baselines'][k]:.6f}", flush=True)

    p = sc["tea_gated"]
    print("\n=== residual slices vs teacher_gated ===", flush=True)
    slices = {}
    for col in ("source", "code", "version", "grades", "month", "w1", "w2", "r1", "c2"):
        tab = slice_table(df, y, p, col)
        slices[col] = tab
        if tab:
            top = tab[0]
            print(f"  {col:8s} worst {top['key']} n={top['n']} rate={top['rate']:.4f} resid={top['resid']:+.4f}", flush=True)
    report["slices"] = slices

    print("\n=== flag complementarity ===", flush=True)
    report["flags"] = {
        "w1_eq_not_w2": float((df["w1"].to_numpy() == (1 - df["w2"].to_numpy())).mean()),
        "t1_eq_not_t2": float((df["t1"].to_numpy() == (1 - df["t2"].to_numpy())).mean()),
        "c1_eq_c2": float((df["c1"].to_numpy() == df["c2"].to_numpy()).mean()),
        "codeD_eq_t3M": float(((df["code"].astype(str) == "D") == df["t3"].astype(str).str[-1].eq("M")).mean()),
        "codeD_eq_car7": float(((df["code"].astype(str) == "D") == df["source"].astype(str).str.startswith("CAR_7")).mean()),
    }
    print(report["flags"], flush=True)

    ann = anniversary_raw(days, y, 50.0)
    report["anniversary"] = ann
    print("\n=== anniversary windows (sorted by lift) ===", flush=True)
    for r in ann[:8] + ann[-4:]:
        print(f"  y={r['years']:5.1f} n={r['n']:4d} rate={r['rate']:.4f} lift={r['lift']:.3f} frozen={r['frozen']}", flush=True)

    print("\n=== extra window 2110-2210 vs frozen gates ===", flush=True)
    m = in_win(days, *NEW2110)
    report["w2110"] = {
        "n": int(m.sum()),
        "rate": float(y[m].mean()) if m.any() else None,
        "p_tea_gated": float(p[m].mean()) if m.any() else None,
        "resid": float((y[m] - p[m]).mean()) if m.any() else None,
    }
    print(report["w2110"], flush=True)

    # Spearman leftover vs residual
    X, names = leftover_matrix(df)
    resid = y - p
    spe = []
    for i, n in enumerate(names):
        if np.std(X[:, i]) < 1e-12:
            continue
        rho = float(pd.Series(X[:, i]).corr(pd.Series(resid), method="spearman"))
        spe.append({"feat": n, "spearman": rho})
    spe.sort(key=lambda r: abs(r["spearman"]), reverse=True)
    report["resid_spearman"] = spe[:20]
    print("\n=== residual Spearman (teacher_gated) ===", flush=True)
    for r in spe[:12]:
        print(f"  {r['feat']:16s} {r['spearman']:+.4f}", flush=True)

    print("\n=== nested 10-fold residual ridge / extra gates ===", flush=True)
    nest = nested_residual_ridge(X, y, sc["tea_max2"], days)
    for k in ("auc_resid_raw", "auc_blend_08", "auc_gate2110", "auc_anniv_nested"):
        print(f"  {k} {nest[k]:.6f}  vs tea_max2 {report['baselines']['tea_max2']:.6f}  vs gated {report['baselines']['tea_gated']:.6f}", flush=True)
    report["nested_ridge"] = {k: nest[k] for k in nest if not k.startswith("oof")}

    # apply frozen gate AFTER nested anniversary (honest: gate on the nested score)
    nest_anniv_then_frozen = rank01(apply_gate(nest["oof_anniv"], days))
    report["nested_ridge"]["auc_anniv_then_frozen"] = auc(y, nest_anniv_then_frozen)
    print(f"  auc_anniv_then_frozen {report['nested_ridge']['auc_anniv_then_frozen']:.6f}", flush=True)

    print("\n=== nested pack blend ===", flush=True)
    b2 = nested_pack_blend(y, {"tea_gated": sc["tea_gated"], "best_gated": sc["best_gated"]})
    b3 = nested_pack_blend(y, {"tea_gated": sc["tea_gated"], "best_gated": sc["best_gated"], "hon_gated": sc["hon_gated"]})
    print(f"  tea+best {b2['auc']:.6f} w={b2['mean_weights']}", flush=True)
    print(f"  tea+best+hon {b3['auc']:.6f} w={b3['mean_weights']}", flush=True)
    report["pack_blend_2"] = {"auc": b2["auc"], "weights": b2["mean_weights"]}
    report["pack_blend_3"] = {"auc": b3["auc"], "weights": b3["mean_weights"]}

    print("\n=== per-source residual TE on teacher_max2 ===", flush=True)
    ps = per_source_resid_te(df, y, sc["tea_max2"])
    ps_g = rank01(apply_gate(ps["oof"], days))
    report["source_resid_te"] = {"auc": ps["auc"], "auc_gated": auc(y, ps_g)}
    print(f"  {report['source_resid_te']}", flush=True)

    # coverage x frequency: w2 as cover proxy * teacher
    cover = np.clip(0.85 + 0.15 * df["w2"].to_numpy(np.float64), 0.5, 1.0)
    two = rank01(sc["tea_max2"] * cover)
    two_g = rank01(apply_gate(two, days))
    report["w2_cover_mult"] = {"auc": auc(y, two), "auc_gated": auc(y, two_g)}
    print(f"  w2_cover_mult {report['w2_cover_mult']}", flush=True)

    best_name, best_auc = max(
        [
            ("tea_gated", report["baselines"]["tea_gated"]),
            ("anniv_nested", nest["auc_anniv_nested"]),
            ("anniv_then_frozen", report["nested_ridge"]["auc_anniv_then_frozen"]),
            ("gate2110", nest["auc_gate2110"]),
            ("ridge_blend", nest["auc_blend_08"]),
            ("pack2", b2["auc"]),
            ("pack3", b3["auc"]),
            ("source_te_gated", report["source_resid_te"]["auc_gated"]),
            ("w2_cover_gated", report["w2_cover_mult"]["auc_gated"]),
        ],
        key=lambda t: t[1],
    )
    report["best_honest"] = {"name": best_name, "auc": best_auc}
    print(f"\nBEST honest {best_name} {best_auc:.6f}  current submit {report['baselines']['tea_gated']:.6f}", flush=True)
    OUT.write_text(json.dumps(report, indent=2, default=lambda o: None))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
