#!/usr/bin/env python3
"""Customer vs insurer adversarial probe (cheap: Ridge / univariate, no CatBoost).

Customer files when expected surplus > hassle:
  surplus ≈ exposure * damage * I(repair worth) * I(not denied)
Insurer pays only outside warranty/franchise windows and after moral-hazard screens.
Observed label ≈ file AND (pay or record). Additive on probability scale (RMSE).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/workspace")
OUT = ROOT / "analysis" / "adversarial"
OUT.mkdir(parents=True, exist_ok=True)

ANNIV = np.array([365.0, 730.0, 1095.0, 1460.0, 1825.0, 2190.0, 2555.0, 2920.0], dtype=np.float64)


def auc(y, s) -> float:
    s = np.asarray(s, dtype=np.float64)
    s = np.nan_to_num(s, nan=np.nanmedian(s) if np.isfinite(s).any() else 0.0)
    a = float(roc_auc_score(y, s))
    b = float(roc_auc_score(y, -s))
    return max(a, b)


def both(y, s):
    s = np.asarray(s, dtype=np.float64)
    s = np.nan_to_num(s, nan=0.0)
    return float(roc_auc_score(y, s)), float(roc_auc_score(y, -s))


def rank01(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def attach(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["car"] = d["source"].astype(str).str.split("|").str[0]
    med = d.groupby("source")["condition"].transform("median")
    g = float(d["condition"].median())
    d["condition_f"] = d["condition"].fillna(med).fillna(g)
    d["cond_rk"] = d.groupby("source")["condition_f"].rank(pct=True)
    sm = d.groupby("source")["condition_f"].transform("median").replace(0, np.nan)
    d["cond_r"] = d["condition_f"] / sm.fillna(sm.median())
    d["rate"] = d["days"] * (1.0 - d["cond_rk"])
    d["ratio"] = d["days"] / d["cond_r"].clip(lower=1e-6)
    d["age_num"] = pd.to_numeric(d["age_range"], errors="coerce").fillna(0.0)
    d["age8"] = (d["age_num"] >= 8).astype(np.float64)
    days = d["days"].to_numpy(np.float64)
    rk = d["cond_rk"].to_numpy(np.float64)
    car = d["car"].to_numpy()

    deny = ((days >= 700) & (days < 880)) | ((days >= 1725) & (days < 1825))
    dump = (days >= 9370) & (days < 9475)
    new = days < 100.0
    d["deny"] = deny.astype(np.float64)
    d["dump"] = dump.astype(np.float64)
    d["claim_util"] = d["rate"] * (1.0 - d["deny"])
    d["expos_net"] = days * (1.0 - d["deny"]) * (1.0 + 0.5 * d["age8"])
    d["dump_poor"] = dump.astype(np.float64) * (1.0 - rk)
    d["lemon"] = (new & (rk < 0.15)).astype(np.float64)
    d["new_good"] = np.where(new, rk, 0.0)
    d["age8_poor"] = d["age8"] * (1.0 - rk)
    mid = 1.0 - 4.0 * (rk - 0.5) ** 2
    d["repair_mid"] = np.clip(mid, 0.0, 1.0)
    d["repair_car10"] = d["repair_mid"] * (car == "CAR_10").astype(np.float64)
    d["mono_car1"] = (1.0 - rk) * (car == "CAR_1").astype(np.float64)
    d["tls_screen"] = ((rk < 0.10) & np.isin(car, ["CAR_7", "CAR_10"])).astype(np.float64)
    d["rev_car7"] = rk * (car == "CAR_7").astype(np.float64)
    dist = np.min(np.abs(days[:, None] - ANNIV[None, :]), axis=1)
    d["anniv_dist"] = dist
    d["near_anniv"] = (dist < 40.0).astype(np.float64)
    d["days_mod365"] = np.mod(days, 365.0)
    d["anniv_cos"] = np.cos(2.0 * np.pi * days / 365.0)
    d["anniv_sin"] = np.sin(2.0 * np.pi * days / 365.0)
    d["w_safe750"] = ((days >= 700) & (days < 880)).astype(np.float64)
    d["w_safe1750"] = ((days >= 1725) & (days < 1825)).astype(np.float64)
    d["w_hot9370"] = dump.astype(np.float64)
    d["pow_rate15"] = np.power(np.clip(days, 0.0, None), 1.5) * np.power(np.clip(1.0 - rk, 0.0, None), 1.23)
    d["util_pow"] = d["pow_rate15"] * (1.0 - d["deny"]) + 2.0 * d["dump_poor"]
    for c in ("w1", "w2", "c1", "c2", "t1", "t2", "r1", "r2"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["cover_sum"] = d["w1"] + d["w2"] + d["c1"] + d["c2"]
    d["flag_and"] = ((d["t1"] > 0.5) & (d["t2"] > 0.5)).astype(np.float64)
    return d


def ridge_oof(X: np.ndarray, y: np.ndarray, nfold: int = 10, alpha: float = 2.0) -> np.ndarray:
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2026)
    for tr, va in skf.split(X, y):
        m = Ridge(alpha=alpha, fit_intercept=True)
        Xt = np.nan_to_num(X[tr], nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(X[va], nan=0.0, posinf=0.0, neginf=0.0)
        m.fit(Xt, y[tr].astype(np.float64))
        oof[va] = m.predict(Xv)
    return oof


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    y = train["label"].to_numpy(np.int32)
    d = attach(train)

    uni = {}
    cols = [
        "days", "rate", "ratio", "cond_rk", "claim_util", "expos_net", "dump_poor",
        "lemon", "new_good", "age8_poor", "repair_mid", "repair_car10", "mono_car1",
        "tls_screen", "rev_car7", "anniv_dist", "near_anniv", "days_mod365",
        "anniv_cos", "anniv_sin", "deny", "w_safe750", "w_safe1750", "w_hot9370",
        "pow_rate15", "util_pow", "w1", "w2", "c1", "c2", "t1", "t2", "r1", "r2",
        "cover_sum", "flag_and", "age8",
    ]
    for c in cols:
        uni[c] = {"max": auc(y, d[c]), "raw": both(y, d[c])[0], "neg": both(y, d[c])[1]}

    # source-specific deny / dump rates
    src_win = []
    for src, g in d.groupby("source"):
        yy = g["label"].to_numpy()
        src_win.append(
            {
                "source": src,
                "n": int(len(g)),
                "rate": float(yy.mean()),
                "deny_n": int(g["deny"].sum()),
                "deny_rate": float(g.loc[g["deny"] > 0.5, "label"].mean()) if g["deny"].sum() else None,
                "dump_n": int(g["dump"].sum()),
                "dump_rate": float(g.loc[g["dump"] > 0.5, "label"].mean()) if g["dump"].sum() else None,
                "lemon_n": int(g["lemon"].sum()),
                "lemon_rate": float(g.loc[g["lemon"] > 0.5, "label"].mean()) if g["lemon"].sum() else None,
            }
        )

    # coverage flags vs deny windows
    flag_tab = {}
    for c in ("w1", "w2", "c1", "c2", "t1", "t2", "r1", "r2"):
        flag_tab[c] = d.groupby(c)["label"].agg(["mean", "count"]).reset_index().to_dict(orient="records")

    adv_cols = [
        "claim_util", "expos_net", "dump_poor", "lemon", "new_good", "age8_poor",
        "repair_car10", "mono_car1", "tls_screen", "rev_car7", "anniv_dist",
        "near_anniv", "anniv_cos", "anniv_sin", "deny", "w_safe750", "w_safe1750",
        "w_hot9370", "util_pow", "age8", "days", "rate", "cond_rk",
    ]
    X = d[adv_cols].to_numpy(np.float64)
    # standardize columns for Ridge
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    oof_adv = ridge_oof(Xs, y, nfold=10, alpha=2.0)
    auc_adv = float(roc_auc_score(y, oof_adv))

    # add source/region one-hot (low card)
    src_oh = pd.get_dummies(d["source"].astype(str), prefix="s", drop_first=True).to_numpy(np.float64)
    reg_oh = pd.get_dummies(d["region"].astype(str), prefix="r", drop_first=True).to_numpy(np.float64)
    X2 = np.hstack([Xs, src_oh, reg_oh])
    oof_adv2 = ridge_oof(X2, y, nfold=10, alpha=4.0)
    auc_adv2 = float(roc_auc_score(y, oof_adv2))

    teacher_path = ROOT / "submissions" / "cb_teacher_oof.csv"
    resid = {}
    if teacher_path.is_file():
        te = pd.read_csv(teacher_path)
        # teacher csv may be id,label,pred_*
        pred_col = None
        for c in ("pred_cb_w62", "pred_blend", "pred_cb_main", "label"):
            if c in te.columns and c != "label":
                pred_col = c
                break
        # parquet preferred
        pq = ROOT / "submissions" / "cb_teacher_oof.parquet"
        if pq.is_file():
            te = pd.read_parquet(pq)
            pred_col = "pred_cb_w62" if "pred_cb_w62" in te.columns else "pred_cb_main"
        tpred = te[pred_col].to_numpy(np.float64)
        ty = te["label"].to_numpy() if "label" in te.columns else y
        # align by id if possible
        if "id" in te.columns:
            m = pd.DataFrame({"id": train["id"].astype(str), "y": y, "adv": oof_adv2, "adv0": oof_adv})
            t2 = te.copy()
            t2["id"] = t2["id"].astype(str)
            m = m.merge(t2[["id", pred_col]], on="id", how="left")
            tpred = m[pred_col].to_numpy(np.float64)
            ty = m["y"].to_numpy()
            oof_adv2_a = m["adv"].to_numpy()
            oof_adv_a = m["adv0"].to_numpy()
        else:
            oof_adv2_a, oof_adv_a, ty = oof_adv2, oof_adv, y
        rt = rank01(tpred)
        ra = rank01(oof_adv2_a)
        corr = float(np.corrcoef(rt, ra)[0, 1])
        # residual AUC: does adv rank add?
        grid = []
        for w in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
            s = (1.0 - w) * rt + w * ra
            grid.append({"w_adv": w, "auc": float(roc_auc_score(ty, s))})
        # max2
        mx = np.maximum(rt, ra)
        grid.append({"w_adv": "max2", "auc": float(roc_auc_score(ty, mx))})
        resid = {
            "teacher_col": pred_col,
            "teacher_auc": float(roc_auc_score(ty, tpred)),
            "adv_auc": float(roc_auc_score(ty, oof_adv_a)),
            "adv2_auc": float(roc_auc_score(ty, oof_adv2_a)),
            "rank_corr": corr,
            "fuse_grid": grid,
        }

    # per-car inverted-U vs monotone: claim-utility story
    shape = []
    for src, g in d.groupby("source"):
        q = pd.qcut(g["cond_rk"], 5, duplicates="drop")
        rates = g.groupby(q, observed=False)["label"].mean().to_list()
        shape.append({"source": src, "n": int(len(g)), "q5_rates": rates})

    payload = {
        "univariate": uni,
        "ridge_adv_only": auc_adv,
        "ridge_adv_src_reg": auc_adv2,
        "n_adv_cols": len(adv_cols),
        "src_windows": src_win,
        "flag_tab": flag_tab,
        "car_cond_q5": shape,
        "teacher_fuse": resid,
        "theory": {
            "customer": "file when surplus=exposure*damage*(1-deny) high; lemon new cars; age8 last-chance; dump window before cover ends; CAR_10 mid-condition repair-worth",
            "insurer": "deny warranty [700,880] and [1725,1825] (~2y/5y); screen total-loss CAR_7/10 worst decile; underwrite by source; anniversary inspection",
        },
    }
    (OUT / "probe.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    top = sorted(uni.items(), key=lambda kv: -kv[1]["max"])[:20]
    print("top univariate", [(k, round(v["max"], 4)) for k, v in top], flush=True)
    print(f"ridge_adv={auc_adv:.5f} ridge_adv+src+reg={auc_adv2:.5f}", flush=True)
    if resid:
        print("teacher fuse", json.dumps(resid["fuse_grid"], ensure_ascii=False), flush=True)
        print("rank_corr", resid.get("rank_corr"), "teacher", resid.get("teacher_auc"), flush=True)
    print("wrote", OUT / "probe.json", flush=True)


if __name__ == "__main__":
    main()
