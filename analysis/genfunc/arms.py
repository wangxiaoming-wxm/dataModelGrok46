"""Fold-internal generating-process arms. Identity-link / RMSE. Spark-portable ops."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import (
    apply_bins,
    fill_condition,
    ohe_levels_apply,
    per_source_rank,
    power_score,
    qcut_apply,
    quantile_knots,
    source_b_map,
    source_median_map,
    standardize_apply,
    standardize_fit,
    te_apply_from_group,
    te_group,
    te_loo_from_group,
    te_pair,
    window_matrix,
)
from exp1_spline import LARGE_N, spline_design
from exp2_car_shapes import car_terms
from exp3_2d_smooth import NEIGH, fill_grid, make_global, predict_table, src_index

TE_KEYS = [
    ("src", 10.0),
    ("reg", 10.0),
    ("src_reg", 15.0),
    ("src_age", 12.0),
    ("reg_age", 12.0),
    ("src_cq", 15.0),
    ("src_dq", 12.0),
    ("cq_dq", 12.0),
    ("src_cq_dq", 10.0),
    ("src_rq", 15.0),
    ("src_tq", 15.0),
    ("reg_dq", 15.0),
    ("reg_cq_dq", 20.0),
    ("src_cq_age", 18.0),
]


def _fill_cond(trn, val=None):
    if val is None:
        (cond_tr,) = fill_condition(trn)
        return cond_tr
    return fill_condition(trn, val)


def make_keys(trn, val, cond_tr, cond_va, rk_tr, rk_va):
    dtr, dva, dbins = qcut_apply(trn["days"], val["days"], 5)
    ctr, cva, cbins = qcut_apply(cond_tr, cond_va, 10)
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    med = pd.DataFrame({"s": src_tr, "c": cond_tr}).groupby("s")["c"].median()
    cr_tr = cond_tr / pd.Series(src_tr).map(med).astype(float).to_numpy()
    cr_va = cond_va / pd.Series(src_va).map(med).fillna(float(np.nanmedian(med))).astype(float).to_numpy()
    rate_tr = trn["days"].to_numpy(float) * (1.0 - rk_tr)
    rate_va = val["days"].to_numpy(float) * (1.0 - rk_va)
    rq_tr, rq_va, _ = qcut_apply(trn["days"].to_numpy(float) / np.clip(cr_tr, 1e-6, None), val["days"].to_numpy(float) / np.clip(cr_va, 1e-6, None), 10)
    tq_tr, tq_va, _ = qcut_apply(rate_tr, rate_va, 10)

    def S(a):
        return np.asarray(a).astype(str)

    tr = {
        "src": S(src_tr),
        "reg": S(trn["region"]),
        "age": S(trn["age_range"].astype(int)),
        "cq": S(ctr),
        "dq": S(dtr),
        "rq": S(rq_tr),
        "tq": S(tq_tr),
    }
    va = {
        "src": S(src_va),
        "reg": S(val["region"]),
        "age": S(val["age_range"].astype(int)),
        "cq": S(cva),
        "dq": S(dva),
        "rq": S(rq_va),
        "tq": S(tq_va),
    }
    for d in (tr, va):
        d["src_reg"] = d["src"] + "|" + d["reg"]
        d["src_age"] = d["src"] + "|" + d["age"]
        d["reg_age"] = d["reg"] + "|" + d["age"]
        d["src_cq"] = d["src"] + "|" + d["cq"]
        d["src_dq"] = d["src"] + "|" + d["dq"]
        d["cq_dq"] = d["cq"] + "|" + d["dq"]
        d["src_cq_dq"] = d["src"] + "|" + d["cq"] + "|" + d["dq"]
        d["src_rq"] = d["src"] + "|" + d["rq"]
        d["src_tq"] = d["src"] + "|" + d["tq"]
        d["reg_dq"] = d["reg"] + "|" + d["dq"]
        d["reg_cq_dq"] = d["reg"] + "|" + d["cq"] + "|" + d["dq"]
        d["src_cq_age"] = d["src"] + "|" + d["cq"] + "|" + d["age"]
    return tr, va, dtr, dva, ctr, cva, dbins, cbins


def ordered_te(keys, y, n_perm=4, m=20.0, seed=0):
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=float)
    n = len(y)
    prior = float(y.mean()) if n else 0.1
    _, inv = np.unique(keys, return_inverse=True)
    nkey = int(inv.max()) + 1 if n else 1
    acc = np.zeros(n)
    rng = np.random.RandomState(seed)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        sm = np.zeros(nkey)
        ct = np.zeros(nkey)
        enc = np.empty(n)
        for i in perm:
            k = inv[i]
            enc[i] = (sm[k] + prior * m) / (ct[k] + m)
            sm[k] += y[i]
            ct[k] += 1.0
        acc += enc
    return acc / n_perm


def fit_spline_car(trn, val, ytr, src_levels, reg_levels, alpha=20.0):
    cond_tr, cond_va = _fill_cond(trn, val)
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
    rk_tr = per_source_rank(src_tr, cond_tr, src_tr, cond_tr)
    rk_va = per_source_rank(src_va, cond_va, src_tr, cond_tr)
    dk = quantile_knots(days_tr, 8)
    gk = quantile_knots(rk_tr, 6)
    large = [s for s in src_levels if int((src_tr == s).sum()) >= LARGE_N]
    extra_tr, _ = car_terms(src_tr, rk_tr)
    extra_va, _ = car_terms(src_va, rk_va)
    Xtr = spline_design(days_tr, rk_tr, src_tr, trn["region"].astype(str), trn["age_range"], dk, gk, src_levels, reg_levels, large, extra_tr)
    Xva = spline_design(days_va, rk_va, src_va, val["region"].astype(str), val["age_range"], dk, gk, src_levels, reg_levels, large, extra_va)
    Xs, mu, sd = standardize_fit(Xtr)
    m = Ridge(alpha=alpha)
    m.fit(Xs, ytr)
    pred_va = m.predict(standardize_apply(Xva, mu, sd))
    pred_tr = m.predict(Xs)
    pack = {
        "kind": "spline_car",
        "days_knots": dk.tolist(),
        "rk_knots": gk.tolist(),
        "large": large,
        "src_levels": src_levels,
        "reg_levels": reg_levels,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "coef": m.coef_.tolist(),
        "intercept": float(m.intercept_),
        "alpha": alpha,
        "med_map": source_median_map(src_tr, cond_tr),
        "glob_cond": float(np.median(cond_tr)),
        "cond_ref_src": src_tr.tolist(),
        "cond_ref_val": cond_tr.tolist(),
    }
    return pred_tr, pred_va, pack, (cond_tr, cond_va, rk_tr, rk_va)


def fit_table8(trn, val, ytr, cond_tr, cond_va, dtr, dva, ctr, cva, src_levels_tab, alpha_glob=0.4):
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    idx_tr = src_index(src_tr, src_levels_tab)
    sm, ct = fill_grid(idx_tr, dtr, ctr, ytr, len(src_levels_tab), 5, 10)
    prior = float(np.mean(ytr))
    neigh = NEIGH[2]
    glob = make_global(sm, ct, prior, 10.0, neigh)
    pred_va = predict_table(src_va, dva, cva, src_levels_tab, sm, ct, prior, 10.0, neigh, alpha_glob, glob)
    pred_tr = predict_table(src_tr, dtr, ctr, src_levels_tab, sm, ct, prior, 10.0, neigh, alpha_glob, glob)
    pack = {
        "kind": "table8",
        "sm": sm.tolist(),
        "ct": ct.tolist(),
        "prior": prior,
        "m": 10.0,
        "neigh": neigh,
        "alpha_glob": alpha_glob,
        "src_levels_tab": src_levels_tab,
    }
    return pred_tr, pred_va, pack


def fit_te_ridge(km_tr, km_va, ytr, table_tr=None, table_va=None, use_ordered=True, alpha=18.0, seed=0):
    cols_tr, cols_va, maps = [], [], {}
    for name, m in TE_KEYS:
        g = te_group(km_tr[name], ytr)
        prior = float(np.mean(ytr))
        if use_ordered:
            tr_enc = ordered_te(km_tr[name], ytr, n_perm=4, m=m, seed=seed + (sum(ord(c) for c in name) % 997))
        else:
            tr_enc = te_loo_from_group(km_tr[name], ytr, g, prior, m)
        va_enc = te_apply_from_group(km_va[name], g, prior, m)
        cols_tr.append(tr_enc)
        cols_va.append(va_enc)
        maps[name] = {
            "m": m,
            "prior": prior,
            "stats": {str(k): {"sum": float(r["sum"]), "count": int(r["count"])} for k, r in g.iterrows()},
        }
    if table_tr is not None:
        cols_tr.append(np.asarray(table_tr, dtype=float))
        cols_va.append(np.asarray(table_va, dtype=float))
    Xtr = np.column_stack(cols_tr)
    Xva = np.column_stack(cols_va)
    model = Ridge(alpha=alpha)
    model.fit(Xtr, ytr)
    pack = {
        "kind": "te_ridge",
        "maps": maps,
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "alpha": alpha,
        "use_ordered": use_ordered,
        "has_table": table_tr is not None,
        "key_order": [n for n, _ in TE_KEYS] + (["table8"] if table_tr is not None else []),
    }
    return model.predict(Xtr), model.predict(Xva), pack


def fit_power_ridge(trn, val, ytr, cond_tr, cond_va, rk_tr, rk_va, src_levels, reg_levels, a=1.5, c=0.5, alpha=4.0):
    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
    bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
    sc_tr = power_score(days_tr, cond_tr, rk_tr, src_tr, a, c, bmap) / 1e5
    sc_va = power_score(days_va, cond_va, rk_va, src_va, a, c, bmap) / 1e5
    Wtr, _ = window_matrix(days_tr)
    Wva, _ = window_matrix(days_va)
    age8_tr = (trn["age_range"].to_numpy(float) >= 8).astype(float)[:, None]
    age8_va = (val["age_range"].to_numpy(float) >= 8).astype(float)[:, None]
    Xtr = np.hstack([sc_tr[:, None], np.log1p(np.clip(sc_tr, 0, None))[:, None], ohe_levels_apply(src_tr, src_levels), ohe_levels_apply(trn["region"].astype(str), reg_levels), age8_tr, Wtr])
    Xva = np.hstack([sc_va[:, None], np.log1p(np.clip(sc_va, 0, None))[:, None], ohe_levels_apply(src_va, src_levels), ohe_levels_apply(val["region"].astype(str), reg_levels), age8_va, Wva])
    Xs, mu, sd = standardize_fit(Xtr)
    m = Ridge(alpha=alpha)
    m.fit(Xs, ytr)
    pack = {"kind": "power", "a": a, "c": c, "bmap": bmap, "mu": mu.tolist(), "sd": sd.tolist(), "coef": m.coef_.tolist(), "intercept": float(m.intercept_)}
    return m.predict(Xs), m.predict(standardize_apply(Xva, mu, sd)), pack


def lgb_dual(trn, val, ytr, cond_tr, cond_va, rk_tr, rk_va, km_tr, km_va, src_levels, reg_levels, rounds=400, seed=2026):
    import lightgbm as lgb

    src_tr = trn["source"].astype(str).to_numpy()
    src_va = val["source"].astype(str).to_numpy()
    days_tr, days_va = trn["days"].to_numpy(float), val["days"].to_numpy(float)
    med = pd.DataFrame({"s": src_tr, "c": cond_tr}).groupby("s")["c"].median()
    cr_tr = cond_tr / pd.Series(src_tr).map(med).astype(float).to_numpy()
    cr_va = cond_va / pd.Series(src_va).map(med).fillna(float(np.nanmedian(med))).astype(float).to_numpy()
    extra_tr, _ = car_terms(src_tr, rk_tr)
    extra_va, _ = car_terms(src_va, rk_va)
    Wtr, _ = window_matrix(days_tr)
    Wva, _ = window_matrix(days_va)
    bmap = source_b_map(src_tr, days_tr, cond_tr, ytr)
    pow_tr = power_score(days_tr, cond_tr, rk_tr, src_tr, 1.5, 0.5, bmap) / 1e5
    pow_va = power_score(days_va, cond_va, rk_va, src_va, 1.5, 0.5, bmap) / 1e5

    def pack_num(days, cond, rk, cr, extra, W, df, pows):
        return np.column_stack(
            [
                days,
                np.log1p(days),
                np.sqrt(np.clip(days, 0, None)),
                cond,
                cr,
                rk,
                days / np.clip(cr, 1e-6, None),
                days * (1.0 - rk),
                days / np.sqrt(np.clip(cond, 1e-6, None)),
                (rk - 0.5) ** 2,
                extra,
                W,
                pows,
                df["age_range"].to_numpy(float),
                (df["age_range"].to_numpy() >= 8).astype(float),
                df["x20"].to_numpy(float),
                df["V"].to_numpy(float),
                df["cc"].to_numpy(float),
            ]
        )

    num_tr = pack_num(days_tr, cond_tr, rk_tr, cr_tr, extra_tr, Wtr, trn, pow_tr)
    num_va = pack_num(days_va, cond_va, rk_va, cr_va, extra_va, Wva, val, pow_va)

    def codes(tr_keys, va_keys):
        uniq = {k: i for i, k in enumerate(pd.Index(tr_keys).unique())}
        return (
            pd.Series(tr_keys).map(uniq).fillna(-1).to_numpy(dtype=float),
            pd.Series(va_keys).map(uniq).fillna(-1).to_numpy(dtype=float),
        )

    src_codes = {s: i for i, s in enumerate(src_levels)}
    reg_codes = {s: i for i, s in enumerate(reg_levels)}
    cat_parts_tr = [
        pd.Series(src_tr).map(src_codes).fillna(-1).to_numpy(dtype=float),
        pd.Series(trn["region"].astype(str)).map(reg_codes).fillna(-1).to_numpy(dtype=float),
        trn["age_range"].to_numpy(float),
    ]
    cat_parts_va = [
        pd.Series(src_va).map(src_codes).fillna(-1).to_numpy(dtype=float),
        pd.Series(val["region"].astype(str)).map(reg_codes).fillna(-1).to_numpy(dtype=float),
        val["age_range"].to_numpy(float),
    ]
    for name in ("src_cq", "src_dq", "src_reg", "src_age", "reg_age", "src_rq"):
        a, b = codes(km_tr[name], km_va[name])
        cat_parts_tr.append(a)
        cat_parts_va.append(b)
    cat_tr = np.column_stack(cat_parts_tr)
    cat_va = np.column_stack(cat_parts_va)
    nnum = num_tr.shape[1]
    Htr = np.hstack([num_tr, cat_tr])
    Hva = np.hstack([num_va, cat_va])
    cat_idx = list(range(nnum, nnum + cat_tr.shape[1]))

    def train_pred(rounds, leaves, lr, sd):
        dtr = lgb.Dataset(Htr, ytr, categorical_feature=cat_idx, free_raw_data=False)
        params = dict(
            objective="regression",
            metric="rmse",
            learning_rate=lr,
            num_leaves=leaves,
            min_data_in_leaf=80,
            feature_fraction=0.75,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=8.0,
            verbose=-1,
            seed=sd,
            num_threads=1,
        )
        m = lgb.train(params, dtr, num_boost_round=rounds)
        return m.predict(Hva)

    p1 = train_pred(rounds, 31, 0.03, seed)
    p2 = train_pred(max(250, rounds - 50), 24, 0.04, seed + 17)
    return p1, p2


def rank01(a):
    return pd.Series(np.asarray(a, dtype=float)).rank(pct=True).to_numpy()


def fuse_rank(parts: dict, weights: dict) -> np.ndarray:
    s = np.zeros(len(next(iter(parts.values()))))
    wsum = 0.0
    for k, w in weights.items():
        if k not in parts:
            continue
        s = s + float(w) * rank01(parts[k])
        wsum += float(w)
    return s / wsum if wsum else s


def fuse_linear(parts: dict, weights: dict) -> np.ndarray:
    s = np.zeros(len(next(iter(parts.values()))))
    wsum = 0.0
    for k, w in weights.items():
        if k not in parts:
            continue
        s = s + float(w) * np.asarray(parts[k], dtype=float)
        wsum += float(w)
    return s / wsum if wsum else s
