#!/usr/bin/env python3
"""Exp4: rule mining on half the data, confirm on the other half (scan-bias control)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import auc, dump_json, fill_condition, load_train, skf, te_val_only

RNG = np.random.RandomState(2026)


def scan_days_windows(days, y, min_n=40):
    days = np.asarray(days, float)
    y = np.asarray(y)
    base = float(y.mean())
    hits = []
    lo, hi = float(days.min()), float(days.max())
    for w in (50, 80, 100, 150, 200, 300, 400):
        start = lo
        while start < hi:
            m = (days >= start) & (days < start + w)
            n = int(m.sum())
            if n >= min_n:
                k = int(y[m].sum())
                rate = k / n
                # two-sided vs baseline
                p = binomtest(k, n, p=base, alternative="two-sided").pvalue
                if p < 0.01 and abs(rate - base) >= 0.04:
                    hits.append(
                        {
                            "lo": float(start),
                            "hi": float(start + w),
                            "w": w,
                            "n": n,
                            "rate": rate,
                            "lift": rate / base,
                            "p": float(p),
                        }
                    )
            start += w / 2.0
    hits.sort(key=lambda z: z["p"])
    return hits[:40]


def scan_cond_thresholds(cond, y, src, min_n=30):
    cond = np.asarray(cond, float)
    y = np.asarray(y)
    src = np.asarray(src)
    base = float(y.mean())
    hits = []
    for s in np.unique(src):
        ms = src == s
        if ms.sum() < 80:
            continue
        cs, ys = cond[ms], y[ms]
        qs = np.quantile(cs, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        for q in qs:
            for side, m in [("le", cs <= q), ("ge", cs >= q)]:
                n = int(m.sum())
                if n < min_n:
                    continue
                k = int(ys[m].sum())
                rate = k / n
                p = binomtest(k, n, p=float(ys.mean()), alternative="two-sided").pvalue
                if p < 0.05 and abs(rate - float(ys.mean())) >= 0.04:
                    hits.append(
                        {
                            "source": str(s),
                            "side": side,
                            "thr": float(q),
                            "n": n,
                            "rate": rate,
                            "src_base": float(ys.mean()),
                            "p": float(p),
                        }
                    )
    hits.sort(key=lambda z: z["p"])
    return hits[:40]


def confirm_window(days, y, rule):
    m = (days >= rule["lo"]) & (days < rule["hi"])
    n = int(m.sum())
    if n < 20:
        return None
    rate = float(y[m].mean())
    base = float(y.mean())
    p = binomtest(int(y[m].sum()), n, p=base, alternative="two-sided").pvalue
    same_dir = np.sign(rate - base) == np.sign(rule["rate"] - rule["rate"] / max(rule["lift"], 1e-6) * 1) or (
        (rate - base) * (rule["rate"] - base) > 0
    )
    # simpler: same side of baseline
    same_dir = (rate - base) * (rule["rate"] - (rule["rate"] / rule["lift"] if rule["lift"] else base)) > 0
    same_dir = (rate < base) == (rule["rate"] < (rule["n"] and True))
    # use stored rate vs 0.10
    same_dir = (rate - 0.1002) * (rule["rate"] - 0.1002) > 0
    return {"n": n, "rate": rate, "p": float(p), "same_dir": bool(same_dir), "confirmed": bool(same_dir and p < 0.1)}


def main():
    df, y = load_train()
    # 50/50 stratified split for discovery/confirm
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=2026)
    disc_i, conf_i = next(sss.split(df, y))
    disc, conf = df.iloc[disc_i], df.iloc[conf_i]
    y_d, y_c = y[disc_i], y[conf_i]
    cond_d = disc["condition"].fillna(disc.groupby("source")["condition"].transform("median"))
    cond_d = cond_d.fillna(disc["condition"].median()).to_numpy()
    cond_c = conf["condition"].fillna(disc.groupby("source")["condition"].median())
    cond_c = cond_c.fillna(disc["condition"].median()).to_numpy()

    day_hits = scan_days_windows(disc["days"].to_numpy(), y_d)
    confirmed_days = []
    for r in day_hits:
        c = confirm_window(conf["days"].to_numpy(), y_c, r)
        if c is None:
            continue
        rec = {**r, "confirm": c}
        confirmed_days.append(rec)

    cond_hits = scan_cond_thresholds(cond_d, y_d, disc["source"].astype(str).to_numpy())
    confirmed_cond = []
    for r in cond_hits:
        msrc = conf["source"].astype(str).to_numpy() == r["source"]
        cc = cond_c[msrc]
        yc = y_c[msrc]
        m = cc <= r["thr"] if r["side"] == "le" else cc >= r["thr"]
        n = int(m.sum())
        if n < 15:
            continue
        rate = float(yc[m].mean())
        same = (rate - float(yc.mean())) * (r["rate"] - r["src_base"]) > 0
        p = binomtest(int(yc[m].sum()), n, p=float(yc.mean()), alternative="two-sided").pvalue
        confirmed_cond.append(
            {**r, "confirm": {"n": n, "rate": rate, "p": float(p), "same_dir": bool(same), "confirmed": bool(same and n >= 20)}}
        )

    # 2D rule: new car + not-worst condition
    days_d = disc["days"].to_numpy()
    days_c = conf["days"].to_numpy()
    two_d = []
    for dthr in (50, 100, 200, 400, 800, 1200):
        for cq in (0.1, 0.2, 0.3, 0.5):
            thr = np.quantile(cond_d, cq)
            m = (days_d < dthr) & (cond_d > thr)
            n = int(m.sum())
            if n < 40:
                continue
            rate = float(y_d[m].mean())
            if abs(rate - y_d.mean()) < 0.04:
                continue
            mc = (days_c < dthr) & (cond_c > thr)
            nc = int(mc.sum())
            if nc < 20:
                continue
            rc = float(y_c[mc].mean())
            same = (rate - y_d.mean()) * (rc - y_c.mean()) > 0
            two_d.append(
                {
                    "days_lt": dthr,
                    "cond_gt_q": cq,
                    "disc_n": n,
                    "disc_rate": rate,
                    "conf_n": nc,
                    "conf_rate": rc,
                    "confirmed": bool(same and abs(rc - y_c.mean()) >= 0.03),
                }
            )

    conf_days = [r for r in confirmed_days if r["confirm"]["confirmed"]]
    conf_cond = [r for r in confirmed_cond if r["confirm"]["confirmed"]]
    conf_2d = [r for r in two_d if r["confirmed"]]

    # Honest OOF of a tiny rule-score: sum of confirmed window indicators with signed lift
    # Build score on full data with 10-fold: discover on train fold, apply on val (nested, no scan leak)
    oof = np.zeros(len(y))
    for tr_i, va_i in skf(y):
        trn, val = df.iloc[tr_i], df.iloc[va_i]
        ytr = y[tr_i]
        hits = scan_days_windows(trn["days"].to_numpy(), ytr, min_n=50)
        # keep top 8 lowest p
        score = np.zeros(len(va_i))
        days_va = val["days"].to_numpy()
        base = float(ytr.mean())
        for r in hits[:8]:
            m = (days_va >= r["lo"]) & (days_va < r["hi"])
            score[m] += np.sign(r["rate"] - base) * min(abs(r["rate"] - base), 0.2)
        oof[va_i] = score

    report = {
        "disc_day_hits": day_hits[:15],
        "confirmed_days": conf_days,
        "n_day_scanned": len(day_hits),
        "n_day_confirmed": len(conf_days),
        "confirmed_cond": conf_cond[:20],
        "n_cond_confirmed": len(conf_cond),
        "confirmed_2d": conf_2d,
        "rule_score_oof_auc": auc(y, oof),
        "note": "Windows/thresholds must confirm on held-out half. OOF rule-score uses train-fold discovery only.",
    }
    print("=== EXP4 confirmed days windows ===")
    for r in conf_days[:12]:
        print(
            f"  [{r['lo']:.0f},{r['hi']:.0f}) disc_n={r['n']} disc_rate={r['rate']:.3f} "
            f"conf_n={r['confirm']['n']} conf_rate={r['confirm']['rate']:.3f} p={r['confirm']['p']:.3g}"
        )
    print("confirmed cond rules", len(conf_cond), "2d", len(conf_2d))
    print("rule score OOF AUC", report["rule_score_oof_auc"])
    dump_json("exp4_rules.json", report)
    print("WROTE exp4_rules.json")


if __name__ == "__main__":
    main()
