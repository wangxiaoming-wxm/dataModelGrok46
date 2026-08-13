"""特征工程: cond_r 世界 (build_main) + rate 世界 (build_alt).

两个特征世界的核心差异:
  - build_main: condition 按 source 中位数标准化 → cond_r, 再衍生 ratio/days 分箱
  - build_alt:  condition 按 source 百分位排名 → rank, 再衍生 rate = days*(1-rank)

两者共享: 二值列 (t1..w2), region/source/month/version 等类别列,
以及大量手工类别交叉 (cross 函数).

特征统计:
  build_main: ~70 数值列 + ~80 类别列 (含交叉) = ~121 总特征, 81 类别
  build_alt:  ~50 数值列 + ~60 类别列 = ~100 总特征

设计依据: v6 最优配置, RATIO_COLS=['V','cc','max_g','x17','x14'].
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BIN_COLS = ["t1", "t2", "r1", "r2", "c1", "c2", "w1", "w2"]
GRADE_MAP = {"s": 1, "ss": 2, "sss": 3}
DAYS_FX = np.array([700, 2500, 5000, 7000, 9000, 10000], dtype=float)
QUANTS = (5, 10, 20, 40)
ALT_QUANTS = (7, 13, 25)
RATIO_COLS = ['V', 'cc', 'max_g', 'x17', 'x14']


def _qbins(v, e):
    return np.digitize(np.asarray(v, dtype=float), e)


# ======================= 臂 1: cond_r 世界 =======================

def fit_edges_main(df):
    """在全集 (train+test) 上拟合分箱边界. 必须在 build_main 之前调用."""
    scale = df.groupby("source")["condition"].median()
    cond = pd.to_numeric(df["condition"])
    days = pd.to_numeric(df["days"])
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)
    edges = {"__scale__": scale}
    for n in QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"c_{n}"] = np.quantile(cond.dropna(), qs)
        edges[f"cr_{n}"] = np.quantile(cond_r, qs)
        edges[f"ra_{n}"] = np.quantile(ratio, qs)
    return edges


def build_main(df, edges):
    """cond_r 世界: ~121 特征, 81 类别. 用于 arm1 (Ordered d5 l2=10)."""
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    scale = edges["__scale__"]
    cond_r = (cond / df["source"].map(scale)).fillna(1.0)
    ratio = days / cond_r.clip(lower=1e-9)

    # 数值特征
    out["days"] = days
    out["log_days"] = np.log1p(days.clip(lower=0))
    out["condition"] = cond
    out["log_condition"] = np.log1p(cond.clip(lower=0))
    out["condition_missing"] = cond.isna().astype(int)
    out["cond_r"] = cond_r.astype(float)
    out["log_cond_r"] = np.log(cond_r.clip(lower=1e-9))
    out["ratio"] = ratio.astype(float)
    out["log_ratio"] = np.log(ratio.clip(lower=1e-9))
    out["ratio_p75"] = (days / cond_r.clip(lower=1e-9) ** 0.75).astype(float)
    out["cond_x_days"] = (cond * days).astype(float)
    out["cond_over_days"] = (cond / (days.abs() + 1.0)).astype(float)
    out["age_range"] = df["age_range"].astype(float)
    out["days_over_age"] = (days / df["age_range"].astype(float)).astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = out[BIN_COLS].sum(axis=1)
    for col in RATIO_COLS:
        vals = pd.to_numeric(df[col], errors='coerce')
        out[f"{col}_over_cr"] = (vals / cond_r.clip(lower=1e-9)).astype(np.float32)

    # 类别特征
    cats = []
    out["region"] = df["region"].astype(str)
    out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str)
    out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str)
    out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = out[BIN_COLS].astype(str).agg("".join, axis=1)
    out["days_fx"] = np.digitize(days.to_numpy(dtype=float), DAYS_FX).astype(str)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat", "days_fx"]

    # 分箱类别
    for n in QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"d_{n}"]).astype(str)
        out[f"r{n}"] = _qbins(ratio, edges[f"ra_{n}"]).astype(str)
        cats += [f"d{n}", f"r{n}"]
    for n in (5, 10, 20):
        out[f"c{n}"] = _qbins(cond.fillna(-1), edges[f"c_{n}"]).astype(str)
        out[f"cr{n}"] = _qbins(cond_r, edges[f"cr_{n}"]).astype(str)
        cats += [f"c{n}", f"cr{n}"]

    # 手工类别交叉 (核心: 捕捉条件×地域×年龄的交互效应)
    def cross(n, *p):
        s = out[p[0]].astype(str)
        for x in p[1:]:
            s = s + "|" + out[x].astype(str)
        out[n] = s
        cats.append(n)

    cross("rs", "region", "source"); cross("d10r", "d10", "region"); cross("d10s", "d10", "source")
    cross("d20r", "d20", "region"); cross("d20s", "d20", "source"); cross("d10a", "d10", "age_cat")
    cross("d10c10", "d10", "c10"); cross("c10r", "c10", "region"); cross("c10s", "c10", "source")
    cross("ra", "region", "age_cat"); cross("sa", "source", "age_cat")
    cross("d10p", "d10", "bin_pat"); cross("rp", "region", "bin_pat")
    cross("d5rs", "d5", "region", "source"); cross("r10r", "r10", "region"); cross("r10s", "r10", "source")
    cross("r10a", "r10", "age_cat"); cross("r20r", "r20", "region"); cross("r10p", "r10", "bin_pat")
    cross("cr10r", "cr10", "region"); cross("cr10a", "cr10", "age_cat")
    cross("c5s", "c5", "source"); cross("c20s", "c20", "source")
    cross("cr5s", "cr5", "source"); cross("cr10s", "cr10", "source"); cross("cr20s", "cr20", "source")
    cross("cr5r", "cr5", "region"); cross("cr20r", "cr20", "region"); cross("c5r", "c5", "region")
    cross("d5c5", "d5", "c5"); cross("d20c20", "d20", "c20")
    cross("d5cr5", "d5", "cr5"); cross("d10cr10", "d10", "cr10")
    cross("d10c10r", "d10", "c10", "region"); cross("d10c10s", "d10", "c10", "source")
    cross("d10c10a", "d10", "c10", "age_cat"); cross("sc10a", "source", "c10", "age_cat")
    cross("rc10a", "region", "c10", "age_cat"); cross("rsa", "region", "source", "age_cat")
    cross("dfs", "days_fx", "source"); cross("dfc10", "days_fx", "c10")
    cross("dfcr10", "days_fx", "cr10"); cross("dfr", "days_fx", "region"); cross("r5rs", "r5", "region", "source")

    # 频率编码
    for c in ("region", "source", "bin_pat", "rs", "d10r", "c10s", "month", "version"):
        out[f"f_{c}"] = out[c].map(out[c].value_counts()).astype(float)

    # 剩余类别
    out["x19c"] = df["x19"].astype(str); out["x20c"] = df["x20"].astype(str)
    out["lvc"] = df["livability"].astype(str); out["t3c"] = df["t3"].astype(str)
    out["cdc"] = df["code"].astype(str)
    cats += ["x19c", "x20c", "lvc", "t3c", "cdc"]
    out["cc"] = df["cc"].astype(float); out["max_g"] = df["max_g"].astype(float)
    out["V"] = df["V"].astype(float)
    cross("x20s", "x20c", "source"); cross("x20r", "x20c", "region"); cross("x20a", "x20c", "age_cat")
    cross("x19l", "x19c", "lvc"); cross("lva", "lvc", "age_cat"); cross("rl", "region", "lvc")
    cross("t3d5", "t3c", "d5"); cross("sx20a", "source", "x20c", "age_cat")
    cross("rx20a", "region", "x20c", "age_cat"); cross("rsx19", "region", "source", "x19c")

    return out, cats


# ======================= 臂 2: rate 世界 =======================

def fit_edges_alt(df):
    """在全集上拟合分箱边界."""
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    days = pd.to_numeric(df["days"])
    rate = days * (1.0 - rk)
    edges = {}
    for n in ALT_QUANTS:
        qs = np.linspace(0, 1, n + 1)[1:-1]
        edges[f"d_{n}"] = np.quantile(days.dropna(), qs)
        edges[f"k_{n}"] = np.quantile(rk, qs)
        edges[f"e_{n}"] = np.quantile(rate, qs)
    return edges


def build_alt(df, edges):
    """rate 世界: ~100 特征. 用于 arm2 (Plain d6 l2=6)."""
    out = pd.DataFrame(index=df.index)
    days = pd.to_numeric(df["days"])
    cond = pd.to_numeric(df["condition"])
    rk = df.groupby("source")["condition"].rank(pct=True).fillna(0.5)
    rate = days * (1.0 - rk)

    out["days"] = days; out["sqrt_days"] = np.sqrt(days.clip(lower=0))
    out["condition"] = cond; out["cond_rk"] = rk; out["rate"] = rate
    out["log_rate"] = np.log1p(rate.clip(lower=0))
    out["rate_over_age"] = rate / df["age_range"].astype(float)
    out["condition_missing"] = cond.isna().astype(int)
    out["age_range"] = df["age_range"].astype(float)
    out["grade_ord"] = df["grades"].map(GRADE_MAP).astype(float)
    for c in BIN_COLS:
        out[c] = df[c].astype(int)
    out["bin_sum"] = out[BIN_COLS].sum(axis=1)

    cats = []
    out["region"] = df["region"].astype(str); out["source"] = df["source"].astype(str)
    out["month"] = df["month"].astype(str); out["version"] = df["version"].astype(str)
    out["grades_c"] = df["grades"].astype(str); out["age_cat"] = df["age_range"].astype(str)
    out["bin_pat"] = out[BIN_COLS].astype(str).agg("".join, axis=1)
    cats += ["region", "source", "month", "version", "grades_c", "age_cat", "bin_pat"]

    for n in ALT_QUANTS:
        out[f"d{n}"] = _qbins(days, edges[f"d_{n}"]).astype(str)
        out[f"k{n}"] = _qbins(rk, edges[f"k_{n}"]).astype(str)
        out[f"e{n}"] = _qbins(rate, edges[f"e_{n}"]).astype(str)
        cats += [f"d{n}", f"k{n}", f"e{n}"]

    def cross(n, *p):
        s = out[p[0]].astype(str)
        for x in p[1:]:
            s = s + "|" + out[x].astype(str)
        out[n] = s; cats.append(n)

    cross("Ak7s", "k7", "source"); cross("Ak13s", "k13", "source"); cross("Ak25s", "k25", "source")
    cross("Ak13r", "k13", "region"); cross("Ak7a", "k7", "age_cat")
    cross("Ad13r", "d13", "region"); cross("Ad13s", "d13", "source"); cross("Ad7a", "d7", "age_cat")
    cross("Ad25r", "d25", "region"); cross("Ae13r", "e13", "region"); cross("Ae13s", "e13", "source")
    cross("Ae7a", "e7", "age_cat"); cross("Ae7p", "e7", "bin_pat")
    cross("Ad7k7", "d7", "k7"); cross("Ad13k13", "d13", "k13")
    cross("Ars", "region", "source"); cross("Ara", "region", "age_cat"); cross("Asa", "source", "age_cat")
    cross("Ad7rs", "d7", "region", "source"); cross("Ak7ra", "k7", "region", "age_cat")
    cross("Ae7rs", "e7", "region", "source"); cross("Ad7p", "d7", "bin_pat"); cross("Arp", "region", "bin_pat")
    for c in ("region", "source", "bin_pat", "Ars", "Ak13s", "Ad13r"):
        out[f"f_{c}"] = out[c].map(out[c].value_counts()).astype(float)
    out["x19c"] = df["x19"].astype(str); out["x20c"] = df["x20"].astype(str)
    out["lvc"] = df["livability"].astype(str); out["t3c"] = df["t3"].astype(str)
    out["cdc"] = df["code"].astype(str)
    cats += ["x19c", "x20c", "lvc", "t3c", "cdc"]
    cross("x20s", "x20c", "source"); cross("x20r", "x20c", "region"); cross("x20a", "x20c", "age_cat")
    cross("rl", "region", "lvc")

    return out, cats
