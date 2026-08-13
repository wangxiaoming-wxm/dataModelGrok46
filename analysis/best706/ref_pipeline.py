# -*- coding: utf-8 -*-
"""
ref_pipeline.py
参考项目（insurance-oof-competition-main，Phase 24，线上 0.69425）核心架构的完整复刻。

本模块按参考项目源码逐函数复刻其特征工程与训练协议，作为我方新的"参考原生管线"：
  - prepare(mode='engineered')      ：基础解析特征（month_num/version_num/t3_num/t3_suffix/
                                       car_num/engine_num/condition_missing/log1p_*）
  - add_risk_surface()              ：风险曲面（condition_signed_log/days_condition_*/
                                       qcut 分位箱/二维网格/组内 pct,z 位置）
  - add_base_features()             ：code_age/region_code/livability_q5/daysq10_*/condq10_*
                                       交叉 + latent cluster 特征（零泄漏无监督）
  - select_foldwise()               ：每折训练集内独立特征选择（Top-85%，数值用 |AUC-0.5|*2，
                                       类别用 adjusted_mutual_info），验证标签不参与
  - add_id_features()               ：6 位 id 十六进制字符位（原生字符串类别喂法）
  - train_ref()                     ：参考超参（lr=0.05/depth=6/l2=5/scale_pos_weight=1.5/
                                       random_strength=1）+ 可选 Bayesian 0.25

纪律（与参考项目一致）：
  - 所有特征变换不依赖 label（无监督/确定性）；foldwise 选择只用外层训练折
  - OOF 在验证折上产生，test 为各折平均
  - 输出 OOF/test 的 fold-rank 版本（rankdata 在每个验证折内归一），供集成使用
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 基础解析
def prepare(df: pd.DataFrame, mode: str = "engineered") -> tuple[pd.DataFrame, list[str]]:
    """复刻 catboost_experiments.prepare()。返回 (x, cat_cols)。"""
    x = df.drop(columns=["label"], errors="ignore").copy()
    if mode in {"id_hex", "engineered"}:
        x["id_hex"] = x["id"].map(lambda v: int(str(v), 16))
        x = x.drop(columns=["id"])
    elif mode == "drop_id":
        x = x.drop(columns=["id"])
    elif mode != "raw_id":
        raise ValueError(mode)

    if mode == "engineered":
        x["month_num"] = pd.to_numeric(x["month"].str.extract(r"(\d+)")[0], errors="coerce")
        x["version_num"] = pd.to_numeric(x["version"].str.extract(r"(\d+)")[0], errors="coerce")
        x["t3_num"] = pd.to_numeric(x["t3"].str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
        x["t3_suffix"] = x["t3"].str.extract(r"([A-Za-z]+)$")[0]
        source_parts = x["source"].str.extract(r"CAR_(\d+)\|ENG_(\d+)")
        x["car_num"] = pd.to_numeric(source_parts[0], errors="coerce")
        x["engine_num"] = pd.to_numeric(source_parts[1], errors="coerce")
        x["condition_missing"] = x["condition"].isna().astype(int)
        for col in ["days", "cc", "max_g"]:
            x[f"log1p_{col}"] = np.log1p(np.clip(x[col], 0, None))

    cat_cols = list(x.select_dtypes(include=["object", "str"]).columns)
    for col in cat_cols:
        x[col] = x[col].fillna("__MISSING__").astype(str)
    return x, cat_cols


def add_risk_surface(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复刻 train_catboost_risk_surface.add_risk_surface()。"""
    n = len(train_x)
    full = pd.concat([train_x, test_x], ignore_index=True)
    abs_condition = full["condition"].abs()
    full["condition_signed_log"] = np.sign(full["condition"]) * np.log1p(abs_condition)
    full["days_condition_product"] = full["days"] * full["condition"]
    full["logdays_condition_product"] = np.log1p(full["days"].clip(lower=0)) * full["condition"]
    full["days_condition_ratio"] = full["days"] / (abs_condition + 0.1)
    full["condition_days_ratio"] = full["condition"] / (full["days"].abs() + 1.0)

    for bins in [5, 10, 20]:
        d = pd.qcut(full["days"], bins, labels=False, duplicates="drop")
        c = pd.qcut(full["condition"], bins, labels=False, duplicates="drop")
        full[f"days_q{bins}"] = d.fillna(-1).astype(int).astype(str)
        full[f"condition_q{bins}"] = c.fillna(-1).astype(int).astype(str)
        full[f"days_condition_grid{bins}"] = full[f"days_q{bins}"] + "_" + full[f"condition_q{bins}"]

    for group in ["region", "source", "code"]:
        for col in ["days", "condition"]:
            full[f"{col}_pct_{group}"] = full.groupby(group, observed=True)[col].rank(pct=True)
            means = full.groupby(group, observed=True)[col].transform("mean")
            stds = full.groupby(group, observed=True)[col].transform("std").replace(0, np.nan)
            full[f"{col}_z_{group}"] = (full[col] - means) / stds
        full[f"daysq10_{group}"] = full["days_q10"] + "_" + full[group].astype(str)
        full[f"condq10_{group}"] = full["condition_q10"] + "_" + full[group].astype(str)

    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True)


def add_base_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    with_latent: bool = False,
    latent_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复刻 train_residual_feature_candidates.add_base_features()。
    = prepare('engineered') 去 id_hex + risk_surface + code_age/region_code/livability_q5
      + daysq10/condq10 交叉 + latent cluster 特征（若 with_latent）。
    返回 (x, xt)，字符串类别保留。"""
    x, _ = prepare(train, "engineered")
    xt, _ = prepare(test, "engineered")
    x = x.drop(columns=["id_hex"])
    xt = xt.drop(columns=["id_hex"])
    x, xt = add_risk_surface(x, xt)

    for d in [x, xt]:
        d["code_age"] = d.code.astype(str) + "_" + d.age_range.astype(str)
        d["region_code"] = d.region.astype(str) + "_" + d.code.astype(str)
        d["livability_q5"] = (
            pd.cut(d.livability, [-np.inf, .1, .2, .3, .4, np.inf], labels=False)
            .fillna(-1).astype(int).astype(str)
        )
        for base in ["days_q10", "condition_q10"]:
            prefix = "daysq10" if base == "days_q10" else "condq10"
            for c in ["t3_suffix", "grades", "month", "age_range", "livability_q5", "version"]:
                d[f"{prefix}_{c}"] = d[base].astype(str) + "_" + d[c].astype(str)

    if with_latent:
        clusters = pd.read_csv(latent_path or (BASE / "latent_cluster_features.csv")).set_index("id")
        # 动态对齐：取新数据簇文件中实际存在的列（参考项目旧列名不可复用）
        selected = [c for c in clusters.columns if c.startswith("cluster_")][:8]
        for frame, ids in [(x, train.id), (xt, test.id)]:
            aligned = clusters.loc[ids, selected].reset_index(drop=True)
            for c in selected:
                frame[c] = aligned[c].astype(str)
                frame[f"daysq10_{c}"] = frame.days_q10.astype(str) + "_" + frame[c]

    # 统一填充字符串缺失
    for frame in (x, xt):
        for col in list(frame.select_dtypes(include=["object", "str"]).columns):
            frame[col] = frame[col].fillna("__MISSING__").astype(str)
    return x, xt


def add_agg_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x: pd.DataFrame,
    xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v3 分组统计族（我方此前验证有效，参考架构遗漏）。

    按多 key（region/version/car/eng/month/region_version/car_eng）分组，
    对 days/cc/max_g/V 统计 train+test 合并（无标签）的 mean/median/count。
    返回 (x+agg, xt+agg)。严格无标签：统计基于 train+test 拼接，不含 label。"""
    # 解析 source -> car/eng
    def parse_source(s):
        parts = s.str.split("|", expand=True)
        car = parts[0].str.replace("CAR_", "", regex=False)
        eng = parts[1].str.replace("ENG_", "", regex=False)
        return pd.to_numeric(car, errors="coerce"), pd.to_numeric(eng, errors="coerce")

    n_tr = len(train)
    keys = ["region", "version", "car", "eng", "month", "region_version", "car_eng"]
    nums = ["days", "cc", "max_g", "V"]

    def _dframe(tt, prefix):
        d = pd.DataFrame(index=tt.index)
        d["region"] = tt["region"].astype(str)
        d["version"] = tt["version"].astype(str)
        car, eng = parse_source(tt["source"])
        d["car"] = car.fillna(-1).astype(int).astype(str)
        d["eng"] = eng.fillna(-1).astype(int).astype(str)
        d["month"] = tt["month"].str.extract(r"(\d+)")[0].astype(str)
        d["region_version"] = d["region"] + "_" + d["version"]
        d["car_eng"] = d["car"] + "_" + d["eng"]
        for num in nums:
            d[num] = tt[num].to_numpy()
        return d

    tr_d = _dframe(train, "tr")
    te_d = _dframe(test, "te")
    merged = pd.concat([tr_d, te_d], axis=0, ignore_index=True)

    agg_tr = pd.DataFrame(index=train.index)
    agg_te = pd.DataFrame(index=test.index)
    for key in keys:
        cnt = merged.groupby(key)[key].transform("size")
        agg_tr[f"cnt_{key}"] = cnt[:n_tr].to_numpy()
        agg_te[f"cnt_{key}"] = cnt[n_tr:].to_numpy()
        for num in nums:
            mean_v = merged.groupby(key)[num].transform("mean")
            med_v = merged.groupby(key)[num].transform("median")
            agg_tr[f"{key}_{num}_mean"] = mean_v[:n_tr].to_numpy()
            agg_te[f"{key}_{num}_mean"] = mean_v[n_tr:].to_numpy()
            agg_tr[f"{key}_{num}_med"] = med_v[:n_tr].to_numpy()
            agg_te[f"{key}_{num}_med"] = med_v[n_tr:].to_numpy()

    return pd.concat([x.reset_index(drop=True), agg_tr], axis=1), pd.concat(
        [xt.reset_index(drop=True), agg_te], axis=1)


# ---------------------------------------------------------------- 无标签特征库（全部 train+test 联合、不含 label）
def _parse_source(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.str.split("|", expand=True)
    car = parts[0].str.replace("CAR_", "", regex=False)
    eng = parts[1].str.replace("ENG_", "", regex=False)
    return pd.to_numeric(car, errors="coerce"), pd.to_numeric(eng, errors="coerce")


def _qcut_str(series: pd.Series, q: int) -> pd.Series:
    return (
        pd.qcut(pd.to_numeric(series, errors="coerce"), q=q, labels=False, duplicates="drop")
        .fillna(-1).astype(int).astype(str)
    )


def add_candidate_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x: pd.DataFrame,
    xt: pd.DataFrame,
    *,
    risk_triples: bool = True,
    frequency_compact: bool = True,
    anonymous_days: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """复刻参考 train_residual_feature_candidates.add_candidate_features()。
    frequency_compact/risk_triples/anonymous_days 三族，全部 train+test 联合无标签。
    返回 (x+features, xt+features, added)。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    raw_full = pd.concat([train.drop(columns=["label"], errors="ignore"), test], ignore_index=True)
    added: list[str] = []

    if frequency_compact:
        for col in ["t3", "version", "source", "age_range"]:
            raw = raw_full[col].fillna("__MISSING__").astype(str)
            counts = raw.value_counts(dropna=False)
            freq = raw.map(counts).astype(float)
            full[f"freq_count_{col}"] = freq.to_numpy()
            full[f"freq_log_{col}"] = np.log1p(freq.to_numpy())
            full[f"freq_bin_{col}"] = pd.cut(
                freq, bins=[-np.inf, 5, 20, 50, 100, 250, 500, 1000, np.inf],
                labels=["le5", "6_20", "21_50", "51_100", "101_250", "251_500", "501_1000", "gt1000"],
            ).astype(str).to_numpy()
            added.extend([f"freq_count_{col}", f"freq_log_{col}", f"freq_bin_{col}"])

    if risk_triples:
        days_q3 = _qcut_str(raw_full["days"], 3)
        condition_q3 = _qcut_str(raw_full["condition"], 3)
        days_q5 = _qcut_str(raw_full["days"], 5)
        src = raw_full["source"].fillna("__MISSING__").astype(str)
        reg = raw_full["region"].fillna("__MISSING__").astype(str)
        full["risk3_source"] = days_q3 + "_" + condition_q3 + "_" + src
        full["risk3_region"] = days_q3 + "_" + condition_q3 + "_" + reg
        full["daysq5_source_targeted"] = days_q5 + "_" + src
        full["daysq5_region_targeted"] = days_q5 + "_" + reg
        added.extend(["risk3_source", "risk3_region", "daysq5_source_targeted", "daysq5_region_targeted"])

    if anonymous_days:
        days_q5 = _qcut_str(raw_full["days"], 5)
        for axis in ["x5", "x13", "x19"]:
            axis_q5 = _qcut_str(raw_full[axis], 5)
            full[f"daysq5_{axis}q5_targeted"] = days_q5 + "_" + axis_q5
            added.append(f"daysq5_{axis}q5_targeted")

    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_deep_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x: pd.DataFrame,
    xt: pd.DataFrame,
    *,
    rowstats: bool = True,
    pca_k: int = 5,
    kmeans_k: tuple = (5, 10, 20),
    pair_top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """x0-x20 匿名轴深度挖掘：行统计 / PCA / KMeans 簇 / 低相关两两交互。零泄漏（train+test 联合）。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    added: list[str] = []
    xs = [f"x{i}" for i in range(21)]
    vals = full[xs].to_numpy(dtype=float)

    if rowstats:
        stats = {
            "x_mean": vals.mean(axis=1), "x_std": vals.std(axis=1),
            "x_min": vals.min(axis=1), "x_max": vals.max(axis=1),
            "x_abs_mean": np.abs(vals).mean(axis=1), "x_l2": np.sqrt((vals ** 2).sum(axis=1)),
            "x_pos_count": (vals > 0).sum(axis=1),
        }
        for k, v in stats.items():
            full[k] = v; added.append(k)

    if pca_k and pca_k > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_k, random_state=42)
        comps = pca.fit_transform(vals)
        for k in range(pca_k):
            full[f"x_pca{k}"] = comps[:, k]; added.append(f"x_pca{k}")

    if kmeans_k:
        from sklearn.cluster import KMeans
        for k in kmeans_k:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            full[f"x_kmeans{k}"] = km.fit_predict(vals).astype(str)
            added.append(f"x_kmeans{k}")

    if pair_top_n and pair_top_n > 0:
        # 低相关对（|corr| 最小）乘积，注入非线性交互
        corr = np.corrcoef(vals.T)
        iu = np.triu_indices(len(xs), 1)
        pairs = sorted(zip(iu[0], iu[1]), key=lambda p: abs(corr[p]))
        for i, j in pairs[:pair_top_n]:
            name = f"x{i}x{j}_prod"
            full[name] = full[xs[i]] * full[xs[j]]
            added.append(name)

    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_source_features(
    train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """source→car/eng 深化交叉 + 频率。零泄漏。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    raw_full = pd.concat([train.drop(columns=["label"], errors="ignore"), test], ignore_index=True)
    added: list[str] = []
    car, eng = _parse_source(raw_full["source"])
    car_s = car.fillna(-1).astype(int).astype(str)
    eng_s = eng.fillna(-1).astype(int).astype(str)
    reg = raw_full["region"].fillna("__MISSING__").astype(str)
    code = raw_full["code"].fillna("__MISSING__").astype(str)
    age = raw_full["age_range"].fillna("__MISSING__").astype(str)
    liv_q5 = (
        pd.cut(pd.to_numeric(raw_full["livability"], errors="coerce"),
               [-np.inf, .1, .2, .3, .4, np.inf], labels=False)
        .fillna(-1).astype(int).astype(str)
    )

    crosses = {
        "car_eng": car_s + "_" + eng_s,
        "car_region": car_s + "_" + reg,
        "eng_region": eng_s + "_" + reg,
        "car_code": car_s + "_" + code,
        "eng_code": eng_s + "_" + code,
        "car_age_range": car_s + "_" + age,
        "eng_age_range": eng_s + "_" + age,
        "source_livability_q5": raw_full["source"].fillna("__MISSING__").astype(str) + "_" + liv_q5,
    }
    for name, v in crosses.items():
        full[name] = v.to_numpy(); added.append(name)
    for col in ["car", "eng"]:
        s = car_s if col == "car" else eng_s
        cnt = s.map(s.value_counts()).astype(float)
        full[f"freq_{col}"] = cnt.to_numpy(); added.append(f"freq_{col}")
        full[f"freq_log_{col}"] = np.log1p(cnt.to_numpy()); added.append(f"freq_log_{col}")

    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_ratio_features(
    train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """数值比率/多项式 + log1p。零泄漏。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    added: list[str] = []
    d, c, cc, V, mg, lv = (pd.to_numeric(full[c], errors="coerce")
                           for c in ["days", "condition", "cc", "V", "max_g", "livability"])
    ratios = {
        "days_per_cc": d / (cc.abs() + 1.0),
        "cc_per_days": cc / (d.abs() + 1.0),
        "condition_per_cc": c / (cc.abs() + 1.0),
        "V_per_days": V / (d.abs() + 1.0),
        "max_g_per_livability": mg / (lv.abs() + 1.0),
        "days_mul_max_g": d * mg,
        "condition_mul_max_g": c * mg,
        "days_mul_cc": d * cc,
        "condition_mul_cc": c * cc,
        "V_mul_days": V * d,
    }
    for name, v in ratios.items():
        full[name] = v.to_numpy(); added.append(name)
        full[f"log1p_{name}"] = np.log1p(np.clip(v, 0, None)).to_numpy(); added.append(f"log1p_{name}")

    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_missing_cross(
    train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """condition 缺失模式 × 类别。零泄漏。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    raw_full = pd.concat([train.drop(columns=["label"], errors="ignore"), test], ignore_index=True)
    added: list[str] = []
    miss = raw_full["condition"].isna().astype(int).astype(str)
    for col in ["region", "source", "month", "code"]:
        v = raw_full[col].fillna("__MISSING__").astype(str)
        full[f"condition_missing_{col}"] = (miss + "_" + v).to_numpy()
        added.append(f"condition_missing_{col}")
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_highorder_cross(
    train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """类别高阶交叉（三阶）。零泄漏。days_q10/condition_q10 由 add_risk_surface 生成。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    added: list[str] = []
    d10 = full["days_q10"].astype(str)
    c10 = full["condition_q10"].astype(str)
    for col in ["region", "source"]:
        v = full[col].astype(str)
        full[f"risk3_{col}"] = (d10 + "_" + c10 + "_" + v).to_numpy()
        added.append(f"risk3_{col}")
    pairs = [("month", "region", "code"), ("version", "region", "age_range"),
             ("grades", "code", "region")]
    for a, b, c in pairs:
        va, vb, vc = (full[col].astype(str) for col in (a, b, c))
        name = f"{a}_{b}_{c}"
        full[name] = (va + "_" + vb + "_" + vc).to_numpy()
        added.append(name)
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_frequency_ext(
    train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """全类别频率编码扩展（count/log）。零泄漏。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    raw_full = pd.concat([train.drop(columns=["label"], errors="ignore"), test], ignore_index=True)
    added: list[str] = []
    car, eng = _parse_source(raw_full["source"])
    base_cols = ["region", "code", "grades", "version", "month", "age_range", "t3"]
    cols = base_cols + ["car", "eng", "source", "region_code", "car_age"]
    for col in base_cols:
        full[f"freq_{col}"] = raw_full[col].fillna("__MISSING__").astype(str) \
            .map(raw_full[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)) \
            .astype(float).to_numpy()
        added.append(f"freq_{col}")
    # car/eng 频率
    for col, s in [("car", car.fillna(-1).astype(int).astype(str)),
                   ("eng", eng.fillna(-1).astype(int).astype(str))]:
        full[f"freq_{col}"] = s.map(s.value_counts(normalize=True)).astype(float).to_numpy()
        added.append(f"freq_{col}")
    # 高阶交叉频率
    for col in ["region_code", "car_age"]:
        pass  # region_code/car_age 在 base 或 source 族中；此处补充 log
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


def add_axis_interact(
    train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """参考 numeric_axis_interaction_screen.csv 中 lift>0 的数值×days/condition_q10 交互。
    按 pair_auc 前 10（全部与 days_q10/condition_q10）。零泄漏。"""
    n = len(x)
    x = x.copy(); xt = xt.copy()
    full = pd.concat([x, xt], ignore_index=True)
    added: list[str] = []
    for axis, q, base in [("x9", 5, "days_q10"), ("x19", 5, "days_q10"), ("max_g", 10, "days_q10"),
                          ("cc", 10, "days_q10"), ("livability", 5, "days_q10"),
                          ("cc", 10, "condition_q10"), ("livability", 20, "condition_q10"),
                          ("x19", 20, "condition_q10"), ("V", 5, "condition_q10"),
                          ("max_g", 10, "condition_q10")]:
        aq = _qcut_str(full[axis], q)
        name = f"{axis}q{q}_{base}"
        full[name] = (aq + "_" + full[base].astype(str)).to_numpy()
        added.append(name)
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True), added


# ---------------------------------------------------------------- foldwise 特征选择
def numeric_strength(values: pd.Series, y: np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    numeric = numeric.fillna(fill).to_numpy()
    if np.unique(numeric).size < 2:
        return 0.0
    auc = roc_auc_score(y, rankdata(numeric, method="average"))
    return float(abs(auc - .5) * 2)


def categorical_strength(values: pd.Series, y: np.ndarray) -> float:
    categorical = values.fillna("__MISSING__").astype(str)
    return float(max(0.0, adjusted_mutual_info_score(y, categorical)))


def select_foldwise(x_fit: pd.DataFrame, y_fit: np.ndarray, fraction: float = .85) -> list[str]:
    """复刻 train_feature_selection_batch3.select_foldwise()。只在训练折内计算，验证标签不参与。"""
    cat_cols = list(x_fit.select_dtypes(include=["object", "str", "category"]).columns)
    num_cols = [col for col in x_fit.columns if col not in cat_cols]
    strengths: dict[str, float] = {}
    for col in num_cols:
        strengths[col] = numeric_strength(x_fit[col], y_fit)
    for col in cat_cols:
        strengths[col] = categorical_strength(x_fit[col], y_fit)
    keep_num = max(1, int(np.ceil(len(num_cols) * fraction)))
    keep_cat = max(1, int(np.ceil(len(cat_cols) * fraction)))
    selected_num = sorted(num_cols, key=lambda c: (-strengths[c], c))[:keep_num]
    selected_cat = sorted(cat_cols, key=lambda c: (-strengths[c], c))[:keep_cat]
    selected_set = set(selected_num + selected_cat)
    return [col for col in x_fit.columns if col in selected_set]


# ---------------------------------------------------------------- id 特征
def add_id_features(x: pd.DataFrame, xt: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series):
    """复刻 add_id()：6 位 id 十六进制字符位，字符串原生类别。返回 (x, xt, added)。"""
    x = x.copy()
    xt = xt.copy()
    added = []
    for pos in range(6):
        name = f"id_hex_char{pos}"
        x[name] = train_ids.astype(str).str[pos]
        xt[name] = test_ids.astype(str).str[pos]
        added.append(name)
    return x, xt, added


def null_id_features(train_ids: pd.Series, test_ids: pd.Series, salt: str = "20260804"):
    """SHA256(salt|id) 前 6 位十六进制，与真实 id 同基数同容量（capacity-matched null 对照）。"""
    import hashlib

    def _hash(series):
        return series.astype(str).map(
            lambda v: hashlib.sha256(f"{salt}|{v}".encode()).hexdigest()[:6])

    return _hash(train_ids), _hash(test_ids)


# ---------------------------------------------------------------- 训练协议
def fold_rank(values: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """在每个验证折内做 rank 归一（rankdata average / n）。"""
    ranked = np.zeros(len(values))
    for idx in folds:
        ranked[idx] = rankdata(values[idx], method="average") / len(values[idx])
    return ranked


REF_PARAMS = dict(
    iterations=1100, learning_rate=.05, depth=6, l2_leaf_reg=5,
    random_strength=1, scale_pos_weight=1.5,
    loss_function="Logloss", eval_metric="AUC",
    task_type="GPU", devices="0", allow_writing_files=False, verbose=False,
)


def train_ref(
    x: pd.DataFrame,
    xt: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Series,
    test_ids: pd.Series,
    *,
    name: str,
    n_folds: int = 3,
    seed: int = 31415,
    use_top85: bool = True,
    use_id: bool = False,
    bayesian025: bool = False,
    iterations: int | None = None,
    out_dir: Path | None = None,
    early_stop: int = 170,
    with_agg: bool = False,
    params_extra: dict | None = None,
) -> dict:
    """参考协议训练。返回 {oof, oof_rank, test, test_rank, report, folds}。
    可选：foldwise Top-85% 选择、id nibbles、Bayesian bagging_temperature=0.25、
    v3 分组统计 agg 特征（我方此前验证有效的增量特征族）。"""
    out_dir = out_dir or BASE
    if with_agg:
        train = pd.read_csv(BASE / "train.csv")
        test = pd.read_csv(BASE / "test.csv")
        x, xt = add_agg_features(train, test, x, xt)
        print(f"[{name}] +agg 特征，现共 {x.shape[1]} 列", flush=True)
    started = time.time()
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y))
    folds = [valid for _, valid in splits]
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(xt))
    fold_aucs: list[float] = []
    best_iters: list[int] = []
    selections: list[dict] = []

    params = dict(REF_PARAMS)
    if iterations is not None:
        params["iterations"] = iterations
    if bayesian025:
        params.update(bootstrap_type="Bayesian", bagging_temperature=.25)
    if params_extra:
        params.update(params_extra)

    for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        if use_top85:
            # 纪律修复(2026-08-07): 不再无条件强制加入 id_hex 列。
            # id 已证随机/无信号, 强制加 id 会绕过 foldwise 信号选择造成 OOF 虚高。
            # 特征一律按 foldwise merit 自然选择; id 派生列 (id_hex_char*) 直接排除。
            selected = select_foldwise(x.iloc[fit_idx], y[fit_idx], .85)
            id_cols = [f"id_hex_char{p}" for p in range(6)]
            selected = [c for c in selected if c not in id_cols]
        else:
            selected = list(x.columns)
        fold_x = x[selected].copy()
        fold_xt = xt[selected].copy()
        cats = list(fold_x.select_dtypes(include=["object", "str", "category"]).columns)
        for col in cats:
            fold_x[col] = fold_x[col].fillna("__MISSING__").astype(str)
            fold_xt[col] = fold_xt[col].fillna("__MISSING__").astype(str)

        model = CatBoostClassifier(random_seed=seed + fold, **params)
        model.fit(
            fold_x.iloc[fit_idx], y[fit_idx], cat_features=cats,
            eval_set=(fold_x.iloc[valid_idx], y[valid_idx]),
            early_stopping_rounds=early_stop, verbose=False,
        )
        oof[valid_idx] = model.predict_proba(fold_x.iloc[valid_idx])[:, 1]
        test_pred += model.predict_proba(fold_xt)[:, 1] / n_folds
        fold_aucs.append(float(roc_auc_score(y[valid_idx], oof[valid_idx])))
        best_iters.append(int(model.get_best_iteration() + 1))
        selections.append({
            "fold": fold,
            "selected_count": len(selected),
            "selected_features": selected,
        })
        print(f"[{name}] fold {fold}: AUC={fold_aucs[-1]:.4f} iters={best_iters[-1]} "
              f"n_feat={len(selected)}", flush=True)

    oof_rank = fold_rank(oof, folds)
    test_rank = fold_rank(test_pred, [np.arange(len(xt))])  # test 无折，整体 rank
    report = {
        "name": name,
        "validation": f"StratifiedKFold({n_folds}, shuffle=True, random_state={seed})",
        "use_top85": use_top85,
        "use_id": use_id,
        "bayesian025": bayesian025,
        "raw_auc": float(roc_auc_score(y, oof)),
        "foldrank_auc": float(roc_auc_score(y, oof_rank)),
        "fold_auc": [float(a) for a in fold_aucs],
        "best_iterations": best_iters,
        "selections": selections,
        "seconds": round(time.time() - started, 1),
    }
    return {
        "oof": oof,
        "oof_rank": oof_rank,
        "test": test_pred,
        "test_rank": test_rank,
        "report": report,
        "folds": folds,
    }


def save_result(result: dict, prefix: str, out_dir: Path | None = None):
    """持久化 OOF/test/report 与 fold-rank 版本。"""
    out_dir = out_dir or BASE
    np.save(out_dir / f"{prefix}_oof.npy", result["oof"])
    np.save(out_dir / f"{prefix}_oof_rank.npy", result["oof_rank"])
    np.save(out_dir / f"{prefix}_test.npy", result["test"])
    np.save(out_dir / f"{prefix}_test_rank.npy", result["test_rank"])
    report = result["report"]
    (out_dir / f"{prefix}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_result(prefix: str, out_dir: Path | None = None) -> dict | None:
    out_dir = out_dir or BASE
    p = out_dir / f"{prefix}_report.json"
    if not p.exists():
        return None
    return {
        "oof": np.load(out_dir / f"{prefix}_oof.npy"),
        "oof_rank": np.load(out_dir / f"{prefix}_oof_rank.npy"),
        "test": np.load(out_dir / f"{prefix}_test.npy"),
        "test_rank": np.load(out_dir / f"{prefix}_test_rank.npy"),
        "report": json.loads(p.read_text(encoding="utf-8")),
    }


if __name__ == "__main__":
    # 冒烟测试：3 折 screen 复刻参考 baseline（无 id、无 Top85、无 Bayesian）
    tr = pd.read_csv(BASE / "train.csv")
    te = pd.read_csv(BASE / "test.csv")
    y = tr.label.to_numpy()
    x, xt = add_base_features(tr, te)
    print(f"base features: {x.shape[1]} cols, {x.shape[0]} rows")
    print("cat cols:", len(x.select_dtypes(include=['object']).columns))
    res = train_ref(x, xt, y, tr.id, te.id, name="smoke_ref_base", n_folds=3, seed=31415)
    print(json.dumps({k: v for k, v in res["report"].items() if k != "selections"},
                     ensure_ascii=False, indent=2))
