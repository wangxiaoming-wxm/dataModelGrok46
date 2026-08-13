# -*- coding: utf-8 -*-
"""参考老数据模型 + 本地模型均值融合（均无 id 特征）。

思路：参考数据（train_old 21328 全精度 + test 10000）训练 CatBoost（与本地同协议），
预测参考 test，经特征级映射迁移到本地 test（cover 1.0）。该模型与本地数据模型
天然低相关（不同数据域训练）。最后与本地 cat_opt5 集成预测做均值，评估增益。

协议：
- 特征：ref_pipeline 完整协议（base+candidate+source+ratio+missing+freq+deep+latent），
  参考数据用参考 latent，本地数据用本地 latent；均无 id
- CatBoost：iterations=1100, lr=0.05, depth=6, l2=5, spw=1.5, Bayesian 0.25, GPU
- StratifiedKFold(5, shuffle)，foldwise Top-85%（仅训练折）
- 映射：HARD_KEYS 硬键 + 数值容差 1e-2（与 homology L2 一致），贪心 1:1 唯一化
- 验证：参考 OOF 映射到本地 train 段 → AUC（诚实验证）；本地 test 覆盖数
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import ref_pipeline as rp
from bootstrap_paired import paired_bootstrap

BASE = Path(__file__).resolve().parent
REF = Path(r"C:\Users\Sirocco\Desktop\insurance-oof-competition-main")
N_FOLDS = 5
SEED = 2026
CUR_OOF = 0.69086

HARD_KEYS = ["month", "t1", "t2", "r1", "r2", "code", "age_range", "w1", "w2", "version"]
SOFT_NUM = (["days", "cc", "condition", "V", "c1", "c2", "max_g"] + [f"x{i}" for i in range(21)])
TOL = 1e-2
REL_EPS = 1e-12

def rank_norm(v): return rankdata(v, method="average") / len(v)

def fold_rank(values, folds):
    ranked = np.zeros(len(values))
    for idx in folds:
        ranked[idx] = rankdata(values[idx], method="average") / len(values[idx])
    return ranked

def cat_model(seed):
    return CatBoostClassifier(
        iterations=1100, learning_rate=0.05, depth=6, l2_leaf_reg=5,
        random_strength=1, scale_pos_weight=1.5, bootstrap_type="Bayesian",
        bagging_temperature=0.25, loss_function="Logloss", eval_metric="AUC",
        random_seed=seed, task_type="GPU", devices="0",
        allow_writing_files=False, verbose=False,
    )

def build_features(train, test, latent_path):
    x, xt = rp.add_base_features(train, test, with_latent=True, latent_path=latent_path)
    x, xt, _ = rp.add_candidate_features(train, test, x, xt)
    x, xt, _ = rp.add_source_features(train, test, x, xt)
    x, xt, _ = rp.add_ratio_features(train, test, x, xt)
    x, xt, _ = rp.add_missing_cross(train, test, x, xt)
    x, xt, _ = rp.add_frequency_ext(train, test, x, xt)
    x, xt, _ = rp.add_deep_features(train, test, x, xt)
    return x, xt

def train_cat(x, xt, y, cat_cols, seed=SEED):
    """5 折训练，返回 (oof_rank, test_rank, folds)。"""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    splits = list(skf.split(x, y))
    folds = [v for _, v in splits]
    oof_raw = np.zeros(len(y)); test_sum = np.zeros(len(xt))
    fold_auc = []
    for fi, (fit_idx, val_idx) in enumerate(splits, 1):
        sel = rp.select_foldwise(x.iloc[fit_idx], y[fit_idx], fraction=0.85)
        cats = [c for c in sel if c in cat_cols]
        m = cat_model(seed + fi)
        m.fit(x[sel].iloc[fit_idx], y[fit_idx], cat_features=cats,
              eval_set=(x[sel].iloc[val_idx], y[val_idx]),
              early_stopping_rounds=200, verbose=False)
        oof_raw[val_idx] = m.predict_proba(x[sel].iloc[val_idx])[:, 1]
        test_sum += m.predict_proba(xt[sel])[:, 1]
        fold_auc.append(float(roc_auc_score(y[val_idx], oof_raw[val_idx])))
        print(f"  [fold {fi}] AUC={fold_auc[-1]:.5f} iters={m.get_best_iteration()+1}", flush=True)
    oof = fold_rank(oof_raw, folds)
    test = rank_norm(test_sum / N_FOLDS)
    print(f"  [5折] rank AUC={roc_auc_score(y, oof):.5f}", flush=True)
    return oof, test, folds

def map_rows(local, ref):
    """特征级 1:1 映射（硬键 + 数值容差 1e-2，贪心唯一化）。"""
    lk = local[HARD_KEYS].copy(); lk["_li"] = np.arange(len(local))
    rk = ref[HARD_KEYS].copy(); rk["_ri"] = np.arange(len(ref))
    cand = lk.merge(rk, on=HARD_KEYS, suffixes=("_l", "_r"))
    lN = local[SOFT_NUM].to_numpy(float); rN = ref[SOFT_NUM].to_numpy(float)
    li = cand["_li"].to_numpy(); ri = cand["_ri"].to_numpy()
    ok = np.ones(len(cand), dtype=bool)
    for j, col in enumerate(SOFT_NUM):
        a = lN[li, j]; b = rN[ri, j]; valid = ~(np.isnan(a) | np.isnan(b))
        ok &= np.where(valid, (np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), REL_EPS)) <= TOL, True)
    pairs = cand[ok].copy()
    if len(pairs) == 0:
        return pairs, {"matched": 0, "cover": 0.0}
    # 距离（在 pairs 子集上）
    li = pairs["_li"].to_numpy(); ri = pairs["_ri"].to_numpy()
    dists = np.zeros(len(pairs))
    for j in range(len(SOFT_NUM)):
        a = lN[li, j]; b = rN[ri, j]; valid = ~(np.isnan(a) | np.isnan(b))
        dists += np.where(valid, np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), REL_EPS), 0.0)
    pairs["_dist"] = dists
    pairs = pairs.sort_values("_dist")
    used_l, used_r, keep = set(), set(), []
    for li2, ri2 in zip(pairs["_li"].to_numpy(), pairs["_ri"].to_numpy()):
        if li2 in used_l or ri2 in used_r: continue
        used_l.add(li2); used_r.add(ri2); keep.append(li2)
    unique = pairs[pairs["_li"].isin(keep)]
    return unique, {"matched": int(len(unique)), "cover": round(len(unique) / len(local), 5)}

def main():
    t0 = time.time()
    # ---- 参考数据特征 ----
    ref_tr = pd.read_csv(REF / "train_old.csv")
    ref_te = pd.read_csv(REF / "test.csv")
    ref_y = ref_tr.label.to_numpy()
    fe_t0 = time.time()
    print("[fe] 参考数据特征构建...", flush=True)
    ref_x, ref_xt = build_features(ref_tr, ref_te, REF / "latent_cluster_features.csv")
    ref_cat = list(ref_x.select_dtypes(include=["object", "str", "category"]).columns)
    ref_num = [c for c in ref_x.columns if c not in ref_cat]
    for d in (ref_x, ref_xt):
        d[ref_num] = d[ref_num].fillna(-999)
    print(f"[fe] 参考特征 {ref_x.shape}（类别 {len(ref_cat)}），{time.time()-fe_t0:.0f}s", flush=True)

    # ---- 参考模型训练（断点恢复：OOF 已存在则跳过）----
    if (BASE / "ref_model_oof.npy").exists():
        print("[train] 发现已存 ref_model_oof.npy，跳过训练直接复用", flush=True)
        ref_oof = rank_norm(np.load(BASE / "ref_model_oof.npy"))
        ref_test = np.load(BASE / "ref_model_test.npy")
        ref_raw_auc = float(roc_auc_score(ref_y, ref_oof))
    else:
        print("[train] 参考模型 5 折训练...", flush=True)
        ref_oof, ref_test, _ = train_cat(ref_x, ref_xt, ref_y, ref_cat, seed=SEED)
        np.save(BASE / "ref_model_oof.npy", ref_oof)
        np.save(BASE / "ref_model_test.npy", ref_test)
        ref_raw_auc = float(roc_auc_score(ref_y, ref_oof))
    print(f"[train] 参考模型 OOF={ref_raw_auc:.5f}", flush=True)

    # ---- 映射到本地 ----
    loc_tr = pd.read_csv(BASE / "train.csv")
    loc_te = pd.read_csv(BASE / "test.csv")
    loc_y = loc_tr.label.to_numpy()

    # (a) 参考 OOF → 本地 train 段（诚实验证）
    m_tr = map_rows(loc_tr, ref_tr)
    lt_oof = np.full(len(loc_tr), np.nan)
    if len(m_tr[0]):
        lp = m_tr[0]["_li"].to_numpy(); rp_i = m_tr[0]["_ri"].to_numpy()
        lt_oof[lp] = ref_oof[rp_i]
    m = np.isfinite(lt_oof)
    tr_auc = float(roc_auc_score(loc_y[m], lt_oof[m])) if m.sum() > 100 else None
    print(f"[map] 本地 train 覆盖 {m.sum()}/{len(loc_tr)} ({m_tr[1]['cover']}), 迁移 AUC={tr_auc:.5f}", flush=True)

    # (b) 参考 test 预测 → 本地 test
    m_te = map_rows(loc_te, ref_te)
    lt_pred = np.full(len(loc_te), np.nan)
    if len(m_te[0]):
        lp = m_te[0]["_li"].to_numpy(); rp_i = m_te[0]["_ri"].to_numpy()
        lt_pred[lp] = ref_test[rp_i]
    covered = int(np.isfinite(lt_pred).sum())
    print(f"[map] 本地 test 覆盖 {covered}/{len(loc_te)} ({m_te[1]['cover']})", flush=True)
    fallback = float(np.nanmean(ref_test))
    ref_loc_test = np.where(np.isfinite(lt_pred), lt_pred, fallback)
    ref_loc_test = rank_norm(ref_loc_test)
    np.save(BASE / "ref_model_local_test.npy", ref_loc_test)

    # ---- 与本地 cat_opt5 均值融合 ----
    local_test = rank_norm(np.load(BASE / "comp_cat_opt5_test.npy"))
    blend = rank_norm((local_test + ref_loc_test) / 2.0)

    # 验证：本地 train 段（ref 迁移 OOF vs 本地 cat_opt5 OOF）
    local_oof = rank_norm(np.load(BASE / "comp_cat_opt5_oof.npy"))
    # 用同 m 子集比较（lt_oof 已按本地行对齐）
    if m.sum() > 100:
        a_local = local_oof[m]
        a_ref = lt_oof[m]
        a_blend = rank_norm((a_local + a_ref) / 2.0)
        auc_local = float(roc_auc_score(loc_y[m], a_local))
        auc_blend = float(roc_auc_score(loc_y[m], a_blend))
        boot = paired_bootstrap(loc_y[m], a_local, a_blend, n_boot=3000, seed=42)
        print(f"\n[验证] 本地 train 匹配段: local={auc_local:.5f} ref迁移={tr_auc:.5f} "
              f"blend={auc_blend:.5f} Δ={auc_blend-auc_local:+.5f}", flush=True)
        print(f"  bootstrap: p_pos={boot['p_pos']:.1%} ci={[round(x,5) for x in boot['ci95']]}", flush=True)
    else:
        auc_local = auc_blend = None; boot = None

    # ---- 生成 submission ----
    sub = pd.DataFrame({"id": loc_te["id"], "label": np.clip(blend, 1e-6, 1 - 1e-6)})
    sub.to_csv(BASE / "submission_blend_ref_local.csv", index=False)

    report = {
        "protocol": "参考老数据(training 21328 全精度) CatBoost + 本地 cat_opt5 集成均值，均无 id",
        "ref_model": {"oof_auc": ref_raw_auc, "n_folds": N_FOLDS, "seed": SEED,
                       "n_features": int(ref_x.shape[1])},
        "mapping": {
            "local_train_to_ref": m_tr[1],
            "local_test_to_ref": m_te[1],
            "local_test_covered": covered,
        },
        "validation": {
            "matched_local_train": int(m.sum()),
            "local_auc": auc_local,
            "ref_transfer_auc": tr_auc,
            "blend_auc": auc_blend,
            "blend_delta_vs_local": (auc_blend - auc_local) if auc_local else None,
            "bootstrap": boot,
            "vs_current_submission": (auc_blend - CUR_OOF) if auc_blend else None,
        },
        "submission": {"file": "submission_blend_ref_local.csv", "rows": len(sub)},
        "seconds": round(time.time() - t0, 1),
    }
    (BASE / "ref_local_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: ref_local_blend_report.json ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
