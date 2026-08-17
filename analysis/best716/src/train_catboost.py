"""CatBoost 训练: 10fold × 多 seed × 3 bagging, RMSE loss.

两个臂配置 (vz17 用):
  arm1: build_main, Ordered, depth=5, l2=10, rsm=1.0  (cond_r 世界, 全特征)
  arm2: build_alt,  Plain,   depth=6, l2=6,  rsm=0.3  (rate 世界, 随机子空间正则)

关键设计决策 (全部有实验证据):
  - RMSE 而非 Logloss: 分类 loss 全部失败 (OOF ~0.51), RMSE 回归有效 (OOF ~0.70)
  - 10fold 而非 5fold: 10fold 比 5fold 提升 ~+0.002 线上
  - 3 bagging: 每 fold 用 3 个不同 random_seed 训练, 取平均
  - rsm=0.3 for arm2: 121 特征的信号稀释问题, 30% 特征子采样是最优正则
    (扫描: 0.3 > 0.5 > 0.25 > 0.4 > 1.0)

用法:
  python3 train_catboost.py --arm arm1 --data-dir /path/to/data --out-dir /path/to/out
  python3 train_catboost.py --arm arm2 --data-dir /path/to/data --out-dir /path/to/out
"""
from __future__ import annotations
import os, time, logging, argparse, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool
from features import fit_edges_main, build_main, fit_edges_alt, build_alt

# ===== 默认配置 (可被命令行覆盖) =====
ARM_CONFIGS = {
    "arm1": {
        "build_fn": "main", "ordered": True, "depth": 5, "l2": 10,
        "rsm": 1.0, "iterations": 800, "seeds": [2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047],
    },
    "arm2": {
        "build_fn": "alt", "ordered": False, "depth": 6, "l2": 6,
        "rsm": 0.3, "iterations": 800, "seeds": [2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047],
    },
}
N_SPLITS = 10
BAG_SEEDS = (0, 1, 2)
LR = 0.03


def train_arm(arm_name, config, train_df, test_df, y, out_dir, log, tc=-1):
    """训练一个臂的所有 seed, 输出 per-seed OOF 和 test 预测 (.npy)."""
    raw_all = pd.concat([train_df.drop(columns=["label"]), test_df], ignore_index=True)

    # 特征工程
    if config["build_fn"] == "main":
        edges = fit_edges_main(raw_all)
        X_all, cats = build_main(raw_all, edges)
    else:
        edges = fit_edges_alt(raw_all)
        X_all, cats = build_alt(raw_all, edges)
    for c in cats:
        X_all[c] = X_all[c].astype(str)
    Xtr = X_all.iloc[:len(train_df)].reset_index(drop=True)
    Xte = X_all.iloc[len(train_df):].reset_index(drop=True)

    for seed in config["seeds"]:
        ckpt_oof = os.path.join(out_dir, f"{arm_name}_seed{seed}_oof.npy")
        ckpt_te = os.path.join(out_dir, f"{arm_name}_seed{seed}_te.npy")
        if os.path.exists(ckpt_oof) and os.path.exists(ckpt_te):
            oof = np.load(ckpt_oof)
            log.info(f"  跳过已有 {arm_name} seed {seed}: OOF={roc_auc_score(y, oof):.5f}")
            continue

        t0 = time.time()
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        te_seed = np.zeros(len(test_df))

        for f, (tri, vali) in enumerate(skf.split(Xtr, y)):
            fold_te = np.zeros(len(test_df))
            for bs in BAG_SEEDS:
                kw = dict(
                    loss_function="RMSE", eval_metric="RMSE",
                    iterations=config["iterations"], learning_rate=LR,
                    depth=config["depth"], l2_leaf_reg=config["l2"],
                    random_strength=0.7, verbose=0, allow_writing_files=False,
                    one_hot_max_size=2, thread_count=tc,
                    random_seed=(seed * 100 + bs),
                    rsm=config["rsm"],
                )
                if config["ordered"]:
                    kw["boosting_type"] = "Ordered"
                m = CatBoostRegressor(**kw)
                m.fit(Pool(Xtr.iloc[tri], y[tri], cat_features=cats), verbose=False)
                oof[vali] += m.predict(Xtr.iloc[vali])
                fold_te += m.predict(Xte)
            oof[vali] /= len(BAG_SEEDS)
            te_seed += fold_te / len(BAG_SEEDS)

        te_seed /= N_SPLITS
        oof_rk = rankdata(oof) / len(oof)
        te_rk = rankdata(te_seed) / len(te_seed)
        a = roc_auc_score(y, oof_rk)
        log.info(f"  [{arm_name}] seed {seed}: OOF={a:.5f} ({time.time()-t0:.0f}s)")
        np.save(ckpt_oof, oof_rk)
        np.save(ckpt_te, te_rk)


def main():
    p = argparse.ArgumentParser(description="CatBoost 训练 (arm1/arm2)")
    p.add_argument("--arm", choices=["arm1", "arm2"], required=True)
    p.add_argument("--data-dir", required=True, help="包含 train.csv, test.csv 的目录")
    p.add_argument("--out-dir", required=True, help="输出 .npy checkpoint 的目录")
    p.add_argument("--threads", type=int, default=-1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    log = logging.getLogger(f"train_{args.arm}")

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"), dtype={"id": str})
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"), dtype={"id": str})
    y = train_df["label"].astype(int).values

    config = ARM_CONFIGS[args.arm]
    log.info(f"=== {args.arm}: build_{config['build_fn']}, "
             f"{'Ordered' if config['ordered'] else 'Plain'} "
             f"d{config['depth']} l2={config['l2']} rsm={config['rsm']} "
             f"{N_SPLITS}fold 3bag ===")
    train_arm(args.arm, config, train_df, test_df, y, args.out_dir, log, args.threads)


if __name__ == "__main__":
    main()
