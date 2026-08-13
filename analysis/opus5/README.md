# 0.69993 方案 — 纯诚实 max2 基线

> 比赛:AUC 二分类 | train 14930×44, test 6398×43, 目标 `label`(10% 正例)
> 嵌套 OOF:**0.69993**(5-block 无偏估计)| 预期公开榜:~0.7095

## 方案是什么

两个 **诚实 CatBoost 臂**(固定树数、无早停、label-free 特征工程)做 **逐元素 max 融合**:

| 臂 | 编码世界 | 配置 | 种子 | 单臂 OOF |
|---|---|---|---|---:|
| `merger_ord8` | v2 主帧 FE + Ordered boosting | depth=5, iter=800, lr=0.03, l2=10 | 2026..2033 (8) | 0.69660 |
| `v2_cat_alt8` | v2 alt 编码世界 (rate=days×(1-rank_pct)) | depth=6, iter=800, lr=0.03, l2=6 | 2026..2033 (8) | 0.69704 |
| **max2 融合** | 逐元素 max(rank-oof) | — | — | **0.69993** |

**为什么是"诚实":**
- 固定 800 棵树,**不用 `use_best_model`**(无早停 → OOF 无乐观偏差)
- 特征工程 label-free(分位数切点、jitter 流都在 train+test 上一次性拟合)
- 5-fold 分层 CV,种子在 `StratifiedKFold` 上控制可复现

**为什么用 max 而非 mean:** 两个臂的相关性 0.96,但 max 对"哪些行被顶到高位"更敏感,在这对臂上 max 比 mean 高 +0.0013。

## 依赖

```bash
pip install catboost lightgbm scikit-learn pandas numpy scipy
```

- Python 3.10+
- 数据(共享路径,本仓库不含数据):
  - `/Volumes/pssd/app/ml/正式比赛/data/train.csv`
  - `/Volumes/pssd/app/ml/正式比赛/data/test.csv`

> 如数据在别处,改 `src/*.py` 顶部的 `DATA = Path(...)` 即可。

## 目录结构

```
.
├── src/
│   ├── v2fe_ord_chunk.py    # 臂1 chunk 脚本(4 chunk × 2 seed)
│   ├── v2_cat_alt_chunk.py  # 臂2 chunk 脚本(4 chunk × 2 seed)
│   ├── combine_chunks.py    # 合并 4 chunk → 8 种子臂
│   ├── fuse_v4b.py          # max2 融合 → submission_v4_honest.csv
│   └── src2/                # 内联依赖(纯模块,无外部路径)
│       ├── features.py      # FE: fit_edges, fit_edges_alt
│       ├── arms.py          # catboost_frame, altboost_frame, ARMS, CAT_BASE
│       ├── jitter.py        # add_jitter_views (确定性 jitter 流)
│       └── te.py            # fold-safe target encoding
├── artifacts/               # 预计算臂(已含,可秒级复现 fuse)
│   ├── merger_ord8.npz      # oof + test_pred + per_seed + seeds + y
│   └── v2_cat_alt8.npz
├── submissions/
│   └── submission_v4_honest.csv  # 最终提交(6398 行)
└── reproduce.sh             # 一键复现
```

## 复现

### 快速(用预计算臂,秒级)
直接跑融合,验证 0.69993:
```bash
python3 src/fuse_v4b.py
```
预期输出:
```
  [honest] merger_ord8   oof_auc=0.69660
  [honest] v2_cat_alt8   oof_auc=0.69704

max2(merger_ord8 + v2_cat_alt8)
  nested=0.69993  full=0.70023
```

### 完整(从数据重跑,~150 min)
```bash
bash reproduce.sh
```
等价于:
```bash
# 臂1: merger_ord8 (Ordered, depth5, 8 seeds)
for c in 0 1 2 3; do python3 src/v2fe_ord_chunk.py $c; done
python3 src/combine_chunks.py merger_ord8 v2fe_ord_c0 v2fe_ord_c1 v2fe_ord_c2 v2fe_ord_c3

# 臂2: v2_cat_alt8 (alt world, depth6, 8 seeds)
for c in 0 1 2 3; do python3 src/v2_cat_alt_chunk.py $c; done
python3 src/combine_chunks.py v2_cat_alt8 v2_cat_alt_c0 v2_cat_alt_c1 v2_cat_alt_c2 v2_cat_alt_c3

# 融合
python3 src/fuse_v4b.py
```

## 指标说明

- **full OOF AUC**(0.70023):全局排序后算 AUC,略乐观(+0.0003)
- **nested OOF AUC**(0.69993):每个 block 内重排再算,反乐观,是无偏估计 ← **以这个为准**

两个 seed 池的 rank 聚合:每个 chunk 内 2-seed rank-pool,跨 chunk 再 rank-pool,等价于 8-seed 平衡 rank 池。

## 已知边界

这是**诚实天花板**——所有尝试加第 3 个诚实臂都掉分(单臂太弱时 max 融合敏感)。要往上突破 0.702 需要更强的第 3 正交臂(非本方案范围)。
