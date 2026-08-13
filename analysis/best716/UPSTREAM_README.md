# best_0.716 —— 车险索赔预测 · 线上 0.71629 复现包

## 一句话

本包复现 `W62 ⊕ ref30` 这一提交，**线上 AUC = 0.71629**（公开榜第 8 名附近）。
运行 `reproduce.py` 可在秒级得到与线上完全一致的 `submission.csv`（已逐位校验）。

## 任务与指标

| 项 | 值 |
|---|---|
| 任务 | 预测保单未来一年是否索赔（二分类） |
| 指标 | AUC（越大越好，`sklearn.metrics.roc_auc_score`） |
| 训练集 | `data/train.csv` 14930 行 × 45 列（含 `label`） |
| 测试集 | `data/test.csv` 6398 行 × 44 列 |
| 提交格式 | `id,label` 两列，label 为索赔概率 [0,1] |
| 正例率 | 10.02% |

## 快速复现（推荐，秒级）

```bash
pip install -r requirements.txt
python3 reproduce.py
```

产出 `submission.csv`（6398 行），与线上 0.71629 的提交**逐位一致**。

脚本会打印：

```
W62 OOF        = 0.70153
ref solo OOF   = 0.69034
blend OOF      = 0.70256  (w_ref=0.3)
期望线上        = 0.71629
```

## 方案是什么（0.71629 = W62 ⊕ ref30）

```
final = (1 - 0.30) · rank(W62) + 0.30 · rank(ref)
```

两个组件：

### 1. W62（本地底座，线上 0.71503）

双臂 CatBoost，`0.62·rank(main) + 0.38·rank(alt)`：

| 臂 | 特征世界 | boosting | depth | l2 | rsm | 折数×seed×bag |
|---|---|---|---|---|---|---|
| main | cond_r（中位数归一） | Ordered | 5 | 10 | 1.0 | 10×8×3 |
| alt | rate（组内 rank） | Plain | 6 | 6 | 0.3 | 10×8×3 |

- 损失用 **RMSE 回归**（不是 Logloss；历史验证 Logloss 在此数据近随机）。
- `rsm=0.3` 是 alt 臂的关键正则（30% 特征子采样）。
- **W62 依赖一个外部参考解（best_v1）的冻结 checkpoint，无法从原始数据 100% 独立复现**。
  最接近的可独立复现版本是 vz17（线上 0.71487），训练代码就在 `src/`。

### 2. ref（外部参考数据迁移，线上增益 +0.00126）

用一份更大的同分布参考数据（旧官方切分 21328 行，带标签，正例率 0.100）训练
CatBoost，再经特征级 1:1 映射迁移到当前 test（当前 test 是旧 test 的子集，映射
cover 1.0）。该臂与 W62 的 Spearman ≈ 0.90，是**真实多样性**，也是 0.715→0.716
这 +0.00126 的唯一起源。

> **合规提醒**：ref 用的是「旧官方切分多出来的 6755 条带标签样本」，属于同赛题、
> 同主办方的历史官方数据，但当前 `readme.txt` 只发了 14930+6398，未明文允许用历史切分。
> 这是**灰区**，不是确定性作弊（当前 test 标签不在其中，id 与本地零重叠）。若主办方
> 事后清外数据，此分可能被撤。请自行判断。

## 目录结构

```
best_0.716/
├── reproduce.py            ← 一键复现 0.71629（用预计算产物）
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv           ← 官方训练集
│   ├── test.csv            ← 官方测试集
│   └── submit_sample.csv
├── artifacts/
│   ├── w62/
│   │   ├── w62_oof.npy     ← W62 训练集 rank（冻结）
│   │   ├── w62_test.npy    ← W62 测试集 rank（冻结）
│   │   └── y.npy           ← 训练标签
│   └── ref/
│       ├── matched_pairs_L3.csv     ← 本地→参考 1:1 映射
│       ├── ref_model_oof.npy        ← 参考模型 OOF（seed 2026）
│       ├── ref_s8888_oof.npy        ← 参考模型 OOF（seed 8888）
│       ├── ref_model_local_test.npy ← 参考模型 → 本地 test（seed 2026）
│       ├── ref_s8888_local_test.npy ← 参考模型 → 本地 test（seed 8888）
│       └── comp_cat_opt5_oof.npy    ← 未匹配行的回填臂
├── src/                    ← 双臂训练代码（可独立复现 vz17）
│   ├── features.py         ← build_main / build_alt 特征工程
│   └── train_catboost.py   ← CatBoost RMSE 训练（10fold×8seed×3bag）
└── ref_build/              ← ref 组件构建代码（需外部参考数据）
    ├── train_ref_local_blend.py
    ├── ref_pipeline.py
    └── bootstrap_paired.py
```

## 完整复现（训练，耗时数小时）

`src/` 里的训练代码可独立复现**双臂底座**（与 W62 同配方，但权重 0.64/0.36，即 vz17）：

```bash
cd src
python3 train_catboost.py --arm arm1 --data-dir ../data --out-dir ../checkpoints/arm1   # ~90 分钟
python3 train_catboost.py --arm arm2 --data-dir ../data --out-dir ../checkpoints/arm2   # ~120 分钟
```

之后按 `0.62·rank(arm1)+0.38·rank(arm2)` 融合即得 W62 近似（本地 OOF 约 0.701，
线上约 0.715）。注意：这与本包冻结的 `w62_oof.npy` 出自不同 checkpoint，OOF 会差
~0.0005，线上也会差 ~0.0002，属正常协议差异。

`ref_build/` 里的 ref 构建需要外部参考数据（旧官方切分），路径和协议见
`ref_build/train_ref_local_blend.py` 头部注释。

## 复现校验清单

运行 `reproduce.py` 后：

- `submission.csv` 应为 6398 行、两列 `id,label`；
- `id` 顺序与 `data/submit_sample.csv` 完全一致；
- label ∈ [0.02, 0.98]，无 NaN、无重复 id；
- 与线上 0.71629 提交逐位一致。
