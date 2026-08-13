# 当前最强可提交方案

更新：2026-08-13。评测 AUC。必须 Spark ML + Scala 可消费（CatBoost parquet join + `InsurerGate`）。

## 当前提交（可交）

`submissions/submission.csv`（备份 `submission_opus5_gated.csv`）

- 主臂：**opus5 HONEST** 8-seed max2
  - `merger_ord8`：v2 主帧 + Ordered Classifier Logloss，depth=5，固定 800 树，5 折 × 8 seed
  - `v2_cat_alt8`：alt 世界 `rate=days*(1-rank(condition|source))`，Plain d6 l2=6
  - 逐元素 max(rank) 融合，nested OOF **0.69993** / full **0.70023**
- 多样性臂：本仓库 LightGBM RMSE 10 折，权重 **0.15**
- 冻结四窗保司硬门后再 rank01
- 本地 **gated OOF 0.70335**（nested 0.70056）
- 纯 opus_max2 + 硬门：gated **0.70274**（nested 0.69993）

口径：opus5 为 **HONEST_NO_ES**（无早停）。LGB 外折 ES 仅 15% 权重。禁止把 VAL-ES 数字写成 HONEST。

参考包：`analysis/opus5/`（20260810-curos-opus5）。

## 配方要点

1. **Honest CatBoostClassifier + Logloss 在这套 FE 上有效**（固定 800 树、无 `use_best_model`）。
   此前「Logloss→AUC 0.51」指的是 **Regressor + Logloss**，不要和 Classifier 混为一谈。
2. 双世界仍然成立：`cond_r` / `ratio` vs `rate=days*(1-rank(condition|source))`。
3. opus5 FE：train+test 一次性拟合分位切点（label-free 转导）；jitter 用 id 哈希当 RNG，不当特征列。
4. **禁止** 对不同 CV 协议的 OOF 做逐元素 max（会捡折内运气，虚高到 0.707）。
5. 保司硬门（rank 融合之后，幅度冻结）：
   - `[1725,1825)`、`[2110,2210)` 全体最低
   - `[700,880)` 秩 −0.10
   - `[9370,9475)` 秩 +0.05

## 入口

```bash
python3 -u analysis/assemble_best.py
# 完整重训 opus5 约 150 min（1 核更长），数据路径已改 /workspace/data
# bash analysis/opus5/reproduce.sh
```

Spark：`BlendApp` / `CatBoostArm` 读 `submissions/final_best_*.parquet` 的 `pred_fuse`，再 `InsurerGate`。

`final_best.py` 的 VAL-ES 4seed 仍可续跑作对照，**结束后必须再跑 assemble_best.py**，避免弱包覆盖 0.703 提交。

## 记账

| 包 | 协议 | ungated | gated |
|---|---|---:|---:|
| opus5 max2 8seed | HONEST 5fold | 0.70023 | **0.70274** |
| 0.85 opus + 0.15 LGB | HONEST + 外折 ES | 0.70084 | **0.70335** |
| VAL-ES seed2026 ⊕ LGB | VAL_ES | ~0.696 | 0.69816 |
| teacher max2 + 硬门 | inner ES | 0.69207 | 0.69431 |
| HONEST 本机 8seed×2bag W62 RMSE | HONEST | 0.68863 | — |

距第 3 名 0.72384 仍约 0.020。

## 禁止再做

id 当模型特征 / 字节 TE / 伪标签；高基数 TE 喂树；同一折先全量 OOF-TE；再搜硬门平移；跨协议 OOF 逐元素 max；1 核重跑 8seed×3bag RMSE HONEST（已被 opus5 Classifier 包超过）。
