# 当前最强可提交方案

更新：2026-08-13。评测 AUC。必须 Spark ML + Scala 可消费（CatBoost parquet join + `InsurerGate`）。

## 配方（按诚实证据锁定）

1. **主臂**：CatBoost 双世界 RMSE（禁止 Logloss）
   - Ordered d5 l2=10 + Plain d6 l2=6 rsm=0.3
   - 10 折 × 3 bag × 多种子；VAL-ES（early stopping 打在 OOF 验证折）
   - 23 列中基数类别 `CATS_BEST`，**禁止** `src|cond_q|days_q`
2. **多样性臂**：LightGBM 双世界 RMSE，同一 10 折，外折 ES
3. **Teacher 对照**：已有 10fold×3bag×1seed inner-ES parquet（ungated max2 **0.69207**）
4. **Rank 融合**：在冻结候选里按 **硬门后 OOF** 选包，不允许“文件名带 w62 就用”
5. **保司硬门**（rank 融合之后，幅度冻结）：
   - `[1725,1825)`、`[2110,2210)` 全体最低
   - `[700,880)` 秩 −0.10
   - `[9370,9475)` 秩 +0.05
   - 再 `rank01` → `submissions/submission.csv`

## 入口

```bash
# 训练（可断点续跑 analysis/final_ckpt/）
FINAL_SEEDS=2026,2036,2046,2056 FINAL_BAGS=3 FINAL_THREADS=1 \
  python3 -u analysis/final_best.py | tee analysis/final_best.log

# 只组装当前已有 OOF/test（训练中也可跑）
python3 -u analysis/assemble_best.py
```

Spark：`claim.BlendApp` / `CatBoostArm.joinTeacher` 读 `submissions/final_best_*.parquet`（`pred_fuse`），再跑 `InsurerGate`。

## 记账口径

| 协议 | 含义 | 不可冒充 |
|---|---|---|
| HONEST_NO_ES | 训练折全量 800 iter | 历史 W62 0.70159 / 线上 0.71503；本机 8seed×2bag 仅 0.68863 |
| VAL_ES | ES 打在 OOF 折 | 略乐观；可提交，但报告里必须写 VAL_ES |
| inner 12% ES | teacher 3-bag | ungated 0.69207 |

`best_oof.npy` 的 VAL_ES 融合 **0.69713**（硬门后约 0.699）**没有 test 预测**，不能直接交。

## 禁止再做

id / 字节 TE / 伪标签；高基数 TE 喂树；同一折先全量 OOF-TE；丢弃 x18/x19 之外再搜硬门平移；1 核上重跑 8seed×3bag HONEST。
