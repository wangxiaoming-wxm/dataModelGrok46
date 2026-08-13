# 当前最强可提交方案

更新：2026-08-13。评测 AUC。必须 Spark ML + Scala 可消费（CatBoost parquet join + `InsurerGate`）。

## 当前提交（可交）

`submissions/submission.csv`（备份 `submission_gauss_opus.csv`、`submission_before_honest10.csv`）

- 主臂：**honest10** HONEST 10 折 × 8 seed max2（opus5 FE，Classifier Logloss，固定 800 树，无早停）
  - ungated **0.70258** / nested **0.70231**（超过历史 W62 OOF **0.70159**）
  - main pool 0.69784，alt pool 0.69875
- 冻结四窗硬门后再 rank01
- 本地 **gated OOF 0.70493**
- 对照：旧 gauss(0.80 opus5-5fold + 0.15 LGB + 0.05 VAL-ES) gated 0.70456；纯 5 折 opus+硬门 0.70274

口径：honest10 为 **HONEST_NO_ES 10fold×8seed**。未混 LGB/VAL-ES。0.706 zip 本地 OOF 仅 0.694，不采用。

参考：`analysis/opus5/HONEST10_NOTES.md`。

## 配方要点

1. **Honest CatBoostClassifier + Logloss 在这套 FE 上有效**（固定 800 树、无 `use_best_model`）。
   此前「Logloss→AUC 0.51」指的是 **Regressor + Logloss**。
2. 双世界仍然成立：`cond_r` / `ratio` vs `rate=days*(1-rank(condition|source))`。
3. 10 折比 5 折同 seed 大约 +0.0015～0.006，是过 W62 的主杠杆。
4. **禁止** 对不同 CV 协议的 OOF 做逐元素 max。
5. 保司硬门（rank 融合之后，幅度冻结）：
   - `[1725,1825)`、`[2110,2210)` 全体最低
   - `[700,880)` 秩 −0.10
   - `[9370,9475)` 秩 +0.05

## 入口

```bash
python3 -u analysis/opus5/train_honest10.py --run     # 断点按 seed 跳过
python3 -u analysis/opus5/submit_honest10.py          # 写 submission.csv
# python3 -u analysis/assemble_best.py                # 会把 honest10 纳入候选
```

Spark：`BlendApp` / `CatBoostArm` 读 `submissions/final_best_*.parquet` 的 `pred_fuse`，再 `InsurerGate`。

## 记账

| 包 | 协议 | ungated | gated |
|---|---|---:|---:|
| **honest10 max2 8seed** | HONEST 10fold | **0.70258** | **0.70493** |
| gauss 0.80 opus5 + 0.15 LGB + 0.05 VAL-ES | 混 | 0.70220 | 0.70456 |
| opus5 max2 8seed | HONEST 5fold | 0.70023 | 0.70274 |
| 历史 W62 | HONEST 10fold×3bag RMSE | **0.70159** | — |
| teacher max2 + 硬门 | inner ES | 0.69207 | 0.69431 |
| 本机 RMSE W62 复现 | HONEST | 0.68863 | — |

距第 3 名 0.72384 仍约 0.019。

## 禁止再做

id 当模型特征 / 字节 TE / 伪标签；高基数 TE 喂树；同一折先全量 OOF-TE；再搜硬门平移；跨协议 OOF 逐元素 max。
