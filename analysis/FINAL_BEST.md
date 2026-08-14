# 当前最强可提交方案

更新：2026-08-14。评测 AUC。必须 Spark ML + Scala 可消费。

## 当前提交（可交）

`submissions/submission.csv`（bag0 备份 `submission_h10_ref30_gauss.csv`）

线上已验证底座：`best_0.716` = W62⊕ref30 = **0.71629**。
本交在冻结 `w_ref=0.30` 下把 W62 换成 honest10 的 **bag0+bag1 rank-pool**，再套 gauss + 四窗硬门：

`gauss(0.70·honest10_bag01 + 0.30·ref) + gate`

- honest10：HONEST 10 折×8 seed×2 bag Classifier max2，ungated **0.70295**
- ref：0.716 包的映射臂（旧官方切分，灰区），权重不重搜
- 本地 **ungated 0.70538 / nested 0.70508 / gated 0.70774**
- vs 上一交 bag0 gauss+gate（0.70742）：gated Δ=+0.00032，bootstrap CI [-0.00027,+0.00088]，p_pos=0.865（区间含 0，不宣称稳过线上）

参考：`analysis/best716/NOTES.md`。

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
python3 -u analysis/opus5/train_honest10.py --run --bag 1
python3 -u analysis/opus5/eval_bag_pool.py            # 默认 dry-run
python3 -u analysis/opus5/eval_bag_pool.py --write    # 仅当 gated+nested 都过锁定配方
```

Spark：`BlendApp` / `CatBoostArm` 读 `submissions/final_best_*.parquet` 的 `pred_fuse`，再 `InsurerGate`。

## 记账

| 包 | 协议 | ungated | gated |
|---|---|---:|---:|
| **gauss h10 bag0+bag1 ⊕ref30 + 硬门** | HONEST 2bag + 灰区 ref | **0.70538** | **0.70774** |
| gauss h10 bag0 ⊕ref30 + 硬门 | HONEST 1bag + 灰区 ref | 0.70512 | 0.70742 |
| **honest10 max2 8seed bag0** | HONEST 10fold | **0.70258** | **0.70493** |
| gauss 0.80 opus5 + 0.15 LGB + 0.05 VAL-ES | 混 | 0.70220 | 0.70456 |
| opus5 max2 8seed | HONEST 5fold | 0.70023 | 0.70274 |
| 历史 W62 | HONEST 10fold×3bag RMSE | **0.70159** | — |
| teacher max2 + 硬门 | inner ES | 0.69207 | 0.69431 |
| 本机 RMSE W62 复现 | HONEST | 0.68863 | — |

距第 3 名 0.72384 仍约 0.019。

## 禁止再做

id 当模型特征 / 字节 TE / 伪标签；高基数 TE 喂树；同一折先全量 OOF-TE；再搜硬门平移；跨协议 OOF 逐元素 max。
