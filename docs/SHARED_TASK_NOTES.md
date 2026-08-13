# 共享任务笔记（主会话与子 agent 必读）

更新时间: 2026-08-13（最强可提交：gauss(0.70 honest10 + 0.30 ref) + 硬门，gated **0.70742**；底座为线上 0.71629 的 W62⊕ref30。见 `analysis/FINAL_BEST.md`）

## 目标

AUC 冲击前三：第3名 **0.72384**，第2 **0.72515**，第1 **0.74952**。
历史可复现上限 W62 线上 **0.71503**（OOF 0.70159）。必须 **Spark ML + Scala** 交付 `submissions/submission.csv`。

## 已证实（不要再走弯路）

1. **Regressor + Logloss 会崩到 AUC≈0.51**。Classifier + Logloss + 固定树数有效（honest10 / opus5）。RMSE 回归 0/1 仍可用，但已被 Classifier 超过。
2. 双世界：`cond_r=condition/median(condition|source)` + `ratio=days/cond_r`；`rate=days*(1-rank(condition|source))`。
3. **禁止 id 及字节 TE**（vz19 线上负迁移）。禁止伪标签。
4. **高基数 TE 不能作为 GBT/LGB 的数值特征**：LOO-TE 喂 LGB 会 best_iter=1、AUC≈0.50。TE 只能当**独立分数**做 rank 融合。
5. 同一折先算全量 OOF-TE 再训模型会**泄漏**，曾虚高到 0.733，诚实嵌套后崩盘。必须 outer-fold 内 fit。
6. CatBoost 原生类别 + RMSE + 我们加的 3 阶交叉，诚实 5 折只有 **0.685**（少于 W62 的 0.701）。稀疏 `src|cond_q|days_q` 不宜作为 CatBoost 类别列；应作为独立 TE 分数。
7. 诚实数字：
   - 数值 LGB/GBT ≈ **0.655–0.66**
   - TE `source×cond_q×days_q` ≈ **0.633**
   - 数值+TE rank 融合 ≈ **0.660**
   - CatBoost 双臂 5 折 ≈ **0.685**
8. condition×source 形状不同：CAR_1 单调保护，CAR_10 倒 U（中间五分位 25.7%）。
9. 丢弃 x18（纯噪声）、x19（=V²）。
10. train/test 干净随机切分，几乎无漂移。

## 业务机理要写进特征

- 客户侧：车龄暴露、车况差更愿索赔、新车+好车况几乎不报、高龄右尾、地区异质。
- 保险公司侧：先按 source 分层核保；condition 效应形状随车型变；噪声列会稀释分裂。

## 代码位置

- Spark 工程：`/workspace/spark-claim/` 入口 `claim.TrainApp`
- 设计：`/workspace/docs/TECHNICAL_DESIGN.md`
- 数据：`/workspace/data/{train,test,submit_sample}.csv`
- 分析探针：`/workspace/analysis/`

## 子 agent 分工（禁止互相覆盖核心文件时不先读）

- **reverse-engineer**：只写 `analysis/`，冲击诚实 OOF≥0.70 的新表示。
- **spark-runner**：让 `spark-claim` 编译跑通，产出 submission，修 bug。
- **catboost-arm**：新增 `CatBoostSpark.scala`（Spark ML Estimator），不要拆掉 GBT 臂，做 rank 融合。

## 停止条件

没有诚实 OOF 证据不要宣称胜利。打不到 0.723 不许放弃，换表示再试。
