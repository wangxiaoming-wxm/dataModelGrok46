# 车险索赔预测（Spark ML / Scala）冲击前三名技术设计

## 1. 目标与约束

| 项 | 值 |
|---|---|
| 任务 | 预测投保人未来一年是否索赔（二分类概率） |
| 指标 | AUC，越大越好 |
| 前三阈值 | 第3名 0.72384，第2名 0.72515，第1名 0.74952 |
| 当前公开可复现上限 | W62 线上 0.71503（本地 OOF 0.70159） |
| 实现约束 | **必须 Spark ML + Scala**；禁止 test 标签/伪标签/外部数据；折内特征工程 |
| 数据 | train 14930×45（正例率 10.02%），test 6398×44 |

必须跨过的缺口：相对 W62 约 **+0.009（进前三）**，相对第1名约 **+0.034**。
第1名与第2名之间存在 0.024 的异常断层，优先假设其找到了更接近数据生成过程的表示，而不是普通调参。

## 2. 双视角业务机理（写入特征）

### 2.1 客户侧：什么情况下最可能提出索赔

- **车龄暴露**：`days` 是最强原始列（AUC 0.593）。车越旧，故障/事故暴露越长，索赔动机越强。
- **车况作为尺度分母**：风险近似 `days / condition^0.5` 或 `days / (condition / median(condition|source))`。车况差时，同样车龄更可能修车并走保险。
- **新车+好车况几乎不赔**：二维分箱显示 days 最低两档且 condition 非最差时，出险率可低至 1.3%–5%；新车但 condition 极差（疑似瑕疵车/带伤投保）仍有 12%–15%。
- **高龄驾驶人右尾**：`age_range>=8` 出险率 16.5%（n=158），符合反应能力/风险暴露。
- **地区风险异质**：`region=f09d` 11.9% vs `c1f5` 3.7%，且 `days` 与 label 的相关在地区间会变号。
- **局部“免赔/不报”窗口**：`days∈[1725,1825)` 103 单 **零索赔**；`[700,880]` 出险率 3.07%。树会抹平，融合后走冻结保司硬门。
- **索赔效用**：`claim_util = days*(1-rank(condition|source))*(1-deny)` 单变量 AUC 0.626。

### 2.2 保险公司侧：什么情况下更可能赔、如何少赔

- **车型费率分组 `source` 是根节点**：11 个车型决定排量/动力/大量 x 列；出险率 5.6%（CAR_8）–15.6%（CAR_10）。核保应先按车型分层。
- **condition 效应在车型间连形状都不同**（不是简单强弱）：
  - CAR_1：单调保护，最差五分位 12.5% → 最好 1.4%
  - CAR_10：倒 U 型，中间五分位高达 25.7%，两端约 4.5%
  - CAR_7：最差五分位仅 2.0%（与 CAR_1 相反）
  这要求 **source×condition 交互必须显式进入模型**，浅树+全局可加模型会抹平。
- **冗余与噪声**：`x19=V²`，`x18` 为按车型振幅的均匀噪声，`code/t3_letter` 由 source 完全决定。把噪声当信号会稀释 GBT 分裂。
- **不要用分类 logloss**：历史实证 Logloss/CrossEntropy 的 OOF≈0.51（近随机），**RMSE 回归 0/1 标签**才能学到概率尺度上的加性结构（更像频率模型而不是 logit GLM）。
- **高基数目标编码必须折内**：全量 TE 本地虚高 0.74、线上崩到 0.66。id 字节 TE 本地涨、线上跌。禁止 id 特征。

## 3. 已证实的有效配方（必须迁移）

1. 双编码世界：
   - Arm-Main：`cond_r = condition / median(condition|source)`，核心 `ratio = days / cond_r`
   - Arm-Alt：`rk = rank_pct(condition|source)`，核心 `rate = days * (1-rk)`
2. 目标函数：**RMSE**（Spark `GBTRegressor` / `lossType=squared`）
3. 80+ 类别交叉 + 有序/折外目标编码（CatBoost 的真正护城河；Spark 无原生 CatBoost，必须手写）
4. 10 折分层 CV、多种子、3-bagging
5. 融合：`0.62 * rank(main) + 0.38 * rank(alt)` 优于 max2
6. Arm1：偏浅、强 L2（depth≈5）；Arm2：略深 + 特征子空间 rsm≈0.3

Spark GBT 弱于 CatBoost Ordered Boosting。补偿策略：把 CatBoost 的类别组合+有序编码做成数值特征，再喂 GBT/RF/FM/GLM，用多样性追回。

## 4. Spark ML 架构

```
CSV
  -> Spark DataFrame
  -> FeatureBuilder（确定性派生、双世界比值、分箱、交叉键）
  -> KFoldTargetEncoder（仅用训练折统计，平滑 m=20）
  -> VectorAssembler
  -> 多模型：
       GBTRegressor (squared, depth 5/6, 双臂超参)
       RandomForestRegressor
       LinearRegression / GeneralizedLinearRegression(gaussian)
       FMRegressor（成对交互）
       大车型子集 GBT（CAR_0/1/2）
  -> OOF 概率
  -> Rank 加权融合（嵌套CV选权，默认 0.62/0.38）
  -> 校准（保序回归，可选；AUC 对单调变换不敏感，校准主要为了概率质量）
  -> submissions/spark_*.csv
```

全部代码位于 `spark-claim/`，入口 `claim.TrainApp`。

## 5. 特征清单（折内）

### 5.1 数值

- days, log1p(days), sqrt(days)
- condition_f（缺失用 source 中位数，再全局中位数），log, sqrt, inv, condition^2
- cond_r, cond_rk, (cond_rk-0.5)^2
- ratio, rate, days/sqrt(condition), log(days)-0.5*log(condition)
- days * inv(condition)
- age_range, I(age>=8), I(condition<0.05)
- 窗口指示：I(days∈[700,880]), I(days∈[1725,1825]), I(days∈[1950,2000]), I(days<50)
- V, cc, max_g, x1, x5, x14, x17, x20（弱信号保留；**丢弃 x18、x19**）
- t3_num；cond_missing 指示

### 5.2 类别与交叉（全部 K-fold TE + 频次编码）

单列：source, region, age_range, grades, code, month, days_q10, cond_q10, ratio_q10, rate_q10  
交叉：source×region, source×age, region×age, source×cond_q, source×days_q, region×cond_q, region×days_q, cond_q×days_q, source×ratio_q, source×rate_q, source×grades, source×cond_q×days_q, source×cond_q×age, region×cond_q×days_q, source×region×age（稀疏，平滑加大）

禁止：id 及任何字节/nibble 编码。

## 6. 验证与防过拟合

- Stratified 10-fold，种子固定可复现
- 编码器只在训练折 fit
- 用 OOF AUC 做唯一选择标准；融合权重用嵌套折，避免用全量 OOF 调权过拟合
- 不追求 OOF 虚高：历史证明 OOF 与线上非单调（vz19）
- 对抗验证已确认 train/test 几乎无漂移（PSI<0.006），可用随机分层而非时间切分

## 7. 冲击 0.724+ 的加分项（按优先级）

1. **忠实复现双世界+交叉 TE+RMSE**（基础盘，目标 OOF≥0.695）
2. **有序目标编码**（按随机置换的 expanding mean，模拟 CatBoost ordered TS）
3. **车型条件形状特征**：按 source 的 condition 分箱 TE（已包含）+ 显式 U 形项
4. **大车型专模残差**（样本≥800 的 source）与全局 rank 融合
5. **FM / GLM 提供低相关臂**（多样性比第三棵同类树更重要）
6. **保司硬门（冻结）**：rank 融合后把 `[1725,1825)` 打到最低、`[700,880)` 降秩、`[9370,9475)` 提秩。Teacher + 门诚实 OOF **0.6931**。不要把对抗 Ridge 融进 CB。
7. **继续逆向生成过程**：分段 days、source 特定 f(condition)，把解析式当特征

## 8. 工程运行

- Spark 3.5.5 / Scala 2.12.18 / local[4] / driver memory 12g
- `sbt package` 后 `spark-submit`
- 每 seed 写 checkpoint，避免长任务丢失
- 产出：`submissions/submission.csv`（id,label），以及 OOF 报告

## 9. 明确不做

- 伪标签、test 泄漏、id 哈希投机
- Spark 里用 LogisticRegression 当主模型（可作弱多样性臂）
- SMOTE / 随机过采样
- 把 x18、version 当核心信号
