# CatBoost 表示网格 — 交接提示词（testsb 分支）

仓库：https://github.com/wangxiaoming-wxm/dataModelGrok46  
分支：`testsb`  
所有文件路径均为 GitHub 全路径。clone 后对应仓库相对路径去掉 `blob/testsb/` 即可。

```
git clone -b testsb https://github.com/wangxiaoming-wxm/dataModelGrok46.git
```

---

你是独立执行者。目标：在 **HONEST_NO_ES** 协议下把 CatBoost OOF 从 0.69x 推到 **≥0.700**（历史可复现上限 0.70159 / 线上 0.71503），再叠冻结保司硬门写提交。评测 **AUC**。必须用 **RMSE** 回归 0/1，禁止 Logloss。

公开榜：1=0.74952（断层） 2=0.72515 3=0.72384。当前诚实提交 0.69353。不达前三不放弃。

---

## 0. 两种协议（必须分开记账）

| 协议 | 做法 | 可报成绩 |
|---|---|---|
| **HONEST_NO_ES** | 训练折 800 iter、无 eval_set、无 inner 12% | 历史 W62 0.70159 |
| **VAL_ES** | ES 打在 OOF 验证折 | exp8h 0.69592，略乐观 |

阶段 A/B 可用 VAL_ES 筛表示。阶段 C 必须两协议都跑。提交只认 HONEST。

---

## 1. 硬件

当前 Cloud VM `nproc=1`、15GB。1 核跑不完 8seed×3bag。请到 **≥8 核、≥32GB** 机器执行本网格。`thread_count` 用满可用核。不要在 1 核机器上开大网格。

---

## 2. 数据（已在仓库）

- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/data/train.csv
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/data/test.csv
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/data/submit_sample.csv

clone 后本地：`data/train.csv`（14930×45，正例率 0.1002）、`data/test.csv`（6398×44）。

---

## 3. 特征 / 训练入口

- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/cb_features.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/cb_w62_full.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/insurer_gate.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/write_teacher_submission.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/cb_features.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/feat80.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp9_hunt.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp9_cb_grid.json
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8h_vales.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8h_vales.json
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8h_max2.npy
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8h_w62.npy
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8h_vales_ckpt.npz
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8j_fullcats.py
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8j_fullcats.json
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/best_oof.npy
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/GENERATING_PROCESS.md
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/docs/TECHNICAL_DESIGN.md
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/docs/SHARED_TASK_NOTES.md

---

## 4. Spark 工程（比赛硬性要求 Spark ML + Scala）

- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/build.sbt
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/TrainApp.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/BlendApp.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/CatBoostArm.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/FeatureEngine.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/InsurerGate.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/GbtTrainer.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/TargetEncode.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/LinearArms.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/OrderedArm.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/OrderedTE.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/src/main/scala/claim/Blend.scala
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/teacher/cb_teacher_oof.parquet
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/teacher/cb_teacher_test.parquet
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/spark-claim/teacher/cb_teacher_metrics.json

---

## 5. 当前提交与 teacher 包

- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/submission.csv
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/cb_teacher_oof.parquet
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/cb_teacher_test.parquet
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/cb_teacher_metrics.json
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/cb_w62_oof.parquet
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/cb_w62_test.parquet
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/cb_w62_metrics.json
- https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/submissions/oof_report.txt

当前提交：teacher max2 + 保司硬门，诚实 OOF **0.69353**。  
`cb_w62_*.parquet` 是 1seed×1bag HONEST（约 0.683），**弱于** teacher 3-bag。选包必须按诚实 AUC，禁止“有 w62 文件就用”。

---

## 6. 已冻结、禁止再搜

1. 损失 = RMSE，禁止 Logloss/CrossEntropy。
2. 禁止 id / 字节 TE / 伪标签。
3. 高基数 TE 不能进 GBT/LGB 数值特征（best_iter=1）。TE 只当独立分数 rank 融合。
4. 丢弃 x18、x19。
5. 稀疏 `src|cond_q|days_q` 不要当 CatBoost 类别列。
6. 保司硬门冻结（rank 融合之后）：
   - days ∈ [1725,1825) → 全体最低（train n=103 全零）
   - days ∈ [700,880) → 秩 −0.10
   - days ∈ [9370,9475) → 秩 +0.05
   然后再 rank01 压回 [0,1]。实现见 `analysis/insurer_gate.py`。
7. 对抗 Ridge / 两段式 p_cover*p_freq 不要当主臂。
8. Inner 12% ES 会把 teacher 从 0.701 降到 0.692。对齐 W62 用全量训练折 + 固定 800 iter。
9. `bucketize` 必须只用有限分位点、`c < cut(i)`。`c >= -Infinity` 会把所有样本打进 bucket 0。
10. 3 路交叉 `src|cond_q|days_q` 不要当 CB cats（exp8j 已否）。

---

## 7. 必须保留的结构

双世界特征：
- `cond_r = condition / median(condition|source)`，`ratio = days / cond_r`
- `rate = days * (1 - rank(condition|source))`

双臂：
- Main：Ordered, depth=5, l2=10, rsm=1
- Alt：Plain, depth=6, l2=6, rsm=0.3
- 融合：`0.62*rank(main)+0.38*rank(alt)`；强 CB 也可用 max2

类别列默认 `CATS_W62`。exp8j 证明 23 列 VAL_ES 2seed 只到 0.69509，不是银弹。

---

## 8. 网格怎么跑

工作目录：仓库根。Python 路径指向 `analysis/` 与 `analysis/reverse/`。

### 阶段 A — 表示筛选（5fold × 2seed × 1bag，VAL_ES 可）

入口：https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp9_hunt.py

断点：https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp9_cb_grid.json  
已有 name 直接跳过。

候选（`feat80.py` 已实现）：`cur_def` `cur_c1` `atom_c4` `atom_c2` `sem_c1` `sem_c4` `c80_c1`。  
5fold < 0.68 的坐标不再放大。

### 阶段 B — 超参坐标下降（10fold × 2seed × 1bag）

对 A 的 top-3：depth ∈ {4,5,6,7}，l2 ∈ {3,6,10,20}，lr ∈ {0.03,0.05,0.08}，rsm ∈ {0.3,0.5,1.0}，border_count ∈ {64,128,254}，bagging_temperature ∈ {0,0.5,1}，max_ctr_complexity ∈ {2,4}。  
类别集：`CATS_W62`、exp8j 的 23 列、`CATS_SEM`。禁止 3 路当 CB cats。

### 阶段 C — 放大（必须两协议）

对 B 的 top-3 各跑 10fold × 8seed × 3bag：
- C1 HONEST：仿 https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/cb_w62_full.py ，设 `CB_BAGS=3`
- C2 VAL_ES：仿 https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/reverse/exp8h_vales.py

目标 HONEST ≥ 0.700。达不到换下一表示，不要停。

### 阶段 D — 提交

最强 HONEST OOF + 冻结硬门 → `submissions/submission.csv`。  
同步写 `submissions/cb_w62_{oof,test}.parquet` 与 metrics，供 Spark `CatBoostArm.joinTeacher` 使用。选包按诚实 AUC，不要用弱 1bag parquet 覆盖 3bag teacher。

写提交脚本：https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/write_teacher_submission.py  
硬门：https://github.com/wangxiaoming-wxm/dataModelGrok46/blob/testsb/analysis/insurer_gate.py

---

## 9. 已知分数（对照，不要当目标上限）

| 模型 | 协议 | AUC |
|---|---|---|
| 历史 W62 8seed×3bag | HONEST | 0.70159 / 线上 0.71503 |
| teacher 10fold×3bag×1seed + 硬门 | HONEST | **0.69353**（当前提交） |
| teacher max2 无硬门 | inner 12% ES | 0.69207 |
| exp8h 8seed×3bag | VAL_ES | w62 0.69580 / max2 0.69592 |
| exp8j 23 类 2seed | VAL_ES | max2 0.69509 |
| 1seed×1bag HONEST w62 parquet | HONEST | ~0.683（弱于 teacher，勿选） |
| 便携无树 / +GBT | — | 0.6705 / 0.6822 |

---

## 10. 禁止

- 禁止 Logloss。
- 禁止 id/字节 TE/伪标签。
- 禁止把高基数 TE 喂给树模型当数值列。
- 禁止泄漏：同一折先全量 OOF-TE 再训模型。
- 禁止用 VAL_ES 分数冒充 HONEST 上报。
- 禁止 `git checkout` / `git switch` / `git reset --hard` 切共享工作区（会杀掉别人的 Spark/CatBoost）。用 `git worktree`。
- 不要提交 `analysis/cb_w62_ckpt/`（约 52MB 训练断点）。npy/npz 用 `git add -f`（`.gitignore` 忽略了 `*.npy`）。

---

## 11. 跑完后

把 HONEST 分数、VAL_ES 分数、用了哪组表示/超参、新的 parquet 路径写进本文件末尾，并推回 `testsb` 或新分支。
