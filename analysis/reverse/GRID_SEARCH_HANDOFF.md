# 大规模网格搜索 — 交给别人执行的完整提示词（不许停）

把下面「执行提示词」整段复制给另一台 **≥8 核、≥32GB** 的机器上的 agent。  
本 Cloud VM 只有 **1 个 vCPU**（`nproc=1`），无法给逆向子任务 2–3 核；W62 8-seed 仍在本机跑，不要杀它来「腾核」。

当前仓库工作区可能在 `cursor/reverse-generating-process-7783`。交付 Spark 代码在 `cursor/sparkml-oof-boost-c687`。**禁止 `git checkout` / `git switch` / `git reset --hard`**（共享盘切分支会毁掉正在跑的训练）。

---

## 执行提示词（从这里复制到下一个 agent）

```
你是车险索赔预测的资深 ML 工程师。任务：把诚实 OOF AUC 从 ~0.696 推到 ≥0.701（对齐历史 W62），并冲击线上前三 0.72384。大规模网格搜索不许停，不许早早放弃。

# 赛题
- 数据：/workspace/data/train.csv（14930×45，正例率 0.1002）、test.csv（6398×44）
- 指标：AUC。提交 submissions/submission.csv，列 id,label，6398 行，id 顺序与 test.csv 一致，label∈[0,1]
- 必须最终能被 Spark ML Scala 工程消费（CatBoost parquet join + InsurerGate）。本阶段允许 Python CatBoost 出 teacher parquet。

# 铁律（违反即废）
1. 目标 RMSE 回归 0/1。禁止 Logloss/CrossEntropy（会掉到 AUC≈0.51）。
2. 折内 fit：median/rank/qcut/TE 只在训练折。禁止全量 TE、test 伪标签、id 特征。
3. 禁止把 src|cond_q|days_q（~1000 水平）当 CatBoost/GBT 类别列。3 路交叉只做独立 TE 分数再 rank 融合。
4. 高基数 TE 禁止喂树当数值列（会 best_iter=1、AUC≈0.50）。
5. 丢弃 x18（噪声）、x19（=V²）。
6. 两种协议必须分开记账，禁止把 val-fold ES 的 0.696 写成「诚实 W62」：
   - HONEST_NO_ES：训练折全量 800 iter，无 eval_set（历史 W62 OOF 0.70159 / 线上 0.71503）
   - VAL_ES：early stopping 打在 OOF 验证折上（exp8h 已到 max2 0.69592，略乐观）
7. 禁止 git checkout/switch/reset-hard。只写 analysis/reverse/ 与 submissions/cb_*.parquet。
8. 每完成一个格子立刻写 json/npz checkpoint，崩溃可续跑。
9. 打不到 0.723 不许停；但不要用泄漏数字当成绩。

# 已有最强数字（不要从零开始）
- Spark 提交：teacher 10fold×3bag×1seed max2 0.69207 + 冻结保司硬门 0.69353
- exp8h VAL_ES 8seed×3bag×10fold：w62=0.69580，max2=0.69592（文件 analysis/reverse/exp8h_*.npy/json）
- exp8j 全 2 路类别 2seed：max2 0.69509
- 1seed×1bag HONEST_NO_ES：~0.683–0.686，必须多种子才有意义
- 便携无 CatBoost：0.670–0.682，不要当主臂
- 对抗 Ridge / 80 路 TE Ridge 与 CB 相关高，重权融合会掉分；弱权 ≤0.15 才允许试
- 保司硬门（冻结，禁止再搜平移量）：rank 融合后
    days∈[1725,1825) → 全体最低
    days∈[700,880) → 秩-0.10
    days∈[9370,9475) → 秩+0.05
  然后再 rank01 压回 [0,1] 写提交。实现：analysis/insurer_gate.py 或 spark-claim/.../InsurerGate.scala

# 双世界特征（必须）
Arm-Main：cond_r=condition/median(condition|source)，ratio=days/cond_r
Arm-Alt：rate=days*(1-rank(condition|source))
中基数类别（默认 CATS，无 3 路）：source, region, age_range, grades, month, 以及 2 路 src_reg/src_age/reg_age/src_cq/src_dq5/reg_cq/reg_dq/src_ratioq，加上 days_q5, cond_q10, ratio_q
代码：analysis/reverse/cb_features.py 的 fold_features；80 路交叉在 analysis/reverse/feat80.py（atomic=13，sem=atomic+语义对，80=全部 pairwise）

# 默认双臂超参
Arm1 Ordered: depth=5, l2=10, rsm=1.0, lr=0.03, RMSE, bagging_temperature=0.2, border_count=128
Arm2 Plain:   depth=6, l2=6,  rsm=0.3, lr=0.03, RMSE, bagging_temperature=1.0, border_count=64
融合：0.62*rank(main)+0.38*rank(alt)，也报告 max2=max(rank main, rank alt)

# 网格（必须全部跑完，按阶段，每格 checkpoint）

## 阶段 A — 表示筛选（5fold × 1bag × 1seed，VAL_ES，只为排序）
脚本：analysis/reverse/exp9_hunt.py（THREADS=1 可改成 nproc）
格子（已部分跑；跳过 exp9_cb_grid.json 里已有的 name）：
  cur_def / cur_c1 / atom_c4 / atom_c2 / sem_c1 / sem_c4 / c80_c1
含义：current|atomic|sem|80 类别集 × max_ctr_complexity ∈ {None,1,2,4}
续跑：python3 -u analysis/reverse/exp9_hunt.py 2>&1 | tee -a analysis/reverse/exp9_hunt.log
若 5fold OOF < 0.68 的表示，阶段 C 不再放大。

## 阶段 B — 超参×类别集筛选（5fold × 1bag × 1seed Ordered 主臂，VAL_ES）
对阶段 A 的 top-3 表示，扫：
  depth ∈ {4,5,6,7}
  l2_leaf_reg ∈ {3,6,10,20}
  learning_rate ∈ {0.02, 0.03, 0.05}
  rsm ∈ {0.3, 0.6, 1.0}          # Plain 臂必带 0.3
  border_count ∈ {64, 128, 254}
  bagging_temperature ∈ {0.2, 1.0}
  max_ctr_complexity ∈ {1, 2, 4, None}
不要笛卡尔全乘爆掉：用「默认 W62 为中心、每次动 1–2 维」的坐标下降，再对胜者做 2 维局部网格。
每个配置写 analysis/reverse/gridB/<name>.json，包含 fold AUC 列表和 OOF。

额外必测类别集（仍禁止 3 路当 CB 类别）：
  CATS_W62（16 列，analysis/cb_features.py）
  exp8j 的 23 列全 2 路
  CATS_SEM（feat80.py）
  不要测把 src_cq_dq 加进 cats 的配置（历史 0.685 < 0.701）

## 阶段 C — 放大（只放大阶段 B 的 top-3）
对每个 top 配置跑两种协议，都要 10fold × 8seed × 3bag：
  C1 HONEST_NO_ES：仿 analysis/cb_w62_full.py（无 eval_set，800 iter）
     env: CB_BAGS=3 CB_THREADS=<nproc-1或8> CB_ITERS=800 CB_NFOLD=10
     CB_SEEDS=2026,2027,2028,2029,2030,2031,2032,2033
  C2 VAL_ES：仿 analysis/reverse/exp8h_vales.py（eval_set=OOF 折，od_wait=80）
     seeds 2026,2036,2046,2056,2066,2076,2086,2096
每 seed 写 npz checkpoint。目标：HONEST_NO_ES ≥ 0.700；VAL_ES 只作参考。

## 阶段 D — 提交
1. 在 HONEST 最强 OOF 上套冻结 InsurerGate，报告门前/门后 AUC
2. 用同一套折模型预测 test（折平均），再对 test 做同样硬门 + rank01
3. 写 submissions/submission.csv 与 submissions/cb_w62_oof.parquet、cb_w62_test.parquet、cb_w62_metrics.json
   parquet 列：oof 要有 id,label,pred_cb_main,pred_cb_alt,pred_cb_w62；test 要有 id,pred_cb_main,pred_cb_alt,pred_cb_w62
4. Spark 侧会 join 这些 parquet；不要改坏 bucketize；不要同时开两个 sbt

# 硬件
推荐 8–16 核、32GB。thread_count 设为 min(8, nproc)。本 1 核机器不要把 THREADS 设成 4（会和 W62 抢到 load>3）。
内存不够时：先停 Spark，保留 CatBoost；不要两个 CatBoost 各 thread_count≥4 同时跑。

# 停止与报告
- 每个阶段结束打印：配置、协议（HONEST vs VAL_ES）、nfold/nseed/nbag、OOF、门后 OOF、是否 ≥0.70
- 没有诚实 OOF 证据不许宣称胜利
- 网格没跑完不许停
```

---

## 本机现状（2026-08-13 06:30 UTC）

| 项 | 值 |
|---|---|
| `nproc` | **1**（无法分 2–3 核给逆向） |
| W62 8-seed | **仍在跑** seed 2027 fold 5/10，1bag HONEST≈0.68x，不要杀 |
| 逆向 | `exp9_hunt.py` 正在跑 5fold 表示网格，`thread_count=1` |
| 已完成 VAL_ES | exp8h 8seed max2 **0.69592** |
| 提交 | teacher+硬门 **0.69353** |

别人接手时优先拷贝：`analysis/reverse/`、`analysis/cb_features.py`、`analysis/cb_w62_full.py`、`analysis/insurer_gate.py`、`data/`、`submissions/cb_teacher_*.parquet`。
