# 客户 vs 保司对抗视角

更新：2026-08-13。数字均为诚实 OOF（折内 fit）。不要把对抗 Ridge 重权融进 CatBoost。

## 1. 观测到的博弈，不是口号

标签在**概率加性尺度**生成（RMSE 有效、logloss 失败）。但保司侧有几扇**硬门**，树会因为窗口样本少而抹平：

| 窗口 | 解释 | train | 出险率 |
|---|---|---:|---:|
| `[700, 880)` | 约 2 年质保/免赔，客户报了也不赔 | 1109 | **3.07%** |
| `[1725, 1825)` | 约 5 年质保结束坑，确认集全零 p=0.006 | 103 | **0%** |
| `[9370, 9475)` | 保障期末客户倾销索赔 | 249 | **18.9%** |

Teacher max2 在这些窗上的**平均秩**仍是 0.17 / **0.22** / 0.76——5 年全零窗被排得过高。这是保司「少赔/不赔」没有被叶子学透。

客户侧索赔效用的最强单变量是

\[
\texttt{claim\_util}=\mathrm{days}\cdot(1-\mathrm{rank}(\mathrm{condition}\mid\mathrm{source}))\cdot(1-\mathbf{1}_{\mathrm{deny}})
\]

双向 AUC **0.6258**（高于 raw `rate` 0.622、`days` 0.593）。含义：暴露 × 损坏 × **未被拒赔**。新车好车况几乎不报；CAR_10 中间车况倒 U（修得值、还不至于推定全损）；CAR_7 最差十分位被核保筛掉。

## 2. 不要做的事

- 对抗 Ridge（0.643）与 teacher 秩相关 **0.80**，任何正权重融都会掉分（0.05 就从 0.691 掉到 0.690）。
- 不要把 `claim_util` / 稀疏窗口当 CatBoost 类别列（exp8b 已证明堆生成特征会从 0.692 掉到 0.679）。
- 不要在全量 OOF 上网格搜索平移量（会把 0.69364 当成成绩）。冻结常量，只用确认过的窗口。

## 3. 冻结的保司门（提交用）

在 **rank 融合之后**：

1. `days ∈ [1725,1825)` → 分数设为全体最小 − 1（严格最低）
2. `days ∈ [700,880)` → 秩 − 0.10
3. `days ∈ [9370,9475)` → 秩 + 0.05

Teacher max2 0.69207 → 门后 **0.69325**（冻结配方）。嵌套 10 折（全零窗 + 折内 750 超额 + 0.05 hot）**0.69310**。Test 同窗约 54 / 452 / 76 行。

实现：`analysis/insurer_gate.py`，Spark `claim.InsurerGate`，`BlendApp` / `TrainApp` / `write_teacher_submission.py`。

## 4. Spark 数值（下一轮 GBT/GLM，不进正在跑的 W62 列清单）

`claim_util`, `deny`, `expos_net`, `dump_poor`, `anniv_dist`, `near_anniv`, `lemon`, `new_good`, `age8_poor`, `repair_car10`, `tls_screen`

质保周年距离 `anniv_dist` 单变量 0.594，与 `days` 共线，仅给浅树用。

## 5. 冲 0.724 的主路径仍然是多种子 Ordered

对抗门大约 +0.001。从 0.693 到 0.724 必须靠 W62 配方 8seed×(1→3)bag×10fold 无 inner ES（历史 OOF 0.7016 / 线上 0.715），再叠加同一扇门。
