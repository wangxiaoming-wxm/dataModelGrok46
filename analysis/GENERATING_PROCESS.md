# 车险索赔生成过程（逆向工程）

更新：2026-08-13。全部数字为 **StratifiedKFold、折内 fit** 的诚实 OOF AUC。正例率 0.1002。  
禁止口径：全量 TE、test 伪标签、id 特征。历史泄漏 0.73 不作真实水平。

当前最佳诚实 OOF（见文末配方）会随多种子 CatBoost 更新；中间融合已到 **0.6936**。

---

## 0. 生成过程总公式（概率尺度，不是 logit）

RMSE 有效、logloss 失败，且 **加法频率模型 Ridge OOF 0.660 > 乘法 ALS 0.643**，支持标签在 **概率尺度加性** 上生成：

\[
p_i=\mathrm{clip}\Big(
 a_{s(i)}+f_{s(i)}(\mathrm{days}_i)+g_{s(i)}(r_i)+h_{r(i)}+c\cdot\mathbf{1}[\mathrm{age}\ge 8]
 + d^\top w_i + \varepsilon(\mathrm{x20},V)
,\,0,1\Big)
\]

\[
y_i\sim\mathrm{Bernoulli}(p_i)
\]

其中 \(s=\mathrm{source}\)，\(r=\mathrm{rank}(\mathrm{condition}\mid s)\)（折内相对训练折经验分布），\(w\) 为分段 days 窗口。

**乘法版** \(p=a_s\cdot f(\mathrm{days})\cdot g_s(r)+h_{\mathrm{region}}+\cdots\) ALS-RMSE 只有 **0.6428**，明显更差。不要往 GLM-logit / log-link 频率模型上靠。

小 MLP（source/region one-hot + days/condition/age）诚实 OOF **0.6419**，**达不到 0.70**，不能把网拆成“已够用的嵌入”。Bayes 上限更像 CatBoost 在同类字段上的 **0.669（仅 5 列）～0.692（10 折×3bag×1seed）**。第 1 名 0.749 仍高于我们能用的灵活模型，优先解释为更强的类别有序编码 + 多种子，而不是 MLP 漏掉的光滑流形。

---

## 1. 分 source 的 \(P(y\mid\mathrm{days},\mathrm{condition})\)

10 折，小车型 \(n<120\) 回退到全局。全局 OOF：

| 模型 | OOF AUC |
|---|---:|
| 多项式 Ridge（days/cond/rk 基 + 交互） | **0.65323** |
| 倒 U：\(f(\mathrm{days})+(r-0.5)^2\) Ridge | 0.64655 |
| isotonic(days)+isotonic(−cond) 可加 | 0.65030 |
| isotonic(days)+isotonic(\((r-0.5)^2\)) | 0.60226 |
| 分段树 depth=3 | 0.63559 |
| KNN80（days, log cond） | 0.63281 |
| 0.45 poly + 0.35 knn + 0.20 iso_u | 0.65042 |

分车型 holdout（OOF 切片）：

| source | n | 正例率 | poly | knn80 | iso_u | 形状结论 |
|---|---:|---:|---:|---:|---:|---|
| CAR_1\|ENG_591 | 3709 | 0.092 | **0.679** | 0.665 | 0.593 | 单调保护；iso_add 0.680 |
| CAR_0\|ENG_709 | 3602 | 0.103 | **0.644** | 0.601 | 0.591 | 弱 U（U=0.546>mono 0.534） |
| CAR_2\|ENG_262 | 3586 | 0.109 | **0.630** | 0.591 | 0.577 | 偏 days |
| CAR_5\|ENG_062 | 1024 | 0.080 | **0.704** | 0.700 | 0.636 | days 主导，b≈0.1 |
| CAR_10\|ENG_651 | 288 | 0.156 | 0.670 | **0.708** | 0.586 | **倒 U**；knn 最强 |
| CAR_7\|ENG_966 | 313 | 0.102 | **0.663** | 0.631 | 0.479 | 最差车况反而低风险 |
| CAR_3/4/6/8/9 | 较小 | — | 0.51–0.64 | 0.50–0.64 | 不稳 | CAR_4 poly 崩到 0.509，必须正则/回退 |

**Spark**：按 source 广播一组 Ridge 系数，或对大车型（CAR_0/1/2/5/10）存 2D 查找表。

---

## 2. 2D 直方图 \(P=\mathrm{smooth\ mean}(y\mid s, \mathrm{days\_bin}, \mathrm{cond\_bin})\)

搜索 qcut 分箱 5/8/10/15 × 平滑 m∈{5,10,20,40,80}，以及等宽 days。

| 配置 | 诚实 OOF |
|---|---:|
| **qcut days=5, cond=10, m=10** | **0.64403** |
| qcut d5 c10 m20 | 0.64393 |
| 已知的 d10×c10×source m≈15 | 0.632–0.633 |
| 等宽 days10 × qcut cond10 m40 | 0.63750 |
| 无 source 的 d10×c10 | 0.61784 |
| source×region×d8×c8 m40 | 0.60282（过稀） |

结论：days 只要 **5 箱**（光滑单调 + 几个坑），condition 要 **10 箱**（形状随车型变、倒 U 需要分辨率）。再细（15×15）掉到 0.59。

Nadaraya–Watson 分 source（hd=800, hc=0.45 on log cond）：**0.64722**，略高于最优直方图。Spark 用 per-source **16×10** 分位数网格 + 8 邻域平滑代替核。

---

## 3. 显式频率模型（RMSE，禁止 logloss）

\[
p=\mathrm{clip}(a_s f(\mathrm{days}) g_s(r)+b_{\mathrm{region}}+c\cdot\mathrm{age}+d^\top w,0,1)
\]
ALS 10 折 OOF **0.64276**。

\[
p=\mathrm{clip}(f_s(\mathrm{days})+g_s(r)+h(\mathrm{region})+\cdots,0,1)
\]
Ridge（source 截距 + source×{days, log days, r, (r−0.5)², days/√cond, rate, 窗口} + region one-hot）OOF **0.65914**；α=2 的 LPM **0.65960**。

再加 **折内最优 \(b_s\)** 的 \(\mathrm{days}/\mathrm{cond}^{b_s}\)、CAR_10 U 形、CAR_1 单调、大车型×region：OOF **0.66240**。线性张成在 ~0.66 饱和。

折内稳定的 \(b_s\)（days/cond^b 最大双向 AUC）：

| source | \(b_s\) |
|---|---|
| CAR_1 | 1.5 |
| CAR_10 | 0.9–1.5 |
| CAR_0 | 0.3 |
| CAR_2 | 0.3–0.5 |
| CAR_5, CAR_7, CAR_9 | 0.1 |
| CAR_4, CAR_6 | 0.7–0.9 |

---

## 4. 规则挖掘（一半发现、一半确认）

**确认的 days 低风险坑**（两半同向且确认集 p 很小）：

- **[700, 880]**（及对齐的 [699,849)、[704,784) 等）：发现集 2.3%–3.4%，确认集 **2.2%–3.3%**，p ~ 1e-6–1e-8。n 约 500–1100。全数据 [700,880] n=1109 率 **0.0307**。
- 高风险窄窗 **[9374, 9474)**：发现 18.4%（n=141），确认 **19.4%**（n=93）。全数据 n=249 率 **0.1888**。

**[1750,1800) 全零**：发现 n=25、确认 n=28 均为 0，但 n 不够，p=0.17/0.11，**不能单凭 50 宽窗宣称**。放宽到 **[1725,1825)**：发现 n=46 全零 p=0.013，确认 n=57 全零 **p=0.006**。全 train **n=103 全零**。exp4 原扫描因发现集 p=0.013 略高于 0.01 被丢掉——这是真实坑，特征里保留 `I(days∈[1725,1825])`。

**确认的 condition 阈值**

- CAR_1, condition ≤ 0.081：发现 20.8%（n=379），确认 **19.0%**（n=352）。好车况 ≥0.345：5.5%→**4.9%**。
- CAR_10, condition ≥ 1.15：低风险（2.5%→6.5%，同向）。
- 2D：days<100 且 condition > q10：发现 1.2%（n=81），确认 **3.8%**（n=80）——新车+非最差车况低风险成立。

规则分数单独 OOF 仅 **0.532**（覆盖窄）。必须作为指示特征进频率模型/GBT，不能当主模型。

---

## 5. Bayes 上限（MLP / KNN / HGB / 核心 CatBoost）

| 模型 | 诚实 10 折 OOF |
|---|---:|
| MLP(64,32) RMSE + source/region OH | 0.64187 |
| MLP(32) tanh | 0.63362 |
| KNN70 | 0.64700 |
| HGB squared + 低基数类别 | 0.65691 |
| **CatBoost RMSE 仅 days, condition, age, source, region** | **0.66906** |

MLP **< 0.70**，不拆网。剩余信号主要在 **中基数交叉类别的有序统计**（CatBoost 护城河），不是更深的 days–condition 光滑面（核/直方图已到 0.644–0.647）。

---

## 6. 残差第三臂

isotonic(ratio) OOF **0.6128**，isotonic(rate) **0.6164**。  
之后残差与特征的 |Spearman|（10 折均值）：

| 特征 | \|ρ\| |
|---|---:|
| **x20** | **0.134** |
| **V** | **0.123** |
| cc | 0.105 |
| x1 / x5 | 0.094 / 0.091 |
| age | 0.082 |
| region 组均值 | 0.012 |
| livability | 0.010 |

第三臂 = iso(ratio) + 残差 HGB(region/age/x20/V/…)：**0.6496**。  
x20、V 必须进第三臂；livability 可丢（≈region）。region 的边际 AUC 0.54 大部分已被 days/condition 路径吃掉，交叉 `region×days_q` 仍有独立 TE（约 0.604）。

---

## 7. 幂次网格（无 PySR，手工 (a,b)）

原始双向 AUC（单调变换，无拟合）：

| 形式 | 最优 | 双向 AUC |
|---|---|---:|
| \(\mathrm{days}^a/\mathrm{cond}^b\) | a=1.5, b=0.7 | 0.6121 |
| \(\mathrm{days}^a/\mathrm{cond\_r}^b\) | a=0.9, b=0.7 | 0.6217 |
| \(\mathrm{days}^a (1-r)^c\) | **a=1.5, c=1.227** | **0.6227** |

5 折 isotonic OOF：best rank 式 **0.6175**，rate **0.6172**，ratio **0.6143**，\(\mathrm{days}/\sqrt{\mathrm{cond}}\) 0.6074。  
全局最优仍接近 b≈0.5 的一族（网格里 a/b 沿 \(b\approx 0.5 a\) 几乎一样），但 **分 source 的 b 差一个数量级**，必须做成 `pow_ratio_s = days / cond^{b_s}`。

---

## 8. 可移植模型与融合（不依赖 CatBoost 原生类别即可部署的部分）

Spark 可实现：数值、分箱、折内 TE 分数、GBT。高基数 `source|cond_q10|days_q10` **只做独立分数，禁止喂树**（历史喂 LGB 会塌到 ~0.50）。

| 臂 | 诚实 OOF | Spark |
|---|---:|---|
| 加法 Ridge 频率模型 | 0.659–0.662 | GLM / LinearRegression squared |
| 有序 TE（4 置换 expanding mean）+ Ridge | 0.6664 | 按键累计 sum/count，多置换平均 |
| TE Ridge（LOO/折外，16 键） | 0.6635 | KFold TE |
| HGB/LGB 数值 + **中基数**类别（source, region, src\|cond_q, src\|days_q5, …） | HGB 0.660；LGB 双臂 0.679* | GBTRegressor；类别先整数编码或 TE |
| 2D 查找表 days_q5×cond_q10×source m=10 | 0.6440 | join 统计表 |
| CatBoost 10 折×3bag×1seed 双世界 RMSE | main 0.6900 / alt 0.6891 / w62 **0.6911** / max2 **0.6921** | 用中基数交叉 TE + GBT 逼近；或 CatBoostSpark |
| max(rank(CB main), rank(CB alt)) ⊕ 0.15 LGB | **0.69357** | rank 加权 |

\*LGB 0.679 用了 **外折 early stopping**，略乐观；CatBoost teacher 用训练折内 12% ES，更诚实。

无 CatBoost 原生类别的便携融合（HGB+freq+TE）**0.6690**，与“仅 5 列 CatBoost 0.669”同一水平。要从 0.67 到 0.70，需要 **有序 boosting / 多种子 bagging**，不是再堆原始列。

---

## 可移植特征列表（Spark）

折内统计：source 的 condition 中位数、分位数切点、TE 的 sum/count。丢掉 x18、x19、id。

### 数值（确定性，broadcast 中位数）

1. `days`, `log1p(days)`, `sqrt(days)`
2. `condition_f` = fill(source median → global median)
3. `cond_r = condition_f / median(condition_f|source)`
4. `cond_rk = rank_pct(condition_f | source)`（val 对训练折 searchsorted）
5. `ratio = days/cond_r`，`rate = days*(1-cond_rk)`
6. `ratio_sqrt = days/sqrt(condition_f)`，`log_ratio = log(days)-0.5*log(condition_f)`
7. **`pow_ratio_s = days / condition_f^{b_s}`**，\(b_s\) 上表，折内可重选
8. **`pow_rate15 = days^{1.5} * (1-cond_rk)^{1.23}`**
9. `u_shape=(cond_rk-0.5)^2`
10. `ushape_car10 = u_shape * I(CAR_10)`，`mono_car1=(1-cond_rk)*I(CAR_1)`，`rev_car7=cond_rk*I(CAR_7)`
11. 窗口：`I(days<50)`, `I(days∈[700,880])`, **`I(days∈[1725,1825])`**, `I(days∈[1950,2000])`, **`I(days∈[9370,9475])`**
12. `I(age_range>=8)`, `I(condition_f<0.05)`, `cond_miss`
13. 弱连续：`x20`, `V`, `cc`, `x1`, `x5`, `x14`, `x17`, `t3_num`（解析数字）

### 分箱（训练折 qcut，val 用同一 edges）

- `days_q5`（不要只用 q10；2D 最优是 5）
- `cond_q10`, `ratio_q10`, `rate_q10`

### 独立 TE 分数（m 平滑，只 rank 融合或进 Ridge，不进 GBT 作为稀疏高基数列）

- `source|cond_q10|days_q5`（优于 d10×c10），m=10
- `cond_q10|days_q5`, `source|ratio_q10`, `source|cond_q10`
- `region|days_q`, `region|age`, `source|region`

### 中基数类别 → 整数编码进 GBT / 或折内 TE 进 GLM

`source, region, age_range, source|region, source|age, region|age, source|cond_q10, source|days_q5`  
**不要**把 `source|cond_q|days_q`（~1100 水平）当作树的类别列。

### 模型

- 臂 Freq：Ridge/LinearRegression squared on 展开基（§3）
- 臂 GBT：Spark `GBTRegressor` loss squared，depth 5 / 6，两套特征（cond_r 世界 vs rate 世界）
- 臂 TE：独立分数 rank
- 融合：`0.62*rank(main)+0.38*rank(alt)`；再与 LGB/Ridge 小权重 rank 混合  
- 大车型 CAR_1/5/10 可加专模残差（knn/poly），权重要小

---

## 当前最佳 OOF 与配方

文件：`/workspace/analysis/reverse/best_oof.npy`

**截至多种子 gen-CatBoost 完成前：**

- CatBoost teacher：10-fold × 3-bag × 1-seed，RMSE，中基数类别（无 src_cq_dq 原生列），inner-ES  
  max2 = **0.69207**，w62 = **0.69107**
- 融合 `0.85 * max(rank(main),rank(alt)) + 0.15 * rank(LGB gen-features)` = **0.69357**
- **保司硬门**（冻结，不搜参）：teacher max2 0.69207 → **0.69310–0.69325**。见 `analysis/ADVERSARIAL.md`。

要稳定跨过 **0.70**（对齐历史 W62 的 8seed×3bag），需要把同一套生成过程特征做 **多种子 Ordered+Plain bagging**，而不是再加原始列。脚本：`analysis/reverse/exp8b_gen.py`（pow_ratio、days_q5、freq_score、确认窗口）。

实验 JSON：`analysis/reverse/exp{1-8}*.json`。

---

## 9. genfunc 闭合公式（新增，2026-08-13）

目录：`analysis/genfunc/`。协议不变：StratifiedKFold、折内 fit、身份 RMSE。详表见 `analysis/genfunc/FORMULA.md`。

**便携诚实 10 折 OOF ≥ 0.68：**

\[
s=0.40\,\mathrm{rank}(\mathrm{GBT}_{\mathrm{RMSE},400})+0.30\,\mathrm{rank}(\mathrm{TE}_{\mathrm{ord}})+0.15\,\mathrm{rank}(T_{8})+0.15\,\mathrm{rank}(f+g)=\mathbf{0.6822}
\]

权重预置（非 10 折上搜索）。5 折冻权同栈 0.6817。Spark：`GBTRegressor` squared、固定 400 棵、禁止 ES；`src|cq|dq` 只做表/TE，不进树。

无树闭合 rank（ordered TE 0.45 + 8 邻域表 0.25 + 样条 0.20 + 幂次 0.10）= **0.6705**（`portable_score.py` 10 折复测）。

相对本文 §2 的 2D 表 0.644：8 邻域 + 全局层次 \(\alpha=0.40\) 把 **source×days_q5×cond_q10** 提到 **0.6577**。LOSO 确认 8 邻域优于无平滑，但跨车型全局面只有 ~0.59，分源表仍必要。

\(g\) 建在 `rank(condition|source)` 上优于原始 condition（5 折 0.6531 vs 0.6505）。CAR_10 倒 U、CAR_1 单调保护、CAR_7 最差车况低风险：标准化系数符号全部符合假说。幂次网格 24 组 5 折 AUC>0.62，最优 \(a=1.5,c=0.5\)，\(b_s\) 见 `exp4_grid_closed.json`。

**不要**再把样条+表+TE 拼成一个大 Ridge（joint 0.623）。分臂再 rank。
