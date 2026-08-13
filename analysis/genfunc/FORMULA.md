# 闭合生成函数（便携、诚实 OOF）

# 便携诚实 10 折 OOF = 0.6822 ≥ 0.68

**口径**：`StratifiedKFold(n=10, seed=2026)`，折内 fit。RMSE / 身份链接。无全量 TE、无 test 标签、无 id。树臂 **固定 400 轮、禁止 early stopping**。

**公式（Spark 可实现）**：对每个折外样本算四臂，再在打分批次上 `percent_rank` 加权：

\[
s=0.40\,\mathrm{rank}(\mathrm{GBT}_1)+0.30\,\mathrm{rank}(\mathrm{TE}_{\mathrm{ord}})+0.15\,\mathrm{rank}(T_{8})+0.15\,\mathrm{rank}(f+g)
\]

该权重是代码里预先写死的（不是 10 折上搜出来的）。5 折筛出的另一组权重 10 折为 **0.6817**。

无树的闭合身份公式 10 折 **0.6705**（`portable_score.py` 复测；exp6 冻权 0.6696）。第 1 名 0.749 仍高于此；剩余差距主要在多种子有序 boosting，不是再加原始列。

---

## 0. 生成过程（概率尺度，不是 logit）

\[
p=\mathrm{clip}\big(
 a_s+f_s(\mathrm{days})+g_s(r)
 +u_{10}\mathbf{1}_{\mathrm{CAR\_10}}(r-0.5)^2
 +m_1\mathbf{1}_{\mathrm{CAR\_1}}(1-r)
 +q_7\mathbf{1}_{\mathrm{CAR\_7}}r
 +h_{\mathrm{region}}+c\cdot\mathbf{1}[\mathrm{age}\ge 8]+d^\top w
 +\lambda\,T(s,\mathrm{days\_q5},\mathrm{cond\_q10})
 +\beta\,\mathrm{days}^{1.5}/\mathrm{cond}^{b_s}(1-r)^{0.5}
,\,0,1\big)
\]

\[
y\sim\mathrm{Bernoulli}(p),\qquad r=\mathrm{rank}(\mathrm{condition}\mid s)
\]

\(r\) 必须用**训练折** condition 的 `searchsorted`，禁止用全量 / 验证折自己的 rank。

客户侧：days 越大暴露越长；\(r\) 低（车况差）更愿修车走保险；新车+好车况几乎不报；`age>=8` 右尾；地区异质进 \(h\)。

保司侧：\(a_s,f_s,g_s,b_s\) 按 source 核保分层；窗口 \([700,880]\)、\([1725,1825)\) 少赔；CAR_1 单调保护；CAR_10 倒 U；CAR_7 最差车况反而低风险。

---

## 1. 诚实 10 折一览

| 臂 / 融合 | 10 折 OOF | Spark |
|---|---:|---|
| **GBT + TE + 表 + 样条 rank（预置权）** | **0.6822** | GBTRegressor + 表 + Ridge + percent_rank |
| 5 折冻权 GBT 栈 | 0.6817 | 同上 |
| 闭合身份 rank（TE+表+样条+幂） | **0.6705** | 无树，`portable_score.py` |
| 等权 rank(样条, 表, TE_ord) | 0.6680 | 无树 |
| TE Ridge（14 键，LOO） | 0.6665 | 折内 sum/count |
| Ordered TE Ridge | 0.6662 | 随机序 expanding mean（训练折） |
| TE + 8 邻域表 一列 | 0.6676 | 上两项拼接 |
| 8 邻域 2D 表 d5×c10 m=10 α=0.4 | **0.6577** | 原 0.6440 +0.014 |
| 分源 8 段 days + 6 段 rank 样条 | 0.6554 | `greatest(x-k,0)` |
| + CAR_10/1/7 可加项 | 0.6556 | 同上 |
| 幂次闭合 a=1.5 c=0.5 + region + 窗口 | 0.6422 | 点积 |
| Nadaraya–Watson 核 | 0.6490 | 可选，Spark 用表代替 |
| 旧 joint 大 Ridge（勿用） | 0.623 | 共线，已弃 |

5 折筛、10 折报。LGB/GBT **没有**用验证折 early stopping。

---

## 2. 无树闭合公式（`portable_score.py`）

```
p_hat = rank_fuse({
    te_ord:  0.45,   # 14 键 ordered/LOO TE → Ridge α=18
    table8:  0.25,   # source × days_q5 × cond_q10，8 邻域，α_glob=0.40
    spline:  0.20,   # a_s + f_s(days) + g_s(r) + CAR 项 + region + 窗口
    power:   0.10,   # days^1.5 / cond^{b_s} * (1-r)^0.5 + region + 窗口
})
```

诚实 10 折 **0.6705**（5 折 0.6684）。实现：`fit_fold_stats` / `score`（纯 numpy/pandas）。`score` 在**当前批次**上做 `rank(pct=True)`，对应 Spark `percent_rank()`，这是 **AUC 用的序**，均值约 0.5。定价概率用 `score_prob`：同一权重对四臂 \(p\) 做线性平均再 `clip(0,1)`（exp6 线性融合 10 折 0.6694）。

### 2.1 样条 \(f_s,g_s\)

- days：训练折 8 分位结点，基 \(\{1,\,x,\,(x-k_j)_+\}\)，\(x=\mathrm{days}/5000\)。
- \(g\)：在 **rank** 上 6 段（5 折筛：rank 0.6531 > 原始 condition 0.6505）。
- 大车型 \(n_{\mathrm{tr}}\ge 200\)：再加 `I(s)·基[:,1:]` 偏差。
- Ridge α=20，折内标准化。

### 2.2 车型可加项（标准化后 10 折均系数，α=20）

| 项 | 系数 | 含义 |
|---|---:|---|
| `I(CAR_10)(r-0.5)^2` | −0.0235 | 倒 U：中间高风险 |
| `I(CAR_10) r^2` | −0.0204 | 同上 |
| `I(CAR_1)(1-r)` | +0.0346 | 车况差 → 索赔↑（单调保护） |
| `I(CAR_7) r` | +0.0268 | 最差车况（低 r）反而低风险 |
| `I(CAR_0)(r-0.5)^2` | −0.0106 | 弱 U |
| `I(CAR_2)(r-0.5)^2` | −0.0112 | 弱 U |

符号与业务假说一致。单独加入样条后 10 折 0.65521 → 0.65556，增量小，但方向稳定，保留。

### 2.3 2D 表 \(T\)（8 邻域）

单元格 \((s,i,j)\)：

\[
T=\frac{w_0 S_{ij}+\sum_{\mathrm{side}} w_s S+\sum_{\mathrm{diag}} w_d S+m\pi}{w_0 n_{ij}+\sum w_s n+\sum w_d n+m}
\]

\(w_0=4,w_s=1,w_d=0.5,m=10\)。再与**全局** days×cond 表混合：\((1-\alpha)T_s+\alpha T_{\mathrm{glob}}\)，\(\alpha=0.40\)。

| 配置 | 10 折 |
|---|---:|
| d5×c10 m=10 无平滑（复现旧 0.644） | 0.64403 |
| d5×c10 m=10 8 邻域 α=0.25 | 0.65766 |
| LOSO 选超参后的 winner（近乎同上） | 0.65760 |
| 核 hd=1200, hc=0.40 | 0.64896 |

**留一源 CV**：全局 days×cond 面在 held-out 车型上最高约 0.594（d8×c10 + 8equal）。说明**共享光滑面存在但不强**；OOF 仍要分源表 + 层次混合。8 邻域对 LOSO 和 OOF 都稳定优于无平滑。

### 2.4 幂次网格（AUC>0.62 的参数）

形式 \(\mathrm{days}^a/\mathrm{cond}^{b_s}(1-r)^c\) + source/region OHE + `I(age>=8)` + 窗口，Ridge 身份。

5 折 AUC>0.62：**24** 组，全部 \(a\ge 1.2\)。最优 \(a=1.5,c=0.5,\alpha=4\)，5 折 0.6404，10 折 **0.6422**。

折内重选的 \(b_s\)（10 折均值）：

| source | \(b_s\) |
|---|---:|
| CAR_1\|ENG_591 | 1.50 |
| CAR_10\|ENG_651 | 1.26 |
| CAR_6\|ENG_331 | 0.98 |
| CAR_4\|ENG_899 | 0.82 |
| CAR_2\|ENG_262 | 0.38 |
| CAR_8\|ENG_843 | 0.36 |
| CAR_0\|ENG_709 | 0.30 |
| CAR_3\|ENG_467 | 0.28 |
| CAR_9\|ENG_811 | 0.16 |
| CAR_5\|ENG_062 | 0.12 |
| CAR_7\|ENG_966 | 0.12 |

CAR_1 / CAR_10 对 condition 敏感（\(b\) 大）；CAR_5/7/9 几乎只看 days。

### 2.5 窗口（加性，不是 logit）

`I(days∈[0,50))`, `[0,200)`, **`[700,880)` 质保几乎不赔**, **`[1725,1825)` 疑似全零**, `[1950,2000)`, `[9370,9475)` 高风险, `I(days≥10000)`。

### 2.6 TE 键（折内；高基 3 键只进 Ridge，不进树）

`src, reg, src_reg, src_age, reg_age, src_cq, src_dq, cq_dq, src_cq_dq, src_rq, src_tq, reg_dq, reg_cq_dq, src_cq_age`

`src_cq_dq` 用 **days_q5 × cond_q10**（不是 q10×q10）。训练折：4 置换 expanding mean；验证折：\((S+m\pi)/(n+m)\)。

---

## 3. Spark 移植（0.682 栈）

1. **折内统计**（broadcast）：source condition 中位数；days/cond/ratio/rate 分位切点；2D `(sum,count)` 立方；14 个 TE 的 `(sum,count)`；样条结点与 Ridge 系数。
2. **样条臂**：`x=days/5000`，`greatest(x-knot,0)`，source/region 指示，CAR 项，窗口，标准化点积。
3. **表臂**：`bucketizer` → 8 邻域公式 → 与全局表 `0.6 T_s + 0.4 T_g`。
4. **TE 臂**：14 个 map-join 平滑均值 → LinearRegression（squared）。训练折可用随机键排序的 expanding mean 近似 Ordered。
5. **GBT 臂**（只此臂用树）：`GBTRegressor` loss squared，**固定 400 棵**，depth 5，minInstances 80，lr≈0.03。特征=生成过程数值（days/log/√、cond、cond_r、r、ratio、rate、pow_ratio、u_shape、CAR 项、窗口、age8、x20、V、cc）+ **中基数整数编码** `source, region, age, src\|cq, src\|dq5, src\|region, src\|age, region\|age, src\|ratio_q`。  
   **禁止**把 `src|cq|dq` 当树的类别列。
6. **融合**：四臂 `percent_rank` 后 `0.40 GBT + 0.30 TE + 0.15 表 + 0.15 样条`。

`portable_score.py` 实现 1–4 与无树融合。GBT 用 Spark 复现，不要在便携脚本里 pickle 树。

---

## 4. 证伪 / 保留

- **乘法 ALS / logit / logloss**：此前更差；本次全部身份 RMSE，不再走。
- **joint 大矩阵 Ridge**：样条+表+TE+x20 拼在一起 → 0.623，共线，已弃。融合必须 **分臂再 rank**。
- **g 建在原始 condition**：不如 rank（5 折 0.6505 vs 0.6531）。
- **2D 再细**：d15 历史掉到 0.59；本次 d8×c10 8 邻域 5 折 0.6576，与 d5×c10 同级。
- **无树封顶 ~0.67**：线性张成 + 查找表 + TE 到 0.6705。过 0.68 需要 GBT/有序 boosting。
- **LGB 600 轮 < 400 轮**（5 折 0.642 vs 0.648）：固定轮数不要过大。

---

## 5. 实验 JSON

| 文件 | 内容 |
|---|---|
| `exp1_spline.json` | days 8 段 + cond/rank 6 段 |
| `exp2_car_shapes.json` | CAR_10 / CAR_1 / CAR_7 |
| `exp3_2d_smooth.json` | LOSO 超参 + 8 邻域 |
| `exp4_grid_closed.json` | 幂次网格，AUC>0.62 的 24 组 |
| `exp5_fuse.json` | 首轮融合（含失败的 joint） |
| `exp6_push.json` | 0.682 栈与闭合 0.670 |
| `portable_score.json` | 便携打分器复测 |
| `portable_params_compact.json` | 结点 / \(b_s\) / 权重（Spark） |
