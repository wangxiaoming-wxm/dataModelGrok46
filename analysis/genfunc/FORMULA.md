# 闭合生成函数（便携、诚实 OOF）

> 结果数字在实验跑完后回填。协议：StratifiedKFold，折内 fit，身份链接 RMSE。禁止全量 TE / test 标签 / id。

若诚实 OOF ≥ 0.68，本节顶部会用大字标明。

---

## 生成过程（概率尺度，不是 logit）

\[
p=\mathrm{clip}\big(
 a_s+f_s(\mathrm{days})+g_s(r)+u_{10}\mathbf{1}_{\mathrm{CAR\_10}}(r-0.5)^2
 +m_1\mathbf{1}_{\mathrm{CAR\_1}}(1-r)+q_7\mathbf{1}_{\mathrm{CAR\_7}}r
 +h_{\mathrm{region}}+c\cdot\mathbf{1}[\mathrm{age}\ge 8]+d^\top w
 +\lambda\,T(s,\mathrm{days\_q5},\mathrm{cond\_q10})
 +\beta\,\mathrm{days}^{a}/\mathrm{cond}^{b_s}(1-r)^{c}
,\,0,1\big)
\]

\[
y\sim\mathrm{Bernoulli}(p),\qquad r=\mathrm{rank}(\mathrm{condition}\mid s)\ \text{（折内 searchsorted）}
\]

- \(f_s\)：days 8 段分段线性（分位结点），大车型有自身偏差。
- \(g_s\)：condition 秩 6 段分段线性。
- \(T\)：source × days_q5 × cond_q10 的 \(m=10\) 平滑均值，8 邻域 + 全局表层次混合。
- 窗口 \(w\)：`[0,50), [0,200), [700,880), [1725,1825), [1950,2000), [9370,9475), [10000,∞)`。

## Spark 移植

见 `portable_score.py`：`fit_fold_stats` / `score`。全部是中位数、分位切点、searchsorted、`greatest(x-k,0)`、map-join 表、点积。

## 实验 JSON

| 文件 | 内容 |
|---|---|
| `exp1_spline.json` | 分源样条 Ridge |
| `exp2_car_shapes.json` | CAR_10 / CAR_1 / CAR_7 可加项 |
| `exp3_2d_smooth.json` | LOSO 超参 + 8 邻域 |
| `exp4_grid_closed.json` | 幂次网格，AUC>0.62 参数 |
| `exp5_fuse.json` | 2D + Ridge rank 融合 |
| `portable_score.json` | 便携打分器诚实 OOF |
