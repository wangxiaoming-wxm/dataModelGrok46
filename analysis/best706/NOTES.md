# best_model_0.706 借鉴笔记

zip 自称 0.706 是 **线上公开榜 0.70599**，不是本地 OOF。本地 v14 OOF 只有 **0.69427**（硬门后 0.69680），弱于 opus5 nested 0.69993 / gated 0.70274。

## 可借鉴

- **Gaussian copula 融合**：`norm.ppf(clip(rank))` 后再加权。用在我们已有的 opus+LGB 权重上，gated 从 0.70335 → **0.70398**（nested 0.70056 → 0.70128）。不采用他们的弱臂权重。
- 伪标签已在该包中被证伪（fold-exclusive −0.0009）。继续禁止。
- seed 筛选有收益，但我们已有 8-seed honest 臂。

## 不要搬

- **ref 迁移**：用另一份 21328 行 `train1` 做特征匹配灌进本地 test。这是外部数据/另一赛道 dump，本地 OOF 只有 0.694，线上虚高来自 ref 权重 0.42 的榜放大，不是更强生成过程。
- `ref_pipeline.add_id_features()`：6 位 id 十六进制字符，违反本赛 id 禁令。
- 他们的 cat_opt5/XGB/ET 单臂 0.61–0.69，与 opus Spearman ≥0.86，加权几乎不涨。
- 跨协议 OOF 逐元素 max。

脚本：`submit_sprint4_v14.py`、`ref_pipeline.py`。不要把该包的 `submission.csv` 当更强本地答案。
