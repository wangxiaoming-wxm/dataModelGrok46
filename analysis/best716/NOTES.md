# 0.71629 强化

上游 `best_0.716`：`0.70·rank(W62) + 0.30·rank(ref)`，线上 **0.71629**。
W62 是冻结 RMSE 双臂（线上 0.71503）。ref 来自旧官方切分多出来的有标签行（灰区）。

## 这一版做了什么

把底座从 W62（本地 0.70153）换成 **honest10**（Classifier 10 折×8 seed）。
ref 权重 **冻结 0.30**，不重搜。再套已冻结的 Gaussian copula 和四窗硬门。
第二袋齐套后改为 **bag0+bag1 rank-pool**（同一 SKF，不同树种子/jitter）。

`gauss(0.70·honest10_bag01 + 0.30·ref) + gate`

| 包 | ungated | nested | gated |
|---|---:|---:|---:|
| 上游 W62⊕ref30 | 0.70256 | 0.70242 | 0.70496 |
| honest10 + 硬门 | 0.70258 | 0.70231 | 0.70493 |
| gauss h10 bag0 ⊕ref30 + 硬门 | 0.70512 | 0.70483 | 0.70742 |
| **本交 gauss h10 bag0+bag1 ⊕ref30 + 硬门** | **0.70538** | **0.70508** | **0.70774** |

配对 bootstrap vs bag0 gauss+gate（2000）：Δ=+0.00032，CI [-0.00027, +0.00088]，p_pos=0.865（含 0）。
bag1 单独 gated 0.70758，不弱于 bag0，故 2 袋是降方差而不是稀释。

## 入口

```bash
python3 -u analysis/opus5/eval_bag_pool.py --write
python3 -u analysis/best716/assemble_h10_ref.py   # 读 artifacts/honest10_max2.npz（现为 2bag pool）
python3 -u analysis/opus5/train_honest10.py --run --bag 1
```
