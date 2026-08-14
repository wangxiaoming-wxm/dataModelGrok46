# HONEST 10 折主臂

同一套 opus5 FE + Classifier Logloss + 固定 800 树、无早停，5 折改成 10 折。

## 8-seed 结果

| 指标 | bag0 | bag0+bag1 |
|---|---:|---:|
| max2 ungated | 0.70258 | **0.70295** |
| max2 nested | 0.70231 | **0.70268** |
| gauss⊕ref30 + 硬门 | 0.70742 | **0.70774** |
| vs 历史 W62 0.70159 | +0.00099 | **+0.00136** |

当前提交走 2 袋 pool：`python3 -u analysis/opus5/eval_bag_pool.py --write`。

## 断点续跑（按 seed，不是按折）

`analysis/opus5/ckpt_honest10/{main,alt}_f10_s{seed}.npz` 存在则跳过。
第二袋文件名 `{arm}_f10_s{seed}_b1.npz`。

```bash
python3 -u analysis/opus5/train_honest10.py --run
python3 -u analysis/opus5/train_honest10.py --run --bag 1
python3 -u analysis/opus5/eval_bag_pool.py
```
