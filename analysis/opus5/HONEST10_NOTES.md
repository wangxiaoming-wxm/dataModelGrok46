# HONEST 10 折主臂

同一套 opus5 FE + Classifier Logloss + 固定 800 树、无早停，5 折改成 10 折。

## 8-seed 结果（已交）

| 指标 | 值 |
|---|---:|
| max2 ungated | **0.70258** |
| max2 nested | **0.70231** |
| max2 + 四窗硬门 | **0.70493** |
| vs 历史 W62 0.70159 | **+0.00099**（ungated） |
| vs opus5 5 折 0.70023 | **+0.00235** |
| main pool | 0.69784 |
| alt pool | 0.69875 |

提交：`submissions/submission.csv`（`python3 -u analysis/opus5/submit_honest10.py`）。

## 断点续跑（按 seed，不是按折）

`analysis/opus5/ckpt_honest10/{main,alt}_f10_s{seed}.npz` 存在则跳过。

```bash
python3 -u analysis/opus5/train_honest10.py --run
python3 -u analysis/opus5/submit_honest10.py
```
