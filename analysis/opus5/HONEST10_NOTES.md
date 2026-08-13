# HONEST 10 折主臂

目标：在 **同一套 opus5 FE + Classifier Logloss + 固定 800 树、无早停** 上，只把 5 折改成 10 折，冲击历史 W62 诚实 OOF **0.70159**。

## 进度（2026-08-13 14:09）

| 状态 | 值 |
|---|---|
| 已完成种子 | **7/8**（2026–2032，main+alt 均已落盘） |
| 进行中 | seed **2033** main，约 8/10 折 |
| 7-seed max2 | **full 0.70240 / nested 0.70218**（已超过 W62 0.70159） |
| 7-seed main pool | 0.69771 |
| 7-seed alt pool | 0.69858 |

## 断点续跑（按 seed，不是按折）

`train_honest10.py` 看到 `analysis/opus5/ckpt_honest10/{main,alt}_f10_s{seed}.npz` 就 **跳过该 (臂, 种子)**。

- **已完成的 seed 安全**：杀进程 / 重启后不会重跑 2026–2032。
- **当前这个 seed 不安全**：2033 还没写成 npz。现在停的话，main 已跑的折会丢掉，重启后 **整颗 2033 重来**（main ~14 min + alt ~7 min）。
- **没有折级断点**。不要指望从 f7 接着训。

重启后续跑：

```bash
python3 -u analysis/opus5/train_honest10.py --run
# 会打印 [main s2026] skip existing ... 然后只训缺的种子
python3 -u analysis/opus5/train_honest10.py --report
```

满 8 seed 后写提交：

```bash
python3 -u analysis/opus5/submit_honest10.py
```

## 同 seed 5 折 vs 10 折（2026）

| 臂 | 5 折 | 10 折 | Δ |
|---|---:|---:|---:|
| main Ordered d5 | 0.690039 | **0.691485** | **+0.00145** |
| alt Plain d6 | 0.688584 | **0.690431** | **+0.00185** |
