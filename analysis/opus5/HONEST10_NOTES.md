# HONEST 10 折主臂

目标：在 **同一套 opus5 FE + Classifier Logloss + 固定 800 树、无早停** 上，只把 5 折改成 10 折，冲击历史 W62 诚实 OOF **0.70159**。

## 为什么是这条臂

- 现有 5 折 8-seed max2：**full 0.70023 / nested 0.69993**，差 W62 约 0.0014。
- 历史 W62 用的是 **10 折 × 8 seed × 3 bag**；opus5 是 5 折 × 8 seed × 1 bag。
- 同 seed 加权融合（0.62/0.38）在 Classifier 这对臂上 **不如 max2**（0.6987 vs 0.6999），所以融合规则保持 max2。
- 本机 RMSE 复现 W62 特征只到 0.688，缺口在表示不在折数；opus5 表示已经够强，缺的是 10 折协议。

## 第一个 seed（2026）对照

| 臂 | 5 折（opus5） | 10 折 | Δ |
|---|---:|---:|---:|
| main Ordered d5 | 0.690039 | **0.691485** | **+0.00145** |
| alt Plain d6 | 0.688584 | **0.690431** | **+0.00185** |
| 1-seed max2 | — | 0.69453 | 不能和 8-seed 0.70023 比 |

单 seed 抬升幅度已经覆盖 W62 缺口。8-seed rank pool 仍在跑。

## 入口

```bash
python3 -u analysis/opus5/train_honest10.py --run
python3 -u analysis/opus5/train_honest10.py --report
```

断点：`analysis/opus5/ckpt_honest10/`（不要提交）。满 8 seed 之前 **不改 submission**。
