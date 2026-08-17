# -*- coding: utf-8 -*-
"""
bootstrap_paired.py
配对分层 bootstrap 显著性检验（参考对方 3000 次配对分层 bootstrap 协议）。

用途：判断实验增益（新 OOF vs 基准 OOF）是"真增益"还是"抽样噪声"。
规则（v2.1 决策红线）：
  - 增益 bootstrap 95% 区间下限 > 0 或正比例 > 90% 才视为"接受"级证据
  - 区间包含 0 → 效果与噪声不可区分，不纳入 submission

用法（作为模块导入）：
    from bootstrap_paired import paired_bootstrap
    res = paired_bootstrap(y, oof_base, oof_new, n_boot=3000, seed=42)

也可命令行运行快速演示：
    python bootstrap_paired.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def _stratified_bootstrap_index(y, rng):
    """按正负标签分层有放回采样，返回采样索引。"""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    idx = np.concatenate([
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
    ])
    return idx


def paired_bootstrap(y, oof_a, oof_b, n_boot=3000, seed=42):
    """
    配对分层 bootstrap：比较 oof_b 相对 oof_a 的 AUC 增益。

    参数:
        y       : (n,) 真实标签
        oof_a   : (n,) 基准模型 OOF 概率
        oof_b   : (n,) 新模型 OOF 概率
        n_boot  : bootstrap 次数（默认 3000）
        seed    : 随机种子（可复现）

    返回:
        dict: full_delta(全量OOF增益), mean, ci95, p_pos,
              verdict('accept'/'ambiguous'/'reject')
    """
    y = np.asarray(y)
    oof_a = np.asarray(oof_a)
    oof_b = np.asarray(oof_b)
    rng = np.random.RandomState(seed)

    auc_a_full = roc_auc_score(y, oof_a)
    auc_b_full = roc_auc_score(y, oof_b)
    full_delta = auc_b_full - auc_a_full

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stratified_bootstrap_index(y, rng)
        deltas[i] = roc_auc_score(y[idx], oof_b[idx]) - roc_auc_score(y[idx], oof_a[idx])

    mean = float(deltas.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_pos = float((deltas > 0).mean())

    # 决策：下限>0 或 正比例>90% → accept；上限<0 或正比例<10% → reject；否则 ambiguous
    if lo > 0 or p_pos > 0.90:
        verdict = 'accept'
    elif hi < 0 or p_pos < 0.10:
        verdict = 'reject'
    else:
        verdict = 'ambiguous'

    return {
        'auc_a': float(auc_a_full),
        'auc_b': float(auc_b_full),
        'full_delta': float(full_delta),
        'boot_mean': mean,
        'ci95': [float(lo), float(hi)],
        'p_pos': p_pos,
        'verdict': verdict,
        'n_boot': n_boot,
    }


if __name__ == '__main__':
    # 演示：人造数据（B 略优于 A）
    rng = np.random.RandomState(0)
    y = np.r_[np.zeros(1800), np.ones(200)]
    y = y[rng.permutation(len(y))]
    a = np.clip(y * 0.6 + rng.randn(len(y)) * 0.5, 0.01, 0.99)
    b = np.clip(y * 0.65 + rng.randn(len(y)) * 0.5, 0.01, 0.99)
    res = paired_bootstrap(y, a, b, n_boot=3000, seed=42)
    for k, v in res.items():
        print(f'{k}: {v}')
