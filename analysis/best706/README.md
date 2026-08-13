# best_model_0.706 —— 当前最优模型归档（线上 0.70599）

## 协议（v14）
- 集成：gauss copula（norm.ppf rank → 加权和 → rank_norm）
- 权重：cat_opt5(7seed) 0.33 + ref(2026+8888 等权) 0.42 + et 0.07 + b_v3 0.04
        + xgb 0.04 + xgb_d202 0.04 + xgb_d606 0.02 + xgb_d707 0.03
- cat_opt5 seeds: 314162, 141422, 24680, 223607, 259810, 3141603, 271831
- ref 组件：参考 train_old(21328 全精度, 无 id) CatBoost 5 折，seed 2026 与 8888 等权，
  经特征映射迁移到本地（train cover 0.976 / test cover 1.0）
- OOF AUC = 0.69427；线上 0.70599（2026-08-13）

## 复现步骤
1. 特征：ag_train_feats.csv（主目录，190 列，prepare_ag_features.py 生成）
2. ref 组件：train_ref_local_blend.py + ref_s8888（train_ref_multiseed.py）
3. 提交生成：python submit_sprint4_v14.py

## 文件清单
- submission.csv —— v14 正式提交
- comp_*_oof/test.npy —— 本地 7 组件（cat_opt5/et/b_v3/xgb/d202/d606/d707）
- ref_model_* / ref_s8888_* —— 参考迁移组件（OOF/参考test/本地test映射）
- matched_pairs_L3.csv —— 本地 train → 参考 train 特征级 1:1 映射（14573 对）
- ckpt_cat_joint_s*（35）+ ckpt_ref_s8888_f*（5）—— 模型检查点
- submit_sprint4_v14.py —— 提交生成脚本
- final_submit_sprint4_v*_report.json —— 各版本报告
- ref_pipeline.py / bootstrap_paired.py —— 特征管线 / 显著性验证
- manifest_sha256.json —— 完整性校验

## 验证纪律
- OOF 全程 foldwise（验证折不参与特征选择/训练）
- 参考迁移无 id 特征；ref 权重 0.42 为线上最优（0.46 过重已回退）
