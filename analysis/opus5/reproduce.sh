#!/usr/bin/env bash
# 一键复现 0.69993 方案(完整,从数据重跑,~150 min)
# 用法: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "================================================"
echo "  复现 0.69993 方案: max2(merger_ord8 + v2_cat_alt8)"
echo "  预计耗时: ~150 min (8 seeds × 5 fold × 2 arms)"
echo "================================================"

# 臂1: merger_ord8 (v2 主帧 FE + Ordered boosting, depth=5, 8 seeds)
echo ""
echo "[1/3] 训练 merger_ord8 (4 chunks × 2 seeds)..."
for c in 0 1 2 3; do
    echo "  --- chunk $c (seeds $((2026+2*c)),$((2027+2*c))) ---"
    python3 src/v2fe_ord_chunk.py "$c"
done
echo "  --- combine 4 chunks -> merger_ord8 ---"
python3 src/combine_chunks.py merger_ord8 v2fe_ord_c0 v2fe_ord_c1 v2fe_ord_c2 v2fe_ord_c3

# 臂2: v2_cat_alt8 (v2 alt 编码世界, depth=6, 8 seeds)
echo ""
echo "[2/3] 训练 v2_cat_alt8 (4 chunks × 2 seeds)..."
for c in 0 1 2 3; do
    echo "  --- chunk $c (seeds $((2026+2*c)),$((2027+2*c))) ---"
    python3 src/v2_cat_alt_chunk.py "$c"
done
echo "  --- combine 4 chunks -> v2_cat_alt8 ---"
python3 src/combine_chunks.py v2_cat_alt8 v2_cat_alt_c0 v2_cat_alt_c1 v2_cat_alt_c2 v2_cat_alt_c3

# 融合
echo ""
echo "[3/3] max2 融合 -> submission_v4_honest.csv"
python3 src/fuse_v4b.py

echo ""
echo "================================================"
echo "  完成。预期 nested OOF = 0.69993"
echo "  提交文件: submissions/submission_v4_honest.csv"
echo "================================================"
