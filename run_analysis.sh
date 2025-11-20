#!/bin/bash
#SBATCH --job-name=embedding_distribution
#SBATCH --output=%j.embedding_distribution.out
#SBATCH --error=%j.embedding_distribution.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --account=EUHPC_D26_076
#SBATCH --partition=boost_usr_prod

cd /leonardo_work/EUHPC_B24_036/yang
source encoding_env/bin/activate

echo "========================================"
echo "Embedding Distribution Analysis"
echo "========================================"

# 创建结果目录
mkdir -p embedding_distribution/results

# 分析模型1: qwen2.5-7b (基线模型)
echo ""
echo "----------------------------------------"
echo "分析模型 1: qwen2.5-7b"
echo "----------------------------------------"
python embedding_distribution/analyze_embedding_distribution.py \
    --model_path scratch_link/embeddings/qwen2.5-7b_CN-specific-Break_rank_top001 \
    --model_name qwen2.5-7b_CN-specific-Break_rank_top001 \
    --languages EN CN FR \
    --output_dir embedding_distribution/results \
    --pca_dims 50 \
    --random_seed 42

# 分析模型2: qwen2.5-7b_Break001 (Break模型)
echo ""
echo "----------------------------------------"
echo "分析模型 2: qwen2.5-7b_Break001"
echo "----------------------------------------"
python embedding_distribution/analyze_embedding_distribution.py \
    --model_path scratch_link/embeddings/qwen2.5-7b_EN-specific-Break_rank_top001 \
    --model_name qwen2.5-7b_EN-specific-Break_rank_top001 \
    --languages EN CN FR \
    --output_dir embedding_distribution/results \
    --pca_dims 50 \
    --random_seed 42


python embedding_distribution/analyze_embedding_distribution.py \
    --model_path scratch_link/embeddings/qwen2.5-7b_FR-specific-Break_rank_top001 \
    --model_name qwen2.5-7b_FR-specific-Break_rank_top001 \
    --languages EN CN FR \
    --output_dir embedding_distribution/results \
    --pca_dims 50 \
    --random_seed 42

echo ""
echo "========================================"
echo "分析完成！"
echo "========================================"
echo "结果保存在: embedding_distribution/results/"
echo ""
echo "生成的文件："
echo "  - qwen2.5-7b_distribution.png"
echo "  - qwen2.5-7b_Break001_distribution.png"
echo "  - qwen2.5-7b_preprocessed.npz"
echo "  - qwen2.5-7b_Break001_preprocessed.npz"
echo "  - qwen2.5-7b_reduced_2d.npz"
echo "  - qwen2.5-7b_Break001_reduced_2d.npz"
echo ""
echo "使用以下命令查看结果："
echo "  ls -lh embedding_distribution/results/"
echo "========================================"

