#!/bin/bash
#SBATCH --job-name=emb_reduce
#SBATCH --output=%j.emb_reduce.out
#SBATCH --error=%j.emb_reduce.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --account=EUHPC_D26_076
#SBATCH --partition=boost_usr_prod

cd /leonardo_work/EUHPC_B24_036/yang
source encoding_env/bin/activate

echo "========================================"
echo "Embedding Dimensionality Reduction"
echo "========================================"

# 设置LLM家族和模型列表
LLM_FAMILY="llama"  # 可改为 llama 或 mistral
OUTPUT_DIR="embedding_distribution/results"

# Qwen模型列表
QWEN_MODELS=(
    # "qwen2.5-7b"
    # "qwen2.5-7b_Break001"
    # "qwen2.5-7b_CN-specific-Break_rank_top001"
    # "qwen2.5-7b_EN-specific-Break_rank_top001"
    # "qwen2.5-7b_FR-specific-Break_rank_top001"\
    "llama2_7b"
    "llama2_7b_Break001"
    "llama2_7b_CN-specific-Break_rank_top001"
    "llama2_7b_EN-specific-Break_rank_top001"
    "llama2_7b_FR-specific-Break_rank_top001"
)


# 降维方法列表
METHODS=("pca_umap" "umap" "tsne")

echo ""
echo "LLM家族: ${LLM_FAMILY}"
echo "模型数量: ${#QWEN_MODELS[@]}"
echo "降维方法: ${METHODS[@]}"
echo ""

# 遍历所有模型进行降维
for model_name in "${QWEN_MODELS[@]}"; do
    echo "========================================"
    echo "降维模型: ${model_name}"
    echo "========================================"
    
    preprocessed_file="${OUTPUT_DIR}/${LLM_FAMILY}/${model_name}/preprocessed.npz"
    output_dir="${OUTPUT_DIR}/${LLM_FAMILY}/${model_name}"
    
    # 检查预处理文件是否存在
    if [ ! -f "$preprocessed_file" ]; then
        echo "警告: 预处理文件不存在: ${preprocessed_file}"
        echo "请先运行 run_preprocessing.sh"
        echo "跳过..."
        continue
    fi
    
    # 应用所有降维方法
    python embedding_distribution/reduce_dimensions.py \
        --preprocessed_file "${preprocessed_file}" \
        --output_dir "${output_dir}" \
        --methods "${METHODS[@]}" \
        --pca_dims 50 \
        --random_seed 42
    
    if [ $? -eq 0 ]; then
        echo "✓ ${model_name} 降维成功"
    else
        echo "✗ ${model_name} 降维失败"
    fi
    echo ""
done

echo "========================================"
echo "所有降维完成！"
echo "========================================"
echo "结果保存在: ${OUTPUT_DIR}/${LLM_FAMILY}/"
echo ""
echo "生成的文件结构："
echo "  ${OUTPUT_DIR}/"
echo "  └── ${LLM_FAMILY}/"
for model_name in "${QWEN_MODELS[@]}"; do
    echo "      └── ${model_name}/"
    echo "          ├── preprocessed.npz"
    echo "          ├── pca_umap_reduced_2d.npz"
    echo "          ├── umap_reduced_2d.npz"
    echo "          └── tsne_reduced_2d.npz"
done
echo ""
echo "下一步: 运行 run_visualization.sh 生成图表"
echo "========================================"

