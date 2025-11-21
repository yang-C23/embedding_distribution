#!/bin/bash
#SBATCH --job-name=emb_visualize
#SBATCH --output=%j.emb_visualize.out
#SBATCH --error=%j.emb_visualize.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --account=EUHPC_D26_076
#SBATCH --partition=boost_usr_prod

cd /leonardo_work/EUHPC_B24_036/yang
source encoding_env/bin/activate

echo "========================================"
echo "Embedding Visualization"
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

# 1. 为每个模型 × 每个方法生成单模型可视化
echo "========================================"
echo "步骤 1: 生成单模型可视化"
echo "========================================"

for model_name in "${QWEN_MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "可视化: ${model_name} - ${method}"
        
        python embedding_distribution/visualize_single_model.py \
            --model_name "${model_name}" \
            --llm_family "${LLM_FAMILY}" \
            --method "${method}" \
            --output_dir "${OUTPUT_DIR}"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ 完成"
        else
            echo "  ✗ 失败"
        fi
    done
    echo ""
done

# 2. 为每个方法生成跨模型对比可视化
echo "========================================"
echo "步骤 2: 生成跨模型对比可视化"
echo "========================================"

for method in "${METHODS[@]}"; do
    echo "生成跨模型对比: ${method}"
    
    python embedding_distribution/visualize_all_models.py \
        --llm_family "${LLM_FAMILY}" \
        --method "${method}" \
        --results_dir "${OUTPUT_DIR}" \
        --max_points 5000
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 完成"
    else
        echo "  ✗ 失败"
    fi
    echo ""
done

# 3. 为每个方法生成同语言对比可视化
echo "========================================"
echo "步骤 3: 生成同语言对比可视化"
echo "========================================"

# for method in "${METHODS[@]}"; do
#     echo "生成同语言对比: ${method}"
    
#     python embedding_distribution/visualize_same_language.py \
#         --llm_family "${LLM_FAMILY}" \
#         --method "${method}" \
#         --results_dir "${OUTPUT_DIR}" \
#         --max_points 5000
    
#     if [ $? -eq 0 ]; then
#         echo "  ✓ 完成"
#     else
#         echo "  ✗ 失败"
#     fi
#     echo ""
# done

echo "========================================"
echo "所有可视化完成！"
echo "========================================"
echo "结果保存在: ${OUTPUT_DIR}/${LLM_FAMILY}/"
echo ""
echo "生成的图表："
echo ""
echo "1. 单模型可视化 (每个模型 × 每个方法):"
for model_name in "${QWEN_MODELS[@]}"; do
    echo "   ${OUTPUT_DIR}/${LLM_FAMILY}/${model_name}/"
    for method in "${METHODS[@]}"; do
        echo "     - ${method}_distribution.png"
    done
done
echo ""
echo "2. 跨模型对比 (每个方法):"
for method in "${METHODS[@]}"; do
    echo "   ${OUTPUT_DIR}/${LLM_FAMILY}/all_models_comparison_${method}.png"
done
# echo ""
# echo "3. 同语言对比 (每个方法):"
# for method in "${METHODS[@]}"; do
#     echo "   ${OUTPUT_DIR}/${LLM_FAMILY}/same_language_comparison_${method}.png"
# done
echo ""
echo "========================================"

