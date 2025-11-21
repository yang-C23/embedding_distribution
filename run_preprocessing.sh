#!/bin/bash
#SBATCH --job-name=emb_preprocess
#SBATCH --output=%j.emb_preprocess.out
#SBATCH --error=%j.emb_preprocess.err
#SBATCH --time=8:00:00
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
echo "Embedding Preprocessing"
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

echo ""
echo "LLM家族: ${LLM_FAMILY}"
echo "模型数量: ${#QWEN_MODELS[@]}"
echo ""

# 遍历所有模型进行预处理
for model_name in "${QWEN_MODELS[@]}"; do
    echo "========================================"
    echo "预处理模型: ${model_name}"
    echo "========================================"
    
    model_path="scratch_link/embeddings/${model_name}"
    
    # 检查模型路径是否存在
    if [ ! -d "$model_path" ]; then
        echo "警告: 模型路径不存在: ${model_path}"
        echo "跳过..."
        continue
    fi
    
    python embedding_distribution/preprocess_embeddings.py \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --llm_family "${LLM_FAMILY}" \
        --languages EN CN FR \
        --output_dir "${OUTPUT_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "✓ ${model_name} 预处理成功"
    else
        echo "✗ ${model_name} 预处理失败"
    fi
    echo ""
done

echo "========================================"
echo "所有预处理完成！"
echo "========================================"
echo "结果保存在: ${OUTPUT_DIR}/${LLM_FAMILY}/"
echo ""
echo "生成的文件结构："
echo "  ${OUTPUT_DIR}/"
echo "  └── ${LLM_FAMILY}/"
for model_name in "${QWEN_MODELS[@]}"; do
    echo "      └── ${model_name}/"
    echo "          └── preprocessed.npz"
done
echo ""
echo "下一步: 运行 run_reduction.sh 进行降维"
echo "========================================"

