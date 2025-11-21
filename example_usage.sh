#!/bin/bash
# 
# Example Usage: Embedding Distribution Analysis
# ===============================================
# 
# This script demonstrates how to use the embedding distribution analysis tools.
# These examples are NOT for SLURM submission - run them interactively or in a regular bash script.
#
# For SLURM batch processing, use:
#   - run_preprocessing.sh
#   - run_reduction.sh
#   - run_visualization.sh
#

# Activate the environment
source encoding_env/bin/activate

# =============================================================================
# Example 1: Preprocess a single model
# =============================================================================
echo "Example 1: Preprocess a single model"
echo "====================================="

python embedding_distribution/preprocess_embeddings.py \
    --model_path scratch_link/embeddings/qwen2.5-7b \
    --model_name qwen2.5-7b \
    --llm_family qwen \
    --languages EN CN FR \
    --output_dir embedding_distribution/results

echo ""
echo "Output: embedding_distribution/results/qwen/qwen2.5-7b/preprocessed.npz"
echo ""

# =============================================================================
# Example 2: Apply all reduction methods to preprocessed data
# =============================================================================
echo "Example 2: Apply all reduction methods"
echo "======================================="

python embedding_distribution/reduce_dimensions.py \
    --preprocessed_file embedding_distribution/results/qwen/qwen2.5-7b/preprocessed.npz \
    --output_dir embedding_distribution/results/qwen/qwen2.5-7b \
    --methods pca_umap umap tsne \
    --pca_dims 50 \
    --random_seed 42

echo ""
echo "Outputs:"
echo "  - embedding_distribution/results/qwen/qwen2.5-7b/pca_umap_reduced_2d.npz"
echo "  - embedding_distribution/results/qwen/qwen2.5-7b/umap_reduced_2d.npz"
echo "  - embedding_distribution/results/qwen/qwen2.5-7b/tsne_reduced_2d.npz"
echo ""

# =============================================================================
# Example 3: Apply only one reduction method
# =============================================================================
echo "Example 3: Apply only t-SNE reduction"
echo "======================================"

python embedding_distribution/reduce_dimensions.py \
    --preprocessed_file embedding_distribution/results/qwen/qwen2.5-7b/preprocessed.npz \
    --output_dir embedding_distribution/results/qwen/qwen2.5-7b \
    --methods tsne \
    --random_seed 42

echo ""

# =============================================================================
# Example 4: Visualize a single model with specific method
# =============================================================================
echo "Example 4: Visualize single model (PCA+UMAP)"
echo "============================================="

python embedding_distribution/visualize_single_model.py \
    --model_name qwen2.5-7b \
    --llm_family qwen \
    --method pca_umap \
    --output_dir embedding_distribution/results

echo ""
echo "Output: embedding_distribution/results/qwen/qwen2.5-7b/pca_umap_distribution.png"
echo ""

# =============================================================================
# Example 5: Visualize single model with t-SNE
# =============================================================================
echo "Example 5: Visualize single model (t-SNE)"
echo "=========================================="

python embedding_distribution/visualize_single_model.py \
    --model_name qwen2.5-7b \
    --llm_family qwen \
    --method tsne \
    --output_dir embedding_distribution/results

echo ""

# =============================================================================
# Example 6: Compare all models (requires all 5 models to be processed)
# =============================================================================
echo "Example 6: Compare all Qwen models with UMAP"
echo "=============================================="

python embedding_distribution/visualize_all_models.py \
    --llm_family qwen \
    --method umap \
    --results_dir embedding_distribution/results \
    --max_points 500

echo ""
echo "Output: embedding_distribution/results/qwen/all_models_comparison_umap.png"
echo ""

# =============================================================================
# Example 7: Same-language comparison
# =============================================================================
echo "Example 7: Same-language comparison with PCA+UMAP"
echo "=================================================="

python embedding_distribution/visualize_same_language.py \
    --llm_family qwen \
    --method pca_umap \
    --results_dir embedding_distribution/results \
    --max_points 5000

echo ""
echo "Output: embedding_distribution/results/qwen/same_language_comparison_pca_umap.png"
echo ""

# =============================================================================
# Example 8: Complete pipeline for one model
# =============================================================================
echo "Example 8: Complete pipeline for one model"
echo "==========================================="

MODEL_NAME="qwen2.5-7b_Break001"
MODEL_PATH="scratch_link/embeddings/${MODEL_NAME}"
LLM_FAMILY="qwen"

# Step 1: Preprocess
echo "Step 1: Preprocessing..."
python embedding_distribution/preprocess_embeddings.py \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --llm_family "${LLM_FAMILY}" \
    --languages EN CN FR \
    --output_dir embedding_distribution/results

# Step 2: Reduce dimensions
echo ""
echo "Step 2: Dimensionality reduction..."
python embedding_distribution/reduce_dimensions.py \
    --preprocessed_file "embedding_distribution/results/${LLM_FAMILY}/${MODEL_NAME}/preprocessed.npz" \
    --output_dir "embedding_distribution/results/${LLM_FAMILY}/${MODEL_NAME}" \
    --methods pca_umap umap tsne

# Step 3: Visualize
echo ""
echo "Step 3: Visualization..."
for method in pca_umap umap tsne; do
    python embedding_distribution/visualize_single_model.py \
        --model_name "${MODEL_NAME}" \
        --llm_family "${LLM_FAMILY}" \
        --method "${method}" \
        --output_dir embedding_distribution/results
done

echo ""
echo "Complete! Check embedding_distribution/results/${LLM_FAMILY}/${MODEL_NAME}/"
echo ""

# =============================================================================
# Example 9: Using explicit file paths
# =============================================================================
echo "Example 9: Using explicit file paths"
echo "====================================="

python embedding_distribution/visualize_single_model.py \
    --reduced_file embedding_distribution/results/qwen/qwen2.5-7b/pca_umap_reduced_2d.npz \
    --output_file embedding_distribution/results/qwen/qwen2.5-7b/my_custom_plot.png

echo ""

# =============================================================================
# Example 10: Processing a Llama model (when available)
# =============================================================================
echo "Example 10: Processing Llama model (template)"
echo "=============================================="
echo "# When you have Llama models, use:"
echo ""
echo "python embedding_distribution/preprocess_embeddings.py \\"
echo "    --model_path scratch_link/embeddings/llama-base \\"
echo "    --model_name llama-base \\"
echo "    --llm_family llama \\"
echo "    --languages EN CN FR \\"
echo "    --output_dir embedding_distribution/results"
echo ""
echo "# Then continue with reduce_dimensions.py and visualize_*.py"
echo "# using --llm_family llama"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "======================================"
echo "Summary of Usage Patterns"
echo "======================================"
echo ""
echo "Workflow:"
echo "  1. preprocess_embeddings.py  → creates preprocessed.npz"
echo "  2. reduce_dimensions.py      → creates *_reduced_2d.npz files"
echo "  3. visualize_*.py            → creates .png visualizations"
echo ""
echo "Key Parameters:"
echo "  --llm_family: qwen, llama, or mistral"
echo "  --method: pca_umap, umap, or tsne"
echo "  --model_name: name of the model (e.g., qwen2.5-7b)"
echo ""
echo "For batch processing on SLURM:"
echo "  sbatch embedding_distribution/run_preprocessing.sh"
echo "  sbatch embedding_distribution/run_reduction.sh"
echo "  sbatch embedding_distribution/run_visualization.sh"
echo ""
echo "======================================"

