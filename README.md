# Embedding Distribution Analysis

这个工具用于分析和可视化不同模型在多语言下生成的embedding分布差异，支持多种降维方法和多个LLM家族。

## 功能

1. **数据加载与预处理**: 从神经编码输出中加载token级别的embeddings，并通过mean pooling聚合为word-level embeddings
2. **多种降维方法**: 支持三种降维方法（PCA+UMAP、直接UMAP、t-SNE）
3. **多样化可视化**: 单模型可视化、跨模型对比、同语言对比
4. **多LLM家族支持**: 支持Qwen、Llama、Mistral等多个LLM家族

## 新的文件结构

```
embedding_distribution/
├── README.md                      # 本文件
├── preprocess_embeddings.py       # 预处理脚本
├── reduce_dimensions.py           # 降维脚本
├── visualize_single_model.py      # 单模型可视化
├── visualize_all_models.py        # 跨模型对比可视化
├── visualize_same_language.py     # 同语言对比可视化
├── run_preprocessing.sh           # 批量预处理 (SLURM)
├── run_reduction.sh               # 批量降维 (SLURM)
├── run_visualization.sh           # 批量可视化 (SLURM)
├── example_usage.sh               # 使用示例
├── analyze_embedding_distribution.py  # (旧脚本，保留用于参考)
└── results/                       # 输出目录
    ├── qwen/                      # Qwen模型结果
    │   ├── qwen2.5-7b/
    │   │   ├── preprocessed.npz
    │   │   ├── pca_umap_reduced_2d.npz
    │   │   ├── umap_reduced_2d.npz
    │   │   ├── tsne_reduced_2d.npz
    │   │   ├── pca_umap_distribution.png
    │   │   ├── umap_distribution.png
    │   │   └── tsne_distribution.png
    │   ├── qwen2.5-7b_Break001/
    │   │   └── (同样的结构)
    │   ├── all_models_comparison_pca_umap.png
    │   ├── all_models_comparison_umap.png
    │   ├── all_models_comparison_tsne.png
    │   ├── same_language_comparison_pca_umap.png
    │   ├── same_language_comparison_umap.png
    │   └── same_language_comparison_tsne.png
    ├── llama/                     # Llama模型结果
    │   └── (同样的结构)
    └── mistral/                   # Mistral模型结果
        └── (同样的结构)
```

## 工作流程

### 完整流程

```
原始数据 (Token级别)
    ↓
[预处理] preprocess_embeddings.py
    ↓ 生成 preprocessed.npz
[降维] reduce_dimensions.py
    ↓ 生成 *_reduced_2d.npz (3个方法)
[可视化] visualize_*.py
    ↓ 生成 *.png (多个图表)
最终结果
```

### 三个阶段

1. **预处理阶段** (`preprocess_embeddings.py`)
   - 加载.npy和.csv文件
   - Subword token → Word level (mean pooling)
   - 合并三种语言 (EN, CN, FR)
   - 输出: `preprocessed.npz`

2. **降维阶段** (`reduce_dimensions.py`)
   - 输入: `preprocessed.npz`
   - 应用3种降维方法:
     - **PCA+UMAP**: 先PCA降到50维，再UMAP降到2维（保留全局结构）
     - **Direct UMAP**: 直接从高维降到2维（更关注局部结构）
     - **t-SNE**: 经典非线性降维（强调局部聚类）
   - 输出: 3个 `*_reduced_2d.npz` 文件

3. **可视化阶段** (`visualize_*.py`)
   - 输入: `*_reduced_2d.npz`
   - 生成多种图表
   - 输出: `.png` 图片

## 降维方法说明

### 1. PCA+UMAP (pca_umap)
- **流程**: 3584维 → PCA(50维) → UMAP(2维)
- **优点**: 
  - 快速高效
  - 保留全局结构
  - PCA去除噪声
- **适用**: 大规模数据，关注整体分布

### 2. Direct UMAP (umap)
- **流程**: 3584维 → UMAP(2维)
- **优点**:
  - 保留更多局部结构
  - 聚类效果好
- **适用**: 关注细粒度聚类

### 3. t-SNE (tsne)
- **流程**: 3584维 → t-SNE(2维)
- **优点**:
  - 强调局部聚类
  - 经典可靠
- **缺点**: 
  - 较慢
  - 不保留全局距离
- **适用**: 小规模数据，探索性分析

## 使用方法

### 快速开始 (SLURM批量处理)

推荐用于处理所有模型：

```bash
cd /leonardo_work/EUHPC_B24_036/yang

# 步骤1: 预处理所有模型
sbatch embedding_distribution/run_preprocessing.sh

# 步骤2: 应用所有降维方法
sbatch embedding_distribution/run_reduction.sh

# 步骤3: 生成所有可视化
sbatch embedding_distribution/run_visualization.sh
```

### 单个模型处理

用于处理单个模型或测试：

```bash
cd /leonardo_work/EUHPC_B24_036/yang
source encoding_env/bin/activate

# 1. 预处理
python embedding_distribution/preprocess_embeddings.py \
    --model_path scratch_link/embeddings/qwen2.5-7b \
    --model_name qwen2.5-7b \
    --llm_family qwen \
    --languages EN CN FR \
    --output_dir embedding_distribution/results

# 2. 降维 (所有方法)
python embedding_distribution/reduce_dimensions.py \
    --preprocessed_file embedding_distribution/results/qwen/qwen2.5-7b/preprocessed.npz \
    --output_dir embedding_distribution/results/qwen/qwen2.5-7b \
    --methods pca_umap umap tsne

# 3. 可视化
python embedding_distribution/visualize_single_model.py \
    --model_name qwen2.5-7b \
    --llm_family qwen \
    --method pca_umap \
    --output_dir embedding_distribution/results
```

### 更多示例

查看 `example_usage.sh` 获取更多使用示例：

```bash
bash embedding_distribution/example_usage.sh
```

## 脚本参数说明

### preprocess_embeddings.py

```bash
--model_path    : 模型embedding目录路径 (必需)
--model_name    : 模型名称 (必需)
--llm_family    : LLM家族 (qwen/llama/mistral, 默认: qwen)
--languages     : 语言列表 (默认: EN CN FR)
--output_dir    : 输出根目录 (默认: embedding_distribution/results)
```

### reduce_dimensions.py

```bash
--preprocessed_file : 预处理文件路径 (必需)
--output_dir        : 输出目录 (必需)
--methods           : 降维方法列表 (默认: pca_umap umap tsne)
--pca_dims          : PCA维度 (默认: 50)
--random_seed       : 随机种子 (默认: 42)
```

### visualize_single_model.py

```bash
--model_name    : 模型名称
--llm_family    : LLM家族 (默认: qwen)
--method        : 降维方法 (默认: pca_umap)
--output_dir    : 结果根目录
--reduced_file  : 或直接指定降维文件路径
--output_file   : 或直接指定输出图片路径
```

### visualize_all_models.py

```bash
--llm_family    : LLM家族 (默认: qwen)
--method        : 降维方法 (默认: pca_umap)
--results_dir   : 结果根目录
--max_points    : 采样点数 (默认: 500)
--output_file   : 输出图片路径 (可选)
```

### visualize_same_language.py

```bash
--llm_family    : LLM家族 (默认: qwen)
--method        : 降维方法 (默认: pca_umap)
--results_dir   : 结果根目录
--max_points    : 采样点数 (默认: 5000)
--output_file   : 输出图片路径 (可选)
```

## 输入数据格式

### 原始Embeddings

```
scratch_link/embeddings/{model_name}/
├── task-lppEN.csv              # English token信息
├── task-lppEN_run-0.npy        # English embeddings (section 0)
├── task-lppEN_run-1.npy        # English embeddings (section 1)
├── task-lppCN.csv              # Chinese token信息
├── task-lppCN_run-0.npy        # Chinese embeddings
├── task-lppFR.csv              # French token信息
└── task-lppFR_run-0.npy        # French embeddings
```

### CSV格式

包含以下列：
- `word_idx`: 单词索引
- `word`: 单词文本
- `onset`: 开始时间
- `offset`: 结束时间
- `hftoken`: HuggingFace token
- `token_id`: Token ID
- `section`: Section编号

### NPY格式

- Shape: `(1, n_tokens, hidden_size)`
- 每个token的embedding向量

## 输出文件说明

### preprocessed.npz

```python
{
    'embeddings': np.array,      # (n_words, hidden_size)
    'language_labels': np.array  # (n_words,) 语言标签
}
```

### *_reduced_2d.npz

```python
{
    'embeddings_2d': np.array,    # (n_words, 2) 2D坐标
    'language_labels': np.array   # (n_words,) 语言标签
}
```

### 可视化图片

- **单模型图**: `{method}_distribution.png`
  - 显示一个模型在三种语言下的分布
  - 红色: English, 蓝色: Chinese, 绿色: French

- **跨模型对比图**: `all_models_comparison_{method}.png`
  - 显示5个模型在三种语言下的分布
  - 不同形状: 圆形(原始)、三角(Break001)、方形(Language-specific)

- **同语言对比图**: `same_language_comparison_{method}.png`
  - 显示CN→CN, EN→EN, FR→FR的对比

## 支持的LLM家族和模型

### Qwen模型 (已实现)

1. `qwen2.5-7b` - 基线模型
2. `qwen2.5-7b_Break001` - Break模型
3. `qwen2.5-7b_CN-specific-Break_rank_top001` - CN特定
4. `qwen2.5-7b_EN-specific-Break_rank_top001` - EN特定
5. `qwen2.5-7b_FR-specific-Break_rank_top001` - FR特定

### Llama模型

1. `llama2_7b` - 基线模型
2. `llama2_7b_Break001` - Break模型
3. `llama2_7b_CN-specific-Break_rank_top001` - CN特定
4. `llama2_7b_EN-specific-Break_rank_top001` - EN特定
5. `llama2_7b_FR-specific-Break_rank_top001` - FR特定

### Mistral模型 (待添加)

相同的5模型结构，只需修改 `run_*.sh` 中的 `LLM_FAMILY="mistral"`

## 性能考虑

### 内存需求

- **预处理**: 16-32GB (取决于模型大小)
- **降维**: 
  - PCA+UMAP: 32-64GB
  - Direct UMAP: 64-128GB
  - t-SNE: 64-128GB
- **可视化**: 8-16GB

### 时间估计

每个模型：
- 预处理: 5-10分钟
- PCA+UMAP: 10-20分钟
- Direct UMAP: 30-60分钟
- t-SNE: 60-120分钟
- 可视化: 2-5分钟

全部5个Qwen模型 + 3种方法：
- 总时间: 约8-12小时

## 依赖环境

使用 `encoding_env` 环境，需要包含：

```
numpy
pandas
matplotlib
scikit-learn      # PCA
umap-learn        # UMAP
scikit-learn      # t-SNE (from sklearn.manifold)
tqdm              # (可选，用于进度条)
```

安装命令：

```bash
source encoding_env/bin/activate
pip install numpy pandas matplotlib scikit-learn umap-learn
```

## 注意事项

1. **随机种子**: 所有降维方法默认使用 `random_seed=42` 保证可复现性
2. **颜色映射**: 
   - English: 红色 (#E74C3C)
   - Chinese: 蓝色 (#3498DB)
   - French: 绿色 (#2ECC71)
3. **文件大小**: preprocessed.npz 文件可能很大 (数百MB)，使用压缩存储
4. **采样**: 可视化时会采样数据点以提高渲染速度

## 常见问题

### Q1: 如何只运行一个降维方法？

```bash
python embedding_distribution/reduce_dimensions.py \
    --preprocessed_file results/qwen/qwen2.5-7b/preprocessed.npz \
    --output_dir results/qwen/qwen2.5-7b \
    --methods tsne
```

### Q2: 如何处理其他LLM家族？

修改 `run_*.sh` 脚本中的配置：

```bash
LLM_FAMILY="llama"  # 或 "mistral"

# 然后修改模型列表
LLAMA_MODELS=(
    "llama-base"
    "llama_Break001"
    # ...
)
```

### Q3: 如何跳过已处理的模型？

脚本会自动检查文件是否存在。如果要强制重新处理，删除对应的输出文件。

### Q4: 内存不足怎么办？

- 减少 `--pca_dims` (例如改为30)
- 使用更大内存的节点
- 在 SLURM 脚本中增加 `--mem`

### Q5: t-SNE太慢怎么办？

- 只在小规模数据上使用t-SNE
- 或者只运行 PCA+UMAP 和 Direct UMAP
- 调整 t-SNE 参数: `max_iter=500` (减少迭代次数)

## 扩展建议

1. **添加新的降维方法**: 在 `reduce_dimensions.py` 中添加新函数
2. **自定义可视化**: 修改 `visualize_*.py` 中的绘图参数
3. **批量对比**: 创建脚本对比不同方法的结果
4. **定量分析**: 添加聚类指标（如轮廓系数）

## 参考资料

- UMAP论文: McInnes et al., 2018
- t-SNE论文: van der Maaten & Hinton, 2008
- PCA: Jolliffe & Cadima, 2016

## 作者与维护

Created for analyzing multilingual LLM embedding distributions.

最后更新: 2024

---

**祝使用愉快！如有问题，请参考 `example_usage.sh` 或查看各脚本的帮助信息 (`--help`)。**
