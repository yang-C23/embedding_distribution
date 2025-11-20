# Embedding Distribution Analysis

这个工具用于分析和可视化不同模型在多语言下生成的embedding分布差异。

## 功能

1. **数据加载**: 从神经编码输出中加载token级别的embeddings
2. **预处理**: 将subword tokens通过mean pooling聚合为word-level embeddings
3. **降维**: PCA (保留50维) → UMAP (降到2维)
4. **可视化**: 绘制EN/CN/FR三种语言的2D分布图

## 文件结构

```
embedding_distribution/
├── README.md                           # 本文件
├── analyze_embedding_distribution.py   # 主分析脚本
├── run_analysis.sh                     # 批量运行脚本
└── results/                            # 输出目录
    ├── qwen2.5-7b_distribution.png         # 基线模型分布图
    ├── qwen2.5-7b_Break001_distribution.png # Break模型分布图
    ├── qwen2.5-7b_preprocessed.npz          # 预处理后的embeddings
    ├── qwen2.5-7b_Break001_preprocessed.npz
    ├── qwen2.5-7b_reduced_2d.npz            # 2D降维结果
    └── qwen2.5-7b_Break001_reduced_2d.npz
```

## 数据格式说明

### 输入数据

- **Embeddings**: `scratch_link/embeddings/{model_name}/task-lpp{lang}_run-{section}.npy`
  - Shape: `(1, n_tokens, hidden_size)`
  - 每个.npy文件对应一个section的所有token embeddings

- **Transcript CSV**: `scratch_link/embeddings/{model_name}/task-lpp{lang}.csv`
  - 包含每个token的元信息：word_idx, word, onset, offset, hftoken, token_id
  - 一个单词可能对应多行（多个subword tokens）

### 预处理步骤

1. **加载数据**: 读取所有runs的.npy文件并拼接
2. **Token到Word聚合**: 
   - 问题：一个单词被分成多个subword tokens（如 "saw" → "s" + "aw"）
   - 解决：按word_idx分组，对同一单词的所有subword embeddings进行mean pooling
   - 效果：避免重复计数，得到真正的word-level表征

3. **跨语言拼接**: 将EN/CN/FR的word embeddings拼接成统一数据集

## 使用方法

### 方法1: 使用批量运行脚本（推荐）

```bash
cd /leonardo_work/EUHPC_B24_036/yang
source encoding_env/bin/activate
bash embedding_distribution/run_analysis.sh
```

这会自动分析两个模型并生成所有结果。

### 方法2: 单独分析某个模型

```bash
cd /leonardo_work/EUHPC_B24_036/yang
source encoding_env/bin/activate

python embedding_distribution/analyze_embedding_distribution.py \
    --model_path scratch_link/embeddings/qwen2.5-7b \
    --model_name qwen2.5-7b \
    --languages EN CN FR \
    --output_dir embedding_distribution/results \
    --pca_dims 50 \
    --random_seed 42
```

### 参数说明

- `--model_path`: 模型embedding目录路径
- `--model_name`: 模型名称（用于输出文件命名和图表标题）
- `--languages`: 要分析的语言列表（默认: EN CN FR）
- `--output_dir`: 输出目录（默认: embedding_distribution/results）
- `--pca_dims`: PCA保留的维度数（默认: 50）
- `--random_seed`: 随机种子（默认: 42）

## 输出结果

### 1. 可视化图片

- `{model_name}_distribution.png`: 2D散点图
  - 红色: English
  - 蓝色: Chinese
  - 绿色: French
  - 分辨率: 300 DPI

### 2. 中间数据文件

- `{model_name}_preprocessed.npz`: 预处理后的word-level embeddings
  - `embeddings`: (n_words, hidden_size)
  - `language_labels`: (n_words,)

- `{model_name}_reduced_2d.npz`: 降维后的2D坐标
  - `embeddings_2d`: (n_words, 2)
  - `language_labels`: (n_words,)

## 分析流程

```
原始数据 (Token级别)
    ↓
[加载] 读取.npy和.csv文件
    ↓
[预处理] Subword → Word (mean pooling)
    ↓
[拼接] 合并三种语言
    ↓
[PCA] 降维到50维 (保留主要方差)
    ↓
[UMAP] 非线性降维到2维
    ↓
[可视化] 绘制散点图
    ↓
结果图片 + 数据文件
```

## 依赖环境

使用 `encoding_env` 环境，需要包含：
- numpy
- pandas
- matplotlib
- scikit-learn (PCA)
- umap-learn (UMAP)
- tqdm

## 注意事项

1. **内存需求**: 对于大型模型和长文本，可能需要较大内存
2. **随机种子**: UMAP是随机算法，设置random_seed保证结果可复现
3. **颜色映射**: 三种语言使用红/蓝/绿便于区分
4. **保存格式**: 图片保存为PNG格式，300 DPI高分辨率

## 预期发现

通过对比两个模型的分布图，可以观察：

1. **语言分离度**: 三种语言在2D空间中的聚类程度
2. **跨语言重叠**: 不同语言embedding的重叠区域
3. **Break模型影响**: Break001模型相比基线模型的分布变化

## 问题排查

如果遇到问题：

1. **文件不存在**: 确保已运行 `encode_qwen2.5-7b_Break001.sh` 生成embeddings
2. **维度不匹配**: 检查.npy和.csv文件的token数量是否一致
3. **内存错误**: 减少batch size或使用更大内存的机器
4. **导入错误**: 确保所有依赖包已安装在encoding_env环境中

## 作者

Created for analyzing multilingual LLM embedding distributions.

