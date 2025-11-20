"""
分析和可视化模型embedding的分布差异

这个脚本：
1. 加载模型在三种语言（EN, CN, FR）下生成的token级别embeddings
2. 通过mean pooling将subword tokens聚合为word-level embeddings
3. 使用PCA预降维到50维
4. 使用UMAP降维到2维进行可视化
5. 绘制三种语言的分布图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm
import argparse

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_embeddings_for_language(model_path, language):
    """
    加载特定语言的所有embeddings和对应的CSV文件
    
    Args:
        model_path: 模型embedding目录路径
        language: 语言代码 (EN, CN, FR)
    
    Returns:
        embeddings: 所有token的embeddings数组
        csv_df: 对应的CSV DataFrame
    """
    model_path = Path(model_path)
    
    # 读取CSV文件获取token信息
    csv_file = model_path / f"task-lpp{language}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
    
    csv_df = pd.read_csv(csv_file, index_col=0)
    print(f"  - CSV文件: {len(csv_df)} 个tokens")
    
    # 加载所有runs的embedding文件
    all_embeddings = []
    sections = sorted(csv_df['section'].unique())
    
    for section in sections:
        npy_file = model_path / f"task-lpp{language}_run-{section}.npy"
        if not npy_file.exists():
            print(f"  警告: 文件不存在 {npy_file}")
            continue
        
        emb = np.load(npy_file)  # shape: (1, n_tokens, hidden_size)
        emb = emb.squeeze(0)  # 移除batch维度: (n_tokens, hidden_size)
        all_embeddings.append(emb)
    
    # 拼接所有sections的embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  - Embeddings形状: {embeddings.shape}")
    
    # 验证embeddings和CSV行数是否匹配
    if len(embeddings) != len(csv_df):
        print(f"  警告: embeddings数量({len(embeddings)})与CSV行数({len(csv_df)})不匹配!")
    
    return embeddings, csv_df


def aggregate_subwords_to_words(embeddings, csv_df):
    """
    将subword token embeddings聚合为word-level embeddings
    
    使用mean pooling将属于同一个单词的多个subword tokens的embeddings平均化
    
    Args:
        embeddings: token级别的embeddings (n_tokens, hidden_size)
        csv_df: 包含word_idx信息的DataFrame
    
    Returns:
        word_embeddings: word级别的embeddings (n_words, hidden_size)
        word_indices: 对应的word索引列表
    """
    # 按word_idx分组进行mean pooling
    word_embeddings = []
    word_indices = []
    
    for word_idx, group in csv_df.groupby('word_idx'):
        # 获取该单词对应的所有token的索引
        token_indices = group.index.tolist()
        
        # 提取对应的embeddings并求平均
        word_emb = embeddings[token_indices].mean(axis=0)
        
        word_embeddings.append(word_emb)
        word_indices.append(word_idx)
    
    word_embeddings = np.stack(word_embeddings)
    
    print(f"  - 聚合前: {len(embeddings)} tokens")
    print(f"  - 聚合后: {len(word_embeddings)} words")
    
    return word_embeddings, word_indices


def load_and_preprocess_model_embeddings(model_path, languages=['EN', 'CN', 'FR']):
    """
    加载并预处理模型在所有语言下的embeddings
    
    Args:
        model_path: 模型embedding目录路径
        languages: 要处理的语言列表
    
    Returns:
        all_embeddings: 所有语言的word-level embeddings拼接 (n_total_words, hidden_size)
        language_labels: 每个embedding对应的语言标签
    """
    all_embeddings = []
    language_labels = []
    
    for lang in languages:
        print(f"\n加载 {lang} 语言数据...")
        
        # 加载token级别的embeddings
        token_embeddings, csv_df = load_embeddings_for_language(model_path, lang)
        
        # 聚合为word级别
        word_embeddings, _ = aggregate_subwords_to_words(token_embeddings, csv_df)
        
        all_embeddings.append(word_embeddings)
        language_labels.extend([lang] * len(word_embeddings))
    
    # 拼接所有语言的embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    language_labels = np.array(language_labels)
    
    print(f"\n总计: {len(all_embeddings)} 个words from {len(languages)} 种语言")
    print(f"Embedding维度: {all_embeddings.shape[1]}")
    
    return all_embeddings, language_labels


def dimensionality_reduction(embeddings, n_pca_components=50, random_state=42):
    """
    执行降维：先PCA到50维，再UMAP到2维
    
    Args:
        embeddings: 输入embeddings (n_samples, hidden_size)
        n_pca_components: PCA保留的维度数
        random_state: 随机种子
    
    Returns:
        embeddings_2d: 降维后的2D坐标 (n_samples, 2)
        pca_model: 训练好的PCA模型
        umap_model: 训练好的UMAP模型
    """
    print(f"\n开始降维...")
    print(f"原始维度: {embeddings.shape}")
    
    # 步骤1: PCA降维到50维
    print(f"PCA降维到 {n_pca_components} 维...")
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    embeddings_pca = pca.fit_transform(embeddings)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"  - PCA保留了 {explained_variance:.2%} 的方差")
    print(f"  - PCA后形状: {embeddings_pca.shape}")
    
    # 步骤2: UMAP降维到2维
    print(f"UMAP降维到 2 维...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state
    )
    embeddings_2d = umap_model.fit_transform(embeddings_pca)
    
    print(f"  - UMAP后形状: {embeddings_2d.shape}")
    
    return embeddings_2d, pca, umap_model


def visualize_embeddings(embeddings_2d, language_labels, model_name, output_path):
    """
    可视化2D embeddings，三种语言用不同颜色
    
    Args:
        embeddings_2d: 2D坐标 (n_samples, 2)
        language_labels: 语言标签数组
        model_name: 模型名称（用于标题）
        output_path: 输出图片路径
    """
    # 定义语言颜色映射
    color_map = {
        'EN': '#E74C3C',  # 红色
        'CN': '#3498DB',  # 蓝色
        'FR': '#2ECC71'   # 绿色
    }
    
    # 定义语言名称映射
    lang_names = {
        'EN': 'English',
        'CN': 'Chinese',
        'FR': 'French'
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 为每种语言绘制散点图
    for lang in ['EN', 'CN', 'FR']:
        mask = language_labels == lang
        if mask.sum() == 0:
            continue
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color_map[lang],
            label=f"{lang_names[lang]} (n={mask.sum()})",
            alpha=0.6,
            s=10,
            edgecolors='none'
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.set_title(f'Embedding Distribution: {model_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='分析模型embedding在多语言下的分布'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型embedding目录路径 (如: scratch_link/embeddings/qwen2.5-7b)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称（用于图表标题和输出文件名）'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['EN', 'CN', 'FR'],
        help='要分析的语言列表'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='embedding_distribution/results',
        help='输出目录'
    )
    parser.add_argument(
        '--pca_dims',
        type=int,
        default=50,
        help='PCA降维目标维度'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"开始分析模型: {args.model_name}")
    print(f"模型路径: {args.model_path}")
    print(f"语言: {args.languages}")
    print("="*80)
    
    # 步骤1: 加载和预处理embeddings
    print("\n[步骤 1/3] 加载和预处理embeddings...")
    embeddings, language_labels = load_and_preprocess_model_embeddings(
        args.model_path,
        args.languages
    )
    
    # 保存预处理后的数据
    preprocessed_path = output_dir / f"{args.model_name}_preprocessed.npz"
    np.savez(
        preprocessed_path,
        embeddings=embeddings,
        language_labels=language_labels
    )
    print(f"\n预处理数据已保存到: {preprocessed_path}")
    
    # 步骤2: 降维
    print("\n[步骤 2/3] 降维...")
    embeddings_2d, pca_model, umap_model = dimensionality_reduction(
        embeddings,
        n_pca_components=args.pca_dims,
        random_state=args.random_seed
    )
    
    # 保存降维结果
    reduced_path = output_dir / f"{args.model_name}_reduced_2d.npz"
    np.savez(
        reduced_path,
        embeddings_2d=embeddings_2d,
        language_labels=language_labels
    )
    print(f"降维结果已保存到: {reduced_path}")
    
    # 步骤3: 可视化
    print("\n[步骤 3/3] 可视化...")
    output_path = output_dir / f"{args.model_name}_distribution.png"
    visualize_embeddings(
        embeddings_2d,
        language_labels,
        args.model_name,
        output_path
    )
    
    # 打印统计信息
    print("\n" + "="*80)
    print("分析完成！统计信息：")
    print("="*80)
    for lang in args.languages:
        count = (language_labels == lang).sum()
        print(f"{lang}: {count} words")
    print(f"总计: {len(embeddings)} words")
    print(f"\n所有结果保存在: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

