#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化单个模型的embedding分布

绘制一个模型在三种语言（EN, CN, FR）下的2D embedding分布
支持多种降维方法的结果
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_embeddings(embeddings_2d, language_labels, model_name, method, output_path):
    """
    可视化2D embeddings，三种语言用不同颜色
    
    Args:
        embeddings_2d: 2D坐标 (n_samples, 2)
        language_labels: 语言标签数组
        model_name: 模型名称（用于标题）
        method: 降维方法名称（用于标题）
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
    
    # 方法名称映射
    method_names = {
        'pca_umap': 'PCA+UMAP',
        'umap': 'UMAP',
        'tsne': 't-SNE'
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
    
    # 设置坐标轴标签
    method_display = method_names.get(method, method.upper())
    ax.set_xlabel(f'{method_display} Dimension 1', fontsize=14)
    ax.set_ylabel(f'{method_display} Dimension 2', fontsize=14)
    ax.set_title(f'Embedding Distribution: {model_name}\n(Method: {method_display})', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, markerscale=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='可视化单个模型的embedding分布'
    )
    parser.add_argument(
        '--reduced_file',
        type=str,
        help='降维后的npz文件路径 (如果不提供，会从其他参数自动构建)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='模型名称'
    )
    parser.add_argument(
        '--llm_family',
        type=str,
        default='qwen',
        choices=['qwen', 'llama', 'mistral'],
        help='LLM家族名称'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pca_umap',
        choices=['pca_umap', 'umap', 'tsne'],
        help='降维方法'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='embedding_distribution/results',
        help='结果根目录'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='输出图片路径 (如果不提供，会自动构建)'
    )
    
    args = parser.parse_args()
    
    # 构建文件路径
    if args.reduced_file:
        reduced_file = Path(args.reduced_file)
    else:
        if not args.model_name:
            parser.error("必须提供 --reduced_file 或 --model_name")
        reduced_file = Path(args.output_dir) / args.llm_family / args.model_name / f"{args.method}_reduced_2d.npz"
    
    if not reduced_file.exists():
        raise FileNotFoundError(f"降维文件不存在: {reduced_file}")
    
    # 构建输出路径
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = reduced_file.parent / f"{args.method}_distribution.png"
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"可视化单个模型分布")
    print(f"模型名称: {args.model_name}")
    print(f"LLM家族: {args.llm_family}")
    print(f"降维方法: {args.method}")
    print(f"输入文件: {reduced_file}")
    print(f"输出文件: {output_path}")
    print("="*80)
    
    # 加载降维后的数据
    print("\n加载降维数据...")
    data = np.load(reduced_file, allow_pickle=True)
    embeddings_2d = data['embeddings_2d']
    language_labels = data['language_labels']
    
    print(f"  - 数据形状: {embeddings_2d.shape}")
    print(f"  - 语言标签数量: {len(language_labels)}")
    
    # 打印语言分布
    print("\n语言分布:")
    for lang in ['EN', 'CN', 'FR']:
        count = (language_labels == lang).sum()
        print(f"  {lang}: {count} words")
    
    # 可视化
    print("\n生成可视化...")
    visualize_embeddings(
        embeddings_2d,
        language_labels,
        args.model_name if args.model_name else reduced_file.parent.name,
        args.method,
        output_path
    )
    
    print("\n" + "="*80)
    print("可视化完成！")
    print(f"输出: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

