#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化：3种模型 × 3种语言，放在一张图上
- 模型1: qwen原始 (圆形)
- 模型2: Break001 (三角形)
- 模型3: Language-specific same-language (方形)
- 颜色: CN=红, EN=蓝, FR=绿

支持新的文件结构和多种降维方法
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_model_language_data(model_name, language, llm_family, method, results_dir):
    """
    加载特定模型的特定语言数据
    
    Args:
        model_name: 模型名称
        language: 语言代码 (EN, CN, FR)
        llm_family: LLM家族 (qwen, llama, mistral)
        method: 降维方法 (pca_umap, umap, tsne)
        results_dir: 结果根目录
    """
    npz_file = Path(results_dir) / llm_family / model_name / f"{method}_reduced_2d.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"文件不存在: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    
    languages = data['language_labels']
    lang_mask = (languages == language)
    
    return data['embeddings_2d'][lang_mask]

def sample_data(data, max_points=5000):
    """采样数据"""
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points, replace=False)
        return data[idx]
    return data

def main():
    parser = argparse.ArgumentParser(
        description='可视化多个模型在三种语言下的embedding分布对比'
    )
    parser.add_argument('--results_dir', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results',
                       help='结果根目录')
    parser.add_argument('--output_file', type=str,
                       help='输出图片路径 (如果不提供会自动生成)')
    parser.add_argument('--max_points', type=int, default=500,
                       help='每个语言的最大采样点数')
    parser.add_argument('--llm_family', type=str, default='qwen',
                       choices=['qwen', 'llama', 'mistral'],
                       help='LLM家族名称')
    parser.add_argument('--method', type=str, default='pca_umap',
                       choices=['pca_umap', 'umap', 'tsne'],
                       help='降维方法')
    args = parser.parse_args()
    
    # 自动生成输出文件名
    if not args.output_file:
        output_dir = Path(args.results_dir) / args.llm_family
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(output_dir / f"all_models_comparison_{args.method}.png")
    
    np.random.seed(42)
    
    # 配置模型名称
    if args.llm_family == 'qwen':
        models = {
            'qwen': 'qwen2.5-7b',
            'break': 'qwen2.5-7b_Break001',
            'cn_specific': 'qwen2.5-7b_CN-specific-Break_rank_top001',
            'en_specific': 'qwen2.5-7b_EN-specific-Break_rank_top001',
            'fr_specific': 'qwen2.5-7b_FR-specific-Break_rank_top001'
        }
    elif args.llm_family == 'llama':
        models = {
            'llama': 'llama2_7b',
            'break': 'llama2_7b_Break001',
            'cn_specific': 'llama2_7b_CN-specific-Break_rank_top001',
            'en_specific': 'llama2_7b_EN-specific-Break_rank_top001',
            'fr_specific': 'llama2_7b_FR-specific-Break_rank_top001'
        }
    elif args.llm_family == 'mistral':
        models = {
            'mistral': 'mistral-base',
            'break': 'mistral_Break001',
            'cn_specific': 'mistral_CN-specific-Break_rank_top001',
            'en_specific': 'mistral_EN-specific-Break_rank_top001',
            'fr_specific': 'mistral_FR-specific-Break_rank_top001'
        }
    
    languages = ['CN', 'EN', 'FR']
    colors = {'CN': 'red', 'EN': 'blue', 'FR': 'green'}
    markers = {'original': 'o', 'break': '^', 'specific': 's'}
    model_names = {
        'original': 'Original',
        'break': 'Break001',
        'specific': 'Lang-Specific'
    }
    
    # 方法名称映射
    method_names = {
        'pca_umap': 'PCA+UMAP',
        'umap': 'UMAP',
        'tsne': 't-SNE'
    }
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 模型1: 原始模型
    model_key = list(models.keys())[0]  # 'qwen', 'llama', or 'mistral'
    print(f"加载 {args.llm_family} 原始模型...")
    for lang in languages:
        try:
            data = load_model_language_data(models[model_key], lang, args.llm_family, 
                                           args.method, args.results_dir)
            data = sample_data(data, args.max_points)
            plt.scatter(data[:, 0], data[:, 1], 
                       c=colors[lang], marker=markers['original'],
                       alpha=0.4, s=15, edgecolors='none',
                       label=f'{model_names["original"]} - {lang}')
            print(f"  {lang}: {len(data)} samples")
        except FileNotFoundError as e:
            print(f"  警告: {lang} 数据未找到")
    
    # 模型2: Break001
    print("\n加载 Break001 模型...")
    for lang in languages:
        try:
            data = load_model_language_data(models['break'], lang, args.llm_family,
                                           args.method, args.results_dir)
            data = sample_data(data, args.max_points)
            plt.scatter(data[:, 0], data[:, 1], 
                       c=colors[lang], marker=markers['break'],
                       alpha=0.4, s=20, edgecolors='none',
                       label=f'{model_names["break"]} - {lang}')
            print(f"  {lang}: {len(data)} samples")
        except FileNotFoundError as e:
            print(f"  警告: {lang} 数据未找到")
    
    # 模型3: Language-specific same-language
    print("\n加载 Language-specific 模型（same-language）...")
    for lang in languages:
        if lang == 'CN':
            model_key = 'cn_specific'
        elif lang == 'EN':
            model_key = 'en_specific'
        else:  # FR
            model_key = 'fr_specific'
        
        try:
            data = load_model_language_data(models[model_key], lang, args.llm_family,
                                           args.method, args.results_dir)
            data = sample_data(data, args.max_points)
            plt.scatter(data[:, 0], data[:, 1], 
                       c=colors[lang], marker=markers['specific'],
                       alpha=0.4, s=25, edgecolors='none',
                       label=f'{model_names["specific"]} - {lang}')
            print(f"  {lang}: {len(data)} samples")
        except FileNotFoundError as e:
            print(f"  警告: {lang} 数据未找到")
    
    # 图例和标签
    method_display = method_names.get(args.method, args.method.upper())
    plt.xlabel(f'{method_display} Dimension 1', fontsize=13)
    plt.ylabel(f'{method_display} Dimension 2', fontsize=13)
    plt.title(f'Embedding Distribution Comparison: {args.llm_family.upper()}\nAcross Models and Languages (Method: {method_display})', 
             fontsize=15, fontweight='bold')
    
    # 创建图例（分为两列）
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=10, ncol=1, framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 已保存: {args.output_file}")

if __name__ == '__main__':
    main()

