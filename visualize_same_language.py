#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化：CN破坏→CN, EN破坏→EN, FR破坏→FR
支持新的文件结构和多种降维方法
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_language_data(model_name, language, llm_family, method, results_dir):
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
    
    # 找到对应语言的索引
    languages = data['language_labels']
    lang_mask = (languages == language)
    
    return data['embeddings_2d'][lang_mask]

def main():
    parser = argparse.ArgumentParser(
        description='可视化同语言对比：CN破坏→CN, EN破坏→EN, FR破坏→FR'
    )
    parser.add_argument('--results_dir', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results',
                       help='结果根目录')
    parser.add_argument('--output_file', type=str,
                       help='输出图片路径 (如果不提供会自动生成)')
    parser.add_argument('--llm_family', type=str, default='qwen',
                       choices=['qwen', 'llama', 'mistral'],
                       help='LLM家族名称')
    parser.add_argument('--method', type=str, default='pca_umap',
                       choices=['pca_umap', 'umap', 'tsne'],
                       help='降维方法')
    parser.add_argument('--max_points', type=int, default=5000,
                       help='每个语言的最大采样点数')
    args = parser.parse_args()
    
    # 构建模型名称
    if args.llm_family == 'qwen':
        cn_model = 'qwen2.5-7b_CN-specific-Break_rank_top001'
        en_model = 'qwen2.5-7b_EN-specific-Break_rank_top001'
        fr_model = 'qwen2.5-7b_FR-specific-Break_rank_top001'
    elif args.llm_family == 'llama':
        cn_model = 'llama2_7b_CN-specific-Break_rank_top001'
        en_model = 'llama2_7b_EN-specific-Break_rank_top001'
        fr_model = 'llama2_7b_FR-specific-Break_rank_top001'
    elif args.llm_family == 'mistral':
        cn_model = 'mistral_CN-specific-Break_rank_top001'
        en_model = 'mistral_EN-specific-Break_rank_top001'
        fr_model = 'mistral_FR-specific-Break_rank_top001'
    
    # 自动生成输出文件名
    if not args.output_file:
        output_dir = Path(args.results_dir) / args.llm_family
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(output_dir / f"same_language_comparison_{args.method}.png")
    
    # 加载数据
    print(f"加载 {args.llm_family.upper()} 模型数据 (方法: {args.method})...")
    try:
        cn_data = load_language_data(cn_model, 'CN', args.llm_family, args.method, args.results_dir)
        en_data = load_language_data(en_model, 'EN', args.llm_family, args.method, args.results_dir)
        fr_data = load_language_data(fr_model, 'FR', args.llm_family, args.method, args.results_dir)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保已经运行了预处理和降维步骤")
        return
    
    print(f"  CN破坏→CN: {len(cn_data)} samples")
    print(f"  EN破坏→EN: {len(en_data)} samples")
    print(f"  FR破坏→FR: {len(fr_data)} samples")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 采样以加快绘制
    np.random.seed(42)
    
    if len(cn_data) > args.max_points:
        idx = np.random.choice(len(cn_data), args.max_points, replace=False)
        cn_data = cn_data[idx]
    if len(en_data) > args.max_points:
        idx = np.random.choice(len(en_data), args.max_points, replace=False)
        en_data = en_data[idx]
    if len(fr_data) > args.max_points:
        idx = np.random.choice(len(fr_data), args.max_points, replace=False)
        fr_data = fr_data[idx]
    
    plt.scatter(cn_data[:, 0], cn_data[:, 1], 
               alpha=0.3, s=10, label='CN-Break → CN', c='red')
    plt.scatter(en_data[:, 0], en_data[:, 1], 
               alpha=0.3, s=10, label='EN-Break → EN', c='blue')
    plt.scatter(fr_data[:, 0], fr_data[:, 1], 
               alpha=0.3, s=10, label='FR-Break → FR', c='green')
    
    # 方法名称映射
    method_names = {
        'pca_umap': 'PCA+UMAP',
        'umap': 'UMAP',
        'tsne': 't-SNE'
    }
    method_display = method_names.get(args.method, args.method.upper())
    
    plt.xlabel(f'{method_display} Dimension 1', fontsize=12)
    plt.ylabel(f'{method_display} Dimension 2', fontsize=12)
    plt.title(f'Same-Language Embedding Distribution: {args.llm_family.upper()}\n(Language-Specific Ablation → Same Language, Method: {method_display})', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 已保存: {args.output_file}")

if __name__ == '__main__':
    main()

