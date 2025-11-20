#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化：3种模型 × 3种语言，放在一张图上
- 模型1: qwen原始 (圆形)
- 模型2: Break001 (三角形)
- 模型3: Language-specific same-language (方形)
- 颜色: CN=红, EN=蓝, FR=绿
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_model_language_data(model_name, language, results_dir):
    """加载特定模型的特定语言数据"""
    npz_file = os.path.join(results_dir, f"{model_name}_reduced_2d.npz")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results')
    parser.add_argument('--output_file', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results/all_models_comparison.png')
    parser.add_argument('--max_points', type=int, default=500)
    args = parser.parse_args()
    
    np.random.seed(42)
    
    # 配置
    models = {
        'qwen': 'qwen2.5-7b',
        'break': 'qwen2.5-7b_Break001',
        'cn_specific': 'qwen2.5-7b_CN-specific-Break_rank_top001',
        'en_specific': 'qwen2.5-7b_EN-specific-Break_rank_top001',
        'fr_specific': 'qwen2.5-7b_FR-specific-Break_rank_top001'
    }
    
    languages = ['CN', 'EN', 'FR']
    colors = {'CN': 'red', 'EN': 'blue', 'FR': 'green'}
    markers = {'qwen': 'o', 'break': '^', 'specific': 's'}
    model_names = {
        'qwen': 'Original',
        'break': 'Break001',
        'specific': 'Lang-Specific'
    }
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 模型1: qwen原始
    print("加载 qwen 原始模型...")
    for lang in languages:
        data = load_model_language_data(models['qwen'], lang, args.results_dir)
        data = sample_data(data, args.max_points)
        plt.scatter(data[:, 0], data[:, 1], 
                   c=colors[lang], marker=markers['qwen'],
                   alpha=0.4, s=15, edgecolors='none',
                   label=f'{model_names["qwen"]} - {lang}')
        print(f"  {lang}: {len(data)} samples")
    
    # 模型2: Break001
    print("\n加载 Break001 模型...")
    for lang in languages:
        data = load_model_language_data(models['break'], lang, args.results_dir)
        data = sample_data(data, args.max_points)
        plt.scatter(data[:, 0], data[:, 1], 
                   c=colors[lang], marker=markers['break'],
                   alpha=0.4, s=20, edgecolors='none',
                   label=f'{model_names["break"]} - {lang}')
        print(f"  {lang}: {len(data)} samples")
    
    # 模型3: Language-specific same-language
    print("\n加载 Language-specific 模型（same-language）...")
    for lang in languages:
        if lang == 'CN':
            model_key = 'cn_specific'
        elif lang == 'EN':
            model_key = 'en_specific'
        else:  # FR
            model_key = 'fr_specific'
        
        data = load_model_language_data(models[model_key], lang, args.results_dir)
        data = sample_data(data, args.max_points)
        plt.scatter(data[:, 0], data[:, 1], 
                   c=colors[lang], marker=markers['specific'],
                   alpha=0.4, s=25, edgecolors='none',
                   label=f'{model_names["specific"]} - {lang}')
        print(f"  {lang}: {len(data)} samples")
    
    # 图例和标签
    plt.xlabel('UMAP Dimension 1', fontsize=13)
    plt.ylabel('UMAP Dimension 2', fontsize=13)
    plt.title('Embedding Distribution Comparison\nAcross Models and Languages', 
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

