#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化：CN破坏→CN, EN破坏→EN, FR破坏→FR
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_language_data(model_name, language, results_dir):
    """加载特定模型的特定语言数据"""
    npz_file = os.path.join(results_dir, f"{model_name}_reduced_2d.npz")
    data = np.load(npz_file, allow_pickle=True)
    
    # 找到对应语言的索引
    languages = data['language_labels']
    lang_mask = (languages == language)
    
    return data['embeddings_2d'][lang_mask]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results')
    parser.add_argument('--output_file', type=str, 
                       default='/leonardo_work/EUHPC_B24_036/yang/embedding_distribution/results/same_language_comparison.png')
    args = parser.parse_args()
    
    # 加载数据
    cn_data = load_language_data('qwen2.5-7b_CN-specific-Break_rank_top001', 'CN', args.results_dir)
    en_data = load_language_data('qwen2.5-7b_EN-specific-Break_rank_top001', 'EN', args.results_dir)
    fr_data = load_language_data('qwen2.5-7b_FR-specific-Break_rank_top001', 'FR', args.results_dir)
    
    print(f"CN破坏→CN: {len(cn_data)} samples")
    print(f"EN破坏→EN: {len(en_data)} samples")
    print(f"FR破坏→FR: {len(fr_data)} samples")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 采样以加快绘制
    max_points = 5000
    
    if len(cn_data) > max_points:
        idx = np.random.choice(len(cn_data), max_points, replace=False)
        cn_data = cn_data[idx]
    if len(en_data) > max_points:
        idx = np.random.choice(len(en_data), max_points, replace=False)
        en_data = en_data[idx]
    if len(fr_data) > max_points:
        idx = np.random.choice(len(fr_data), max_points, replace=False)
        fr_data = fr_data[idx]
    
    plt.scatter(cn_data[:, 0], cn_data[:, 1], 
               alpha=0.3, s=10, label='CN-Break → CN', c='red')
    plt.scatter(en_data[:, 0], en_data[:, 1], 
               alpha=0.3, s=10, label='EN-Break → EN', c='blue')
    plt.scatter(fr_data[:, 0], fr_data[:, 1], 
               alpha=0.3, s=10, label='FR-Break → FR', c='green')
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Same-Language Embedding Distribution\n(Language-Specific Ablation → Same Language)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 已保存: {args.output_file}")

if __name__ == '__main__':
    main()

