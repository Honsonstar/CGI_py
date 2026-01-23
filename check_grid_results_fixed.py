#!/usr/bin/env python3
"""
检查CFdemo_gene_text网格搜索实验结果
修正版：正确查找所有exp的summary.csv
"""

import os
import pandas as pd
from pathlib import Path
import re

def extract_lr_from_dirname(dirname):
    """从目录名提取text_lr和gene_lr"""
    match = re.search(r'text([0-9.]+e-[0-9]+)_gene([0-9.]+e-[0-9]+)', dirname)
    if match:
        text_lr = match.group(1)
        gene_lr = match.group(2)
        return text_lr, gene_lr
    return None, None

def parse_summary_csv(dir_path):
    """解析summary.csv文件"""
    # 在结果目录的子目录中查找summary.csv
    results_path = Path(dir_path)
    summary_files = list(results_path.glob("**/summary.csv"))

    if not summary_files:
        return None

    summary_file = summary_files[0]
    try:
        df = pd.read_csv(summary_file)
        cindex_mean = df['val_cindex'].mean()
        cindex_std = df['val_cindex'].std()
        return {
            'cindex_mean': cindex_mean,
            'cindex_std': cindex_std,
            'file': str(summary_file)
        }
    except Exception as e:
        print(f"Error parsing {summary_file}: {e}")
        return None

def main():
    results_dir = "/root/autodl-tmp/newcfdemo/CFdemo_gene_text/results"

    # 查找所有exp实验
    exp_dirs = []
    for dirname in os.listdir(results_dir):
        if "gusion_exp" in dirname:
            exp_num_match = re.search(r'gusion_exp(\d+)', dirname)
            if exp_num_match:
                exp_num = int(exp_num_match.group(1))
                text_lr, gene_lr = extract_lr_from_dirname(dirname)
                exp_dirs.append((exp_num, dirname, text_lr, gene_lr))

    # 按exp_num排序
    exp_dirs.sort(key=lambda x: x[0])

    # 解析结果
    results = []
    for exp_num, dirname, text_lr, gene_lr in exp_dirs:
        dir_path = os.path.join(results_dir, dirname)
        summary = parse_summary_csv(dir_path)

        if summary:
            results.append({
                'exp': exp_num,
                'dirname': dirname,
                'text_lr': text_lr,
                'gene_lr': gene_lr,
                'cindex_mean': summary['cindex_mean'],
                'cindex_std': summary['cindex_std'],
                'summary_file': summary['file']
            })
        else:
            print(f"❌ exp{exp_num}: 没有找到summary.csv")

    # 按C-index排序
    results.sort(key=lambda x: x['cindex_mean'], reverse=True)

    # 输出排名
    print("="*100)
    print("CFdemo_gene_text 网格搜索实验结果排名 (按C-index排序)")
    print("="*100)
    print(f"{'排名':<4} {'实验':<6} {'text_lr':<12} {'gene_lr':<12} {'C-index':<10} {'±':<8}")
    print("-"*100)

    for i, result in enumerate(results, 1):
        print(f"{i:<4} {result['exp']:<6} {result['text_lr']:<12} {result['gene_lr']:<12} "
              f"{result['cindex_mean']:.4f}    ±{result['cindex_std']:.4f}")

    # 特别标注exp2-4
    print("\n" + "="*100)
    print("exp2-4 详细信息:")
    print("="*100)
    exp2_4_results = [r for r in results if r['exp'] in [2, 3, 4]]
    if exp2_4_results:
        for result in exp2_4_results:
            rank = next(i for i, r in enumerate(results, 1) if r['exp'] == result['exp'])
            print(f"\nexp{result['exp']}:")
            print(f"  配置: text_lr={result['text_lr']}, gene_lr={result['gene_lr']}")
            print(f"  C-index: {result['cindex_mean']:.4f} ± {result['cindex_std']:.4f}")
            print(f"  排名: 第{rank}名 (共{len(results)}个实验)")
    else:
        print("❌ exp2-4 还没有完成，没有找到summary.csv文件")

    # 检查配置
    print("\n" + "="*100)
    print("实验配置检查 (检查前5个实验的epochs):")
    print("="*100)
    for result in results[:5]:
        print(f"\nexp{result['exp']} (text_lr={result['text_lr']}, gene_lr={result['gene_lr']}):")
        print(f"  目录名: {result['dirname']}")
        if 'epochs_20' in result['dirname']:
            print(f"  ✅ epochs: 20")
        else:
            print(f"  ❓ epochs: 未在目录名中找到")

    # 总结
    print("\n" + "="*100)
    print("总结:")
    print("="*100)
    print(f"总共找到 {len(results)} 个完成的实验")
    print(f"exp2-4 中完成实验数: {len(exp2_4_results)}")
    if exp2_4_results:
        best_exp = max(exp2_4_results, key=lambda x: x['cindex_mean'])
        print(f"exp2-4 中最佳配置: exp{best_exp['exp']} (C-index={best_exp['cindex_mean']:.4f})")

if __name__ == "__main__":
    main()
