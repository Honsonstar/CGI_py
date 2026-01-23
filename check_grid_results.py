#!/usr/bin/env python3
"""
检查CFdemo_gene_text网格搜索实验结果
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
    summary_files = list(Path(dir_path).glob("**/summary.csv"))
    if not summary_files:
        return None

    summary_file = summary_files[0]
    try:
        df = pd.read_csv(summary_file)
        cindex_mean = df['val_cindex'].mean()
        cindex_std = df['val_cindex'].std()
        return {
            'cindex_mean': cindex_mean,
            'cindex_std': cindex_std
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
                'cindex_std': summary['cindex_std']
            })

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
    for result in results:
        if result['exp'] in [2, 3, 4]:
            rank = next(i for i, r in enumerate(results, 1) if r['exp'] == result['exp'])
            print(f"\nexp{result['exp']}:")
            print(f"  配置: text_lr={result['text_lr']}, gene_lr={result['gene_lr']}")
            print(f"  C-index: {result['cindex_mean']:.4f} ± {result['cindex_std']:.4f}")
            print(f"  排名: 第{rank}名 (共{len(results)}个实验)")

    # 检查epochs配置
    print("\n" + "="*100)
    print("检查实验配置:")
    print("="*100)
    for result in results[:5]:  # 只检查前5个
        dir_path = os.path.join(results_dir, result['dirname'])
        log_files = list(Path(dir_path).glob("**/*.log"))
        if log_files:
            log_file = log_files[0]
            with open(log_file, 'r') as f:
                content = f.read()
                if 'max_epochs' in content or 'epochs' in content:
                    for line in content.split('\n'):
                        if 'max_epochs' in line.lower() or '--epochs' in line:
                            print(f"exp{result['exp']}: {line.strip()}")
                            break

if __name__ == "__main__":
    main()
