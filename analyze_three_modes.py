#!/usr/bin/env python3
"""
分析CFdemo_gene_text三个模式的结果
- 模式1: 多模态融合 (ab_model=3)
- 模式2: 仅文本 (ab_model=1)
- 模式3: 仅基因 (ab_model=2)
"""

import os
import pandas as pd
from pathlib import Path

def parse_summary_csv(dir_path):
    """解析summary.csv文件"""
    results_path = Path(dir_path)
    summary_files = list(results_path.glob("**/summary.csv"))

    if not summary_files:
        return None

    summary_file = summary_files[0]
    try:
        df = pd.read_csv(summary_file)
        return {
            'cindex_mean': df['val_cindex'].mean(),
            'cindex_std': df['val_cindex'].std(),
            'ipcw_mean': df['val_cindex_ipcw'].mean(),
            'ipcw_std': df['val_cindex_ipcw'].std(),
            'ibs_mean': df['val_IBS'].mean(),
            'ibs_std': df['val_IBS'].std(),
            'iauc_mean': df['val_iauc'].mean(),
            'iauc_std': df['val_iauc'].std(),
        }
    except Exception as e:
        return None

def main():
    results_dir = "/root/autodl-tmp/newcfdemo/CFdemo_gene_text/results"

    print("="*120)
    print("📊 CFdemo_gene_text 三模式对比分析 (统一学习率)")
    print("="*120)

    # 三个模式的结果目录
    modes = [
        ("多模态融合 (基因+文本)", 1, "case_results_unified_mode1"),
        ("仅文本模式", 2, "case_results_unified_mode2"),
        ("仅基因模式", 3, "case_results_unified_mode3"),
    ]

    results = []
    for mode_name, ab_model, results_dir_name in modes:
        dir_path = os.path.join(results_dir, results_dir_name)
        summary = parse_summary_csv(dir_path)

        if summary:
            results.append({
                'mode': mode_name,
                'ab_model': ab_model,
                'cindex': summary['cindex_mean'],
                'cindex_std': summary['cindex_std'],
                'ipcw': summary['ipcw_mean'],
                'ibs': summary['ibs_mean'],
                'iauc': summary['iauc_mean'],
            })
            print(f"\n✅ {mode_name} (ab_model={ab_model}):")
            print(f"   C-index: {summary['cindex_mean']:.4f} ± {summary['cindex_std']:.4f}")
            print(f"   IPCW:    {summary['ipcw_mean']:.4f} ± {summary['ipcw_std']:.4f}")
            print(f"   IBS:     {summary['ibs_mean']:.4f} ± {summary['ibs_std']:.4f}")
            print(f"   IAUC:    {summary['iauc_mean']:.4f} ± {summary['iauc_std']:.4f}")
        else:
            print(f"\n❌ {mode_name} (ab_model={ab_model}): 结果尚未完成或未找到")

    if not results:
        print("\n⚠️ 所有结果都尚未完成，请稍后重试")
        return

    # 按C-index排序
    results.sort(key=lambda x: x['cindex'], reverse=True)

    # 输出排名表格
    print("\n" + "="*120)
    print("🏆 三模式性能排名 (按C-index排序)")
    print("="*120)
    print(f"{'排名':<4} {'模式':<30} {'C-index':<12} {'±':<8} {'IPCW':<12} {'IBS':<10} {'IAUC':<10}")
    print("-"*120)

    for i, result in enumerate(results, 1):
        print(f"{i:<4} {result['mode']:<30} {result['cindex']:.4f}      ±{result['cindex_std']:.4f} "
              f"{result['ipcw']:.4f}  {result['ibs']:.4f}  {result['iauc']:.4f}")

    # 对比分析
    print("\n" + "="*120)
    print("📈 关键对比分析")
    print("="*120)

    if len(results) >= 2:
        best = results[0]
        print(f"\n🏆 最佳模式: {best['mode']}")
        print(f"   C-index: {best['cindex']:.4f} ± {best['cindex_std']:.4f}")

        # 与其他模式对比
        for result in results[1:]:
            diff = result['cindex'] - best['cindex']
            diff_pct = (diff / best['cindex'] * 100)
            print(f"\n对比 {result['mode']}:")
            print(f"   性能差异: {diff:+.4f} ({diff_pct:+.2f}%)")
            print(f"   {result['cindex']:.4f} vs {best['cindex']:.4f}")

    # 模式间差异分析
    print("\n" + "="*120)
    print("🔍 模式差异分析")
    print("="*120)

    # 多模态 vs 仅基因
    multi = next((r for r in results if '多模态' in r['mode']), None)
    gene = next((r for r in results if '仅基因' in r['mode']), None)
    text = next((r for r in results if '仅文本' in r['mode']), None)

    if multi and gene:
        diff = multi['cindex'] - gene['cindex']
        print(f"\n多模态融合 vs 仅基因:")
        print(f"   性能提升: {diff:+.4f} ({(diff/gene['cindex']*100):+.2f}%)")
        print(f"   {multi['cindex']:.4f} vs {gene['cindex']:.4f}")

    if multi and text:
        diff = multi['cindex'] - text['cindex']
        print(f"\n多模态融合 vs 仅文本:")
        print(f"   性能提升: {diff:+.4f} ({(diff/text['cindex']*100):+.2f}%)")
        print(f"   {multi['cindex']:.4f} vs {text['cindex']:.4f}")

    if gene and text:
        diff = gene['cindex'] - text['cindex']
        print(f"\n仅基因 vs 仅文本:")
        print(f"   性能差异: {diff:+.4f} ({(diff/text['cindex']*100):+.2f}%)")
        print(f"   {gene['cindex']:.4f} vs {text['cindex']:.4f}")

    print("\n" + "="*120)
    print("✅ 三模式对比分析完成")
    print("="*120)

if __name__ == "__main__":
    main()
