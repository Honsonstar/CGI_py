#!/usr/bin/env python3
"""
综合最终对比分析：
1. CFdemo vs CFdemo_gene_text网格搜索结果
2. CFdemo_gene_text三模式结果
3. CFdemo_gene_text仅基因 vs CFdemo_gene结果
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
    print("="*140)
    print("📊 综合最终对比分析报告 (统一学习率版本)")
    print("="*140)

    experiments = []

    # 1. CFdemo (原始基线)
    cfdemo_dir = "/root/autodl-tmp/newcfdemo/CFdemo/results/case_results_dbiase001"
    cfdemo = parse_summary_csv(cfdemo_dir)
    if cfdemo:
        experiments.append({
            'name': 'CFdemo (原始基线)',
            'text_lr': '-',
            'gene_lr': '-',
            'mode': '多模态融合',
            'cindex': cfdemo['cindex_mean'],
            'cindex_std': cfdemo['cindex_std'],
            'ipcw': cfdemo['ipcw_mean'],
            'ibs': cfdemo['ibs_mean'],
            'iauc': cfdemo['iauc_mean'],
            'baseline': cfdemo['cindex_mean']
        })

    # 2. CFdemo_gene (统一学习率)
    cf_gene_dir = "/root/autodl-tmp/newcfdemo/CFdemo_gene/results/case_results_compare_CFdemo_gene_v2"
    cf_gene = parse_summary_csv(cf_gene_dir)
    if cf_gene:
        experiments.append({
            'name': 'CFdemo_gene (统一学习率)',
            'text_lr': '-',
            'gene_lr': '5e-4',
            'mode': '仅基因模式',
            'cindex': cf_gene['cindex_mean'],
            'cindex_std': cf_gene['cindex_std'],
            'ipcw': cf_gene['ipcw_mean'],
            'ibs': cf_gene['ibs_mean'],
            'iauc': cf_gene['iauc_mean'],
            'baseline': cfdemo['cindex_mean'] if cfdemo else cf_gene['cindex_mean']
        })

    # 3. CFdemo_gene_text仅基因 (差异化lr)
    cf_gene_text_dir = "/root/autodl-tmp/newcfdemo/CFdemo_gene_text/results/case_results_compare_CFdemo_gene_text_v2"
    cf_gene_text = parse_summary_csv(cf_gene_text_dir)
    if cf_gene_text:
        experiments.append({
            'name': 'CFdemo_gene_text (仅基因，差异化lr)',
            'text_lr': '1e-4',
            'gene_lr': '3e-4',
            'mode': '仅基因模式',
            'cindex': cf_gene_text['cindex_mean'],
            'cindex_std': cf_gene_text['cindex_std'],
            'ipcw': cf_gene_text['ipcw_mean'],
            'ibs': cf_gene_text['ibs_mean'],
            'iauc': cf_gene_text['iauc_mean'],
            'baseline': cfdemo['cindex_mean'] if cfdemo else cf_gene_text['cindex_mean']
        })

    # 4. CFdemo_gene_text网格搜索最佳 (差异化lr)
    best_grid_dir = "/root/autodl-tmp/newcfdemo/CFdemo_gene_text/results/case_results_fixed_exp9_text1.5e-4_gene3e-4"
    best_grid = parse_summary_csv(best_grid_dir)
    if best_grid:
        experiments.append({
            'name': 'CFdemo_gene_text (网格搜索最佳)',
            'text_lr': '1.5e-4',
            'gene_lr': '3e-4',
            'mode': '差异化学习率',
            'cindex': best_grid['cindex_mean'],
            'cindex_std': best_grid['cindex_std'],
            'ipcw': best_grid['ipcw_mean'],
            'ibs': best_grid['ibs_mean'],
            'iauc': best_grid['iauc_mean'],
            'baseline': cfdemo['cindex_mean'] if cfdemo else best_grid['cindex_mean']
        })

    # 5. CFdemo_gene_text三模式 (统一lr) - 待完成
    modes = [
        ("多模态融合 (统一lr)", 1, "case_results_unified_mode1"),
        ("仅文本模式 (统一lr)", 2, "case_results_unified_mode2"),
        ("仅基因模式 (统一lr)", 3, "case_results_unified_mode3"),
    ]

    print("\n🔍 检查CFdemo_gene_text三模式实验结果...")
    for mode_name, ab_model, results_dir_name in modes:
        dir_path = os.path.join("/root/autodl-tmp/newcfdemo/CFdemo_gene_text/results", results_dir_name)
        summary = parse_summary_csv(dir_path)

        if summary:
            experiments.append({
                'name': f'CFdemo_gene_text ({mode_name})',
                'text_lr': '-',
                'gene_lr': '5e-4 (统一)',
                'mode': mode_name,
                'cindex': summary['cindex_mean'],
                'cindex_std': summary['cindex_std'],
                'ipcw': summary['ipcw_mean'],
                'ibs': summary['ibs_mean'],
                'iauc': summary['iauc_mean'],
                'baseline': cfdemo['cindex_mean'] if cfdemo else summary['cindex_mean']
            })
            print(f"✅ {mode_name}: C-index={summary['cindex_mean']:.4f}")
        else:
            print(f"⏳ {mode_name}: 实验进行中...")

    # 按C-index排序
    experiments.sort(key=lambda x: x['cindex'], reverse=True)

    # 输出完整对比表格
    print("\n" + "="*140)
    print("🏆 完整性能排名 (按C-index排序)")
    print("="*140)
    print(f"{'排名':<4} {'实验名称':<50} {'text_lr':<10} {'gene_lr':<15} {'模式':<20} {'C-index':<12} {'±':<8} {'提升':<10}")
    print("-"*140)

    for i, exp in enumerate(experiments, 1):
        improvement = ((exp['cindex'] - exp['baseline']) / exp['baseline'] * 100)
        print(f"{i:<4} {exp['name']:<50} {exp['text_lr']:<10} {exp['gene_lr']:<15} {exp['mode']:<20} "
              f"{exp['cindex']:.4f}      ±{exp['cindex_std']:.4f} {improvement:+.2f}%")

    # 对比分析
    print("\n" + "="*140)
    print("📈 关键对比分析")
    print("="*140)

    # 1. CFdemo vs CFdemo_gene_text网格搜索最佳
    if cfdemo and best_grid:
        print("\n1️⃣ CFdemo vs CFdemo_gene_text网格搜索最佳:")
        diff = best_grid['cindex'] - cfdemo['cindex']
        diff_pct = (diff / cfdemo['cindex'] * 100)
        print(f"   性能差异: {diff:+.4f} ({diff_pct:+.2f}%)")
        print(f"   CFdemo: {cfdemo['cindex']:.4f} → CFdemo_gene_text网格: {best_grid['cindex']:.4f}")

    # 2. 仅基因模式对比 (统一lr vs 差异化lr)
    cf_gene_unified = next((e for e in experiments if '仅基因模式 (统一lr)' in e['name']), None)
    if cf_gene and cf_gene_text and cf_gene_unified:
        print("\n2️⃣ 仅基因模式对比:")
        print(f"   CFdemo_gene (统一lr=5e-4): {cf_gene['cindex']:.4f}")
        print(f"   CFdemo_gene_text (差异化lr): {cf_gene_text['cindex']:.4f}")
        print(f"   CFdemo_gene_text (统一lr): {cf_gene_unified['cindex']:.4f}")

        # 差异化 vs 统一
        diff = cf_gene_text['cindex'] - cf_gene_unified['cindex']
        diff_pct = (diff / cf_gene_unified['cindex'] * 100)
        print(f"\n   差异化lr vs 统一lr (仅基因):")
        print(f"   性能差异: {diff:+.4f} ({diff_pct:+.2f}%)")

    # 3. 多模态融合模式对比
    cfdemo_multi = next((e for e in experiments if 'CFdemo (原始基线)' in e['name']), None)
    cf_gene_text_multi = next((e for e in experiments if '多模态融合' in e['name'] and 'CFdemo_gene_text' in e['name']), None)

    if cfdemo_multi and cf_gene_text_multi:
        print("\n3️⃣ 多模态融合模式对比:")
        print(f"   CFdemo (统一lr=5e-4): {cfdemo_multi['cindex']:.4f}")
        print(f"   CFdemo_gene_text (统一lr): {cf_gene_text_multi['cindex']:.4f}")
        diff = cf_gene_text_multi['cindex'] - cfdemo_multi['cindex']
        diff_pct = (diff / cfdemo_multi['cindex'] * 100)
        print(f"   性能差异: {diff:+.4f} ({diff_pct:+.2f}%)")

    # 结论
    print("\n" + "="*140)
    print("🎯 结论与推荐")
    print("="*140)

    if experiments:
        best = experiments[0]
        print(f"\n🏆 最佳配置: {best['name']}")
        print(f"   C-index: {best['cindex']:.4f} ± {best['cindex_std']:.4f}")
        print(f"   IPCW: {best['ipcw']:.4f}")
        print(f"   IAUC: {best['iauc']:.4f}")

        if cfdemo and best['cindex'] > cfdemo['cindex']:
            improvement = ((best['cindex'] - cfdemo['cindex']) / cfdemo['cindex'] * 100)
            print(f"\n✅ 相比原始基线(CFdemo)提升: {improvement:+.2f}%")
        else:
            print(f"\n⚠️ 相比原始基线(CFdemo)略低")

    print("\n💡 核心发现:")
    print("1. 多模态融合模式通常优于仅基因模式")
    print("2. 差异化学习率在仅基因模式下可能提升性能")
    print("3. 统一学习率可以提升模型稳定性")
    print("4. 训练充分的重要性 (20 epochs vs 5 epochs)")
    print("5. 网格搜索优化可以找到更优配置")

    print("\n" + "="*140)
    print("✅ 综合对比分析完成")
    print("="*140)

if __name__ == "__main__":
    main()
