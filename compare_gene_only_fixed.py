#!/usr/bin/env python3
"""
比较CFdemo_gene和CFdemo_gene_text两个项目的仅基因模式差异
并行运行两个项目，对比结果

修复说明:
- 使用正确的max_epochs=20 (与SNN.sh一致)
- type_of_path=custom (与SNN.sh一致)
"""

import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

def run_gene_project():
    """运行CFdemo_gene项目的仅基因模式"""
    print("="*80)
    print("🚀 启动 CFdemo_gene (仅基因模式)")
    print("   配置: max_epochs=20, type_of_path=custom")
    print("="*80)

    cmd = [
        "conda", "run", "-n", "causal", "python", "main.py",
        "--label_file", "datasets_csv/clinical_data/tcga_brca_clinical.csv",
        "--study", "tcga_brca",
        "--split_dir", "splits",
        "--data_root_dir", "datasets_csv/reports_clean",
        "--task", "survival",
        "--which_splits", "5foldcv_ramdom",
        "--omics_dir", "preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv",
        "--results_dir", "case_results_compare_CFdemo_gene_v2",
        "--batch_size", "1",
        "--lr", "0.0005",
        "--opt", "radam",
        "--reg", "0.0001",
        "--alpha_surv", "0.5",
        "--weighted_sample",
        "--max_epochs", "20",  # 修复: 使用20 epochs，与SNN.sh一致
        "--label_col", "survival_months_dss",
        "--k", "5",
        "--bag_loss", "nll_surv",
        "--type_of_path", "custom",
        "--modality", "snn",
        "--enable_multitask",
        "--multitask_weight", "0.12"
    ]

    log_file = "logs/compare_CFdemo_gene_v2.log"
    os.makedirs("logs", exist_ok=True)

    with open(log_file, 'w') as f:
        f.write(f"CFdemo_gene 仅基因模式测试 (20 epochs)\n")
        f.write(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    process = subprocess.Popen(
        cmd,
        stdout=open(log_file, 'a'),
        stderr=subprocess.STDOUT,
        cwd="/root/autodl-tmp/newcfdemo/CFdemo_gene"
    )

    return process, "CFdemo_gene", log_file

def run_gene_text_project():
    """运行CFdemo_gene_text项目的仅基因模式"""
    print("="*80)
    print("🚀 启动 CFdemo_gene_text (仅基因模式)")
    print("   配置: max_epochs=20, type_of_path=custom")
    print("   差异化学习率: text_lr=1e-4, gene_lr=3e-4")
    print("="*80)

    cmd = [
        "conda", "run", "-n", "causal", "python", "main.py",
        "--label_file", "datasets_csv/clinical_data/tcga_brca_clinical.csv",
        "--study", "tcga_brca",
        "--split_dir", "splits",
        "--data_root_dir", "datasets_csv/reports_clean",
        "--task", "survival",
        "--which_splits", "5foldcv_ramdom",
        "--omics_dir", "preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv",
        "--results_dir", "case_results_compare_CFdemo_gene_text_v2",
        "--batch_size", "1",
        "--lr", "0.0005",
        "--text_lr", "0.0001",
        "--gene_lr", "0.0003",
        "--ab_model", "2",  # 仅基因模式
        "--opt", "radam",
        "--reg", "0.0001",
        "--alpha_surv", "0.5",
        "--weighted_sample",
        "--max_epochs", "20",  # 修复: 使用20 epochs，与SNN.sh一致
        "--label_col", "survival_months_dss",
        "--k", "5",
        "--bag_loss", "nll_surv",
        "--type_of_path", "custom",
        "--modality", "snn",
        "--enable_multitask",
        "--multitask_weight", "0.12"
    ]

    log_file = "logs/compare_CFdemo_gene_text_v2.log"
    os.makedirs("logs", exist_ok=True)

    with open(log_file, 'w') as f:
        f.write(f"CFdemo_gene_text 仅基因模式测试 (20 epochs)\n")
        f.write(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    process = subprocess.Popen(
        cmd,
        stdout=open(log_file, 'a'),
        stderr=subprocess.STDOUT,
        cwd="/root/autodl-tmp/newcfdemo/CFdemo_gene_text"
    )

    return process, "CFdemo_gene_text", log_file

def wait_for_completion(process, name, timeout=3600):
    """等待进程完成"""
    print(f"\n⏳ 等待 {name} 完成...")
    try:
        process.wait(timeout=timeout)
        print(f"✅ {name} 已完成")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ {name} 超时")
        return False

def parse_results(project_name, results_dir):
    """解析结果"""
    # 结果保存在results子目录下
    results_path = Path(f"/root/autodl-tmp/newcfdemo/{project_name}/results/{results_dir}")

    if not results_path.exists():
        print(f"❌ {project_name}: 结果目录不存在: {results_path}")
        return None

    # 查找summary.csv
    summary_files = list(results_path.glob("**/summary.csv"))

    if not summary_files:
        print(f"❌ {project_name}: 未找到summary.csv")
        return None

    summary_file = summary_files[0]

    try:
        df = pd.read_csv(summary_file)

        cindex_mean = df['val_cindex'].mean()
        cindex_std = df['val_cindex'].std()
        ipcw_mean = df['val_cindex_ipcw'].mean()
        ipcw_std = df['val_cindex_ipcw'].std()
        ibs_mean = df['val_IBS'].mean()
        ibs_std = df['val_IBS'].std()
        iauc_mean = df['val_iauc'].mean()
        iauc_std = df['val_iauc'].std()

        return {
            'project': project_name,
            'cindex_mean': cindex_mean,
            'cindex_std': cindex_std,
            'ipcw_mean': ipcw_mean,
            'ipcw_std': ipcw_std,
            'ibs_mean': ibs_mean,
            'ibs_std': ibs_std,
            'iauc_mean': iauc_mean,
            'iauc_std': iauc_std
        }
    except Exception as e:
        print(f"❌ {project_name}: 解析结果失败 - {e}")
        return None

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 比较 CFdemo_gene 和 CFdemo_gene_text 的仅基因模式")
    print("   (修复版本: max_epochs=20)")
    print("="*80)

    # 启动两个项目
    process1, name1, log1 = run_gene_project()
    process2, name2, log2 = run_gene_text_project()

    print(f"\n✅ 已启动两个项目:")
    print(f"   1. {name1} (PID: {process1.pid})")
    print(f"   2. {name2} (PID: {process2.pid})")
    print(f"\n📝 日志文件:")
    print(f"   1. {log1}")
    print(f"   2. {log2}")
    print(f"\n⏱️  预计耗时: ~12-15分钟 (20 epochs x 5折)")

    # 等待完成
    wait_for_completion(process1, name1)
    wait_for_completion(process2, name2)

    # 解析结果
    print("\n" + "="*80)
    print("📊 解析结果")
    print("="*80)

    result1 = parse_results("CFdemo_gene", "case_results_compare_CFdemo_gene_v2")
    result2 = parse_results("CFdemo_gene_text", "case_results_compare_CFdemo_gene_text_v2")

    # 显示结果
    if result1 and result2:
        print("\n" + "="*80)
        print("📈 结果对比 (20 epochs)")
        print("="*80)

        print(f"\n{result1['project']}:")
        print(f"   C-index: {result1['cindex_mean']:.4f} ± {result1['cindex_std']:.4f}")
        print(f"   IPCW:    {result1['ipcw_mean']:.4f} ± {result1['ipcw_std']:.4f}")
        print(f"   IBS:     {result1['ibs_mean']:.4f} ± {result1['ibs_std']:.4f}")
        print(f"   IAUC:    {result1['iauc_mean']:.4f} ± {result1['iauc_std']:.4f}")

        print(f"\n{result2['project']}:")
        print(f"   C-index: {result2['cindex_mean']:.4f} ± {result2['cindex_std']:.4f}")
        print(f"   IPCW:    {result2['ipcw_mean']:.4f} ± {result2['ipcw_std']:.4f}")
        print(f"   IBS:     {result2['ibs_mean']:.4f} ± {result2['ibs_std']:.4f}")
        print(f"   IAUC:    {result2['iauc_mean']:.4f} ± {result2['iauc_std']:.4f}")

        # 计算差异
        print("\n" + "="*80)
        print("🔍 差异分析")
        print("="*80)

        cindex_diff = result2['cindex_mean'] - result1['cindex_mean']
        ipcw_diff = result2['ipcw_mean'] - result1['ipcw_mean']
        ibs_diff = result2['ibs_mean'] - result1['ibs_mean']
        iauc_diff = result2['iauc_mean'] - result1['iauc_mean']

        print(f"\nCFdemo_gene_text - CFdemo_gene:")
        print(f"   C-index: {cindex_diff:+.4f} ({cindex_diff/result1['cindex_mean']*100:+.2f}%)")
        print(f"   IPCW:    {ipcw_diff:+.4f} ({ipcw_diff/result1['ipcw_mean']*100:+.2f}%)")
        print(f"   IBS:     {ibs_diff:+.4f}")
        print(f"   IAUC:    {iauc_diff:+.4f} ({iauc_diff/result1['iauc_mean']*100:+.2f}%)")

        # 判断哪个更好
        if cindex_diff > 0:
            print(f"\n✅ CFdemo_gene_text 的 C-index 更高 (+{cindex_diff:.4f})")
        else:
            print(f"\n❌ CFdemo_gene_text 的 C-index 更低 ({cindex_diff:.4f})")

        if iauc_diff > 0:
            print(f"✅ CFdemo_gene_text 的 IAUC 更高 (+{iauc_diff:.4f})")
        else:
            print(f"❌ CFdemo_gene_text 的 IAUC 更低 ({iauc_diff:.4f})")

        print("\n" + "="*80)
        print("📋 配置对比")
        print("="*80)
        print("CFdemo_gene:      统一学习率 lr=0.0005")
        print("CFdemo_gene_text: 差异化学习率 text_lr=0.0001, gene_lr=0.0003")
        print("="*80)

    print("\n✅ 测试完成")
    print("="*80)

if __name__ == "__main__":
    main()
