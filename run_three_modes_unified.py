#!/usr/bin/env python3
"""
并行运行CFdemo_gene_text的三个模式
- 模式1: 基因+文本 (多模态融合, ab_model=3)
- 模式2: 仅文本 (ab_model=1)
- 模式3: 仅基因 (ab_model=2)
使用统一学习率 (lr=5e-4)
"""

import os
import subprocess
import time

def run_mode(mode_name, ab_model, results_dir):
    """运行单个模式"""
    print("="*80)
    print(f"🚀 启动 {mode_name} (ab_model={ab_model})")
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
        "--results_dir", results_dir,
        "--batch_size", "1",
        "--lr", "0.0005",  # 统一学习率
        "--opt", "radam",
        "--reg", "0.0001",
        "--alpha_surv", "0.5",
        "--weighted_sample",
        "--max_epochs", "20",
        "--label_col", "survival_months_dss",
        "--k", "5",
        "--bag_loss", "nll_surv",
        "--type_of_path", "custom",
        "--modality", "snn",
        "--enable_multitask",
        "--multitask_weight", "0.12",
        "--ab_model", str(ab_model)  # 运行模式
    ]

    log_file = f"logs/mode_{ab_model}_{results_dir}.log"
    os.makedirs("logs", exist_ok=True)

    with open(log_file, 'w') as f:
        f.write(f"{mode_name} 实验 (ab_model={ab_model}, 统一学习率 lr=5e-4)\n")
        f.write(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")

    process = subprocess.Popen(
        cmd,
        stdout=open(log_file, 'a'),
        stderr=subprocess.STDOUT,
        cwd="/root/autodl-tmp/newcfdemo/CFdemo_gene_text"
    )

    return process, mode_name, log_file

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 CFdemo_gene_text 三模式并行实验")
    print("   (使用统一学习率 lr=5e-4)")
    print("="*80)

    # 启动三个模式
    processes = []
    modes = [
        ("多模态融合 (基因+文本)", 3, "case_results_unified_mode1"),
        ("仅文本模式", 1, "case_results_unified_mode2"),
        ("仅基因模式", 2, "case_results_unified_mode3"),
    ]

    for mode_name, ab_model, results_dir in modes:
        process, name, log = run_mode(mode_name, ab_model, results_dir)
        processes.append((process, name, log))
        print(f"✅ 已启动: {name} (PID: {process.pid})")
        print(f"   日志文件: {log}")
        print(f"   结果目录: {results_dir}")
        print()

    print("\n" + "="*80)
    print("⏳ 等待所有模式完成...")
    print("="*80)
    print(f"预计耗时: ~12-15分钟 (每个模式 20 epochs x 5折)")

    # 等待所有进程完成
    for process, name, log in processes:
        try:
            process.wait(timeout=3600)  # 1小时超时
            print(f"✅ {name} 已完成")
        except subprocess.TimeoutExpired:
            print(f"⏰ {name} 超时")

    print("\n" + "="*80)
    print("✅ 所有模式实验完成")
    print("="*80)
    print("\n📝 日志文件:")
    for _, name, log in processes:
        print(f"   - {name}: {log}")

    print("\n🔍 结果目录:")
    for mode_name, ab_model, results_dir in modes:
        print(f"   - {mode_name}: {results_dir}")

if __name__ == "__main__":
    main()
