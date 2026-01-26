#!/usr/bin/env python3
"""
完整的嵌套CPCG流程实现
为每折独立运行Stage1（参数化和半参数化）和Stage2
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import subprocess
import shutil

# 确保在正确的目录
os.chdir('/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy')

def setup_fold_data(study, fold, train_ids, work_dir):
    """为折准备数据"""
    print(f"\n{'='*70}")
    print(f"准备 Fold {fold} 数据")
    print(f"{'='*70}")

    fold_work_dir = f'{work_dir}/fold_{fold}'
    os.makedirs(fold_work_dir, exist_ok=True)

    # 读取原始数据
    clinical_file = f'preprocessing/CPCG_algo/raw_data/tcga_{study}/clinical.CSV'
    exp_file = f'preprocessing/CPCG_algo/raw_data/tcga_{study}/data.csv'

    clinical = pd.read_csv(clinical_file)
    exp = pd.read_csv(exp_file)

    print(f"   原始临床数据: {clinical.shape}")
    print(f"   原始表达数据: {exp.shape}")
    print(f"   训练样本数: {len(train_ids)}")

    # 筛选训练样本
    train_clinical = clinical[clinical['case_submitter_id'].isin(train_ids)].copy()
    print(f"   筛选后临床: {train_clinical.shape}")

    # 保存临床文件
    clinical_out = f'{fold_work_dir}/clinical.CSV'
    train_clinical.to_csv(clinical_out, index=False)

    # 准备表达数据
    gene_names = exp['gene_name'].values
    sample_ids = exp.columns[1:].tolist()
    expression_matrix = exp.iloc[:, 1:].values

    exp_df = pd.DataFrame(
        expression_matrix.T,
        index=sample_ids,
        columns=gene_names
    )

    # 筛选训练样本
    train_exp = exp_df.loc[exp_df.index.intersection(train_ids)].copy()
    print(f"   筛选后表达: {train_exp.shape}")

    # 转置回格式（基因 x 样本）
    train_exp_t = train_exp.T.reset_index()
    train_exp_t.rename(columns={'index': 'gene_name'}, inplace=True)

    exp_out = f'{fold_work_dir}/data.csv'
    train_exp_t.to_csv(exp_out, index=False)

    print(f"   ✅ 数据准备完成: {fold_work_dir}")

    return fold_work_dir

def run_stage1_parametric(study, fold, fold_work_dir):
    """运行Stage1参数化模型"""
    print(f"\n{'='*70}")
    print(f"Stage1 Parametric - Fold {fold}")
    print(f"{'='*70}")

    # 创建输出目录
    output_dir = f'preprocessing/CPCG_algo/Stage1_parametric_model/parametric_result_20250916_n100/tcga_{study}_fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"   输出目录: {output_dir}")

    # 切换到Stage1目录
    stage1_dir = 'preprocessing/CPCG_algo/Stage1_parametric_model'

    # 准备参数
    clinical_file = f'{fold_work_dir}/clinical.CSV'
    exp_file = f'{fold_work_dir}/data.csv'

    print(f"   输入临床: {clinical_file}")
    print(f"   输入表达: {exp_file}")
    print(f"   运行参数化模型...")

    # 运行参数化筛选
    # Stage1需要clinical_final.CSV和data.csv
    clinical_final = pd.read_csv(clinical_file)
    clinical_final.to_csv(f'{stage1_dir}/parametric_input_clinical.CSV', index=False)

    shutil.copy2(exp_file, f'{stage1_dir}/parametric_input_data.csv')

    # 使用screen.py
    try:
        # 需要在Stage1目录运行
        os.chdir(stage1_dir)
        result = subprocess.run([
            'python', 'screen.py',
            '--clinical', 'parametric_input_clinical.CSV',
            '--exp', 'parametric_input_data.csv',
            '--output', output_dir,
            '--h_type', 'OS'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"   ✅ Stage1参数化完成")
            return output_dir
        else:
            print(f"   ❌ Stage1参数化失败: {result.stderr}")
            return None
    except Exception as e:
        print(f"   ❌ Stage1参数化出错: {str(e)}")
        return None
    finally:
        os.chdir('/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy')

def run_stage1_semi_parametric(study, fold, fold_work_dir):
    """运行Stage1半参数化模型"""
    print(f"\n{'='*70}")
    print(f"Stage1 Semi-Parametric - Fold {fold}")
    print(f"{'='*70}")

    # 创建输出目录
    output_dir = f'preprocessing/CPCG_algo/Stage1_semi_parametric_model/semi-parametric_result_20250916_n100/tcga_{study}_fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"   输出目录: {output_dir}")

    # 切换到Stage1目录
    stage1_dir = 'preprocessing/CPCG_algo/Stage1_semi_parametric_model'

    # 准备数据
    clinical_file = f'{fold_work_dir}/clinical.CSV'
    exp_file = f'{fold_work_dir}/data.csv'

    print(f"   输入临床: {clinical_file}")
    print(f"   输入表达: {exp_file}")
    print(f"   运行半参数化模型...")

    try:
        # 需要创建clinical_final.CSV - 在切换目录前读取和准备文件
        clinical_final = pd.read_csv(clinical_file)
        exp_file_abs = os.path.abspath(exp_file)  # 获取绝对路径

        # 切换到semi-parametric目录
        os.chdir(stage1_dir)

        # 写入文件（使用绝对路径确保正确）
        clinical_final.to_csv('semi_input_clinical.CSV', index=False)
        shutil.copy2(exp_file_abs, 'semi_input_data.csv')

        # 运行半参数化筛选
        result = subprocess.run([
            'python', 'screen.py',
            '--clinical', 'semi_input_clinical.CSV',
            '--exp', 'semi_input_data.csv',
            '--output', output_dir,
            '--h_type', 'OS'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"   ✅ Stage1半参数化完成")
            return output_dir
        else:
            print(f"   ❌ Stage1半参数化失败: {result.stderr}")
            return None
    except Exception as e:
        print(f"   ❌ Stage1半参数化出错: {str(e)}")
        return None
    finally:
        os.chdir('/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy')

def run_stage2(study, fold, para_dir, semi_dir):
    """运行Stage2"""
    print(f"\n{'='*70}")
    print(f"Stage2 - Fold {fold}")
    print(f"{'='*70}")

    if not para_dir or not semi_dir:
        print(f"   ❌ Stage1结果不存在，跳过Stage2")
        return None

    # 创建输出目录
    output_dir = f'preprocessing/CPCG_algo/Stage2/result_m2m3_base_0916_n100/tcga_{study}_fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"   输出目录: {output_dir}")

    # 准备目录结构
    # Stage2需要从para_result和semi_result读取
    para_source = f'preprocessing/CPCG_algo/Stage1_parametric_model/parametric_result_20250916_n100'
    semi_source = f'preprocessing/CPCG_algo/Stage1_semi_parametric_model/semi-parametric_result_20250916_n100'

    # 创建符号链接或复制数据
    para_link = f'preprocessing/CPCG_algo/raw_data/para_result/tcga_{study}_fold_{fold}'
    semi_link = f'preprocessing/CPCG_algo/raw_data/semi_result/tcga_{study}_fold_{fold}'

    os.makedirs(f'preprocessing/CPCG_algo/raw_data/para_result', exist_ok=True)
    os.makedirs(f'preprocessing/CPCG_algo/raw_data/semi_result', exist_ok=True)

    # 复制数据到标准位置
    if os.path.exists(f'{para_dir}/result.csv'):
        os.makedirs(para_link, exist_ok=True)
        shutil.copy2(f'{para_dir}/result.csv', f'{para_link}/result.csv')

    if os.path.exists(f'{semi_dir}/result.csv'):
        os.makedirs(semi_link, exist_ok=True)
        shutil.copy2(f'{semi_dir}/result.csv', f'{semi_link}/result.csv')

    # 运行Stage2
    stage2_dir = 'preprocessing/CPCG_algo/Stage2'

    try:
        os.chdir(stage2_dir)
        result = subprocess.run([
            'python', 'main.py'
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"   ✅ Stage2完成")
            return output_dir
        else:
            print(f"   ❌ Stage2失败: {result.stderr}")
            return None
    except Exception as e:
        print(f"   ❌ Stage2出错: {str(e)}")
        return None
    finally:
        os.chdir('/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy')

def run_complete_nested_cpcg(study):
    """运行完整的嵌套CPCG"""
    print(f"\n{'='*80}")
    print(f"运行完整嵌套CPCG流程: {study}")
    print(f"{'='*80}")

    # 1. 创建嵌套划分
    clinical_file = f'datasets_csv/clinical_data/tcga_{study}_clinical.csv'
    splits_dir = f'splits/nested_cv/{study}'
    work_dir = f'work/cpcg_nested/{study}'

    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n📊 创建嵌套划分...")
    clinical = pd.read_csv(clinical_file)
    clinical = clinical.dropna(subset=['case_id', 'censorship'])

    sample_ids = clinical['case_id'].values
    labels = clinical['censorship'].values

    print(f"   有效样本数: {len(sample_ids)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_splits = []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(sample_ids, labels)):
        train_val_ids = sample_ids[train_val_idx]
        test_ids = sample_ids[test_idx]
        train_val_labels = labels[train_val_idx]

        # 划分训练/验证
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_ids)),
            test_size=0.15,
            stratify=train_val_labels,
            random_state=42
        )

        train_ids = train_val_ids[train_idx]
        val_ids = train_val_ids[val_idx]

        # 保存划分
        max_len = max(len(train_ids), len(val_ids), len(test_ids))
        split_df = pd.DataFrame({
            'train': list(train_ids) + [''] * (max_len - len(train_ids)),
            'val': list(val_ids) + [''] * (max_len - len(val_ids)),
            'test': list(test_ids) + [''] * (max_len - len(test_ids))
        })

        split_file = f'{splits_dir}/nested_splits_{fold}.csv'
        split_df.to_csv(split_file, index=False)

        fold_splits.append({
            'fold': fold,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'test_ids': test_ids
        })

        print(f"   Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 2. 为每折运行完整的CPCG
    print(f"\n🧬 开始每折完整CPCG筛选...")
    print(f"{'='*80}")

    all_results = []

    for fold_info in fold_splits:
        fold = fold_info['fold']
        train_ids = fold_info['train_ids']

        print(f"\n{'='*80}")
        print(f"处理 Fold {fold}")
        print(f"{'='*80}")

        # 准备数据
        fold_work_dir = setup_fold_data(study, fold, train_ids, work_dir)

        # 运行Stage1参数化
        para_dir = run_stage1_parametric(study, fold, fold_work_dir)

        # 运行Stage1半参数化
        semi_dir = run_stage1_semi_parametric(study, fold, fold_work_dir)

        # 运行Stage2
        stage2_dir = run_stage2(study, fold, para_dir, semi_dir)

        if stage2_dir:
            print(f"   ✅ Fold {fold} 完整CPCG完成")
            all_results.append({
                'fold': fold,
                'stage2_dir': stage2_dir,
                'train_size': len(train_ids)
            })
        else:
            print(f"   ❌ Fold {fold} CPCG失败")

    print(f"\n{'='*80}")
    print(f"✅ 所有折CPCG完成!")
    print(f"{'='*80}")

    return splits_dir, all_results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python run_complete_nested_cpcg.py <study>")
        print("示例: python run_complete_nested_cpcg.py brca")
        sys.exit(1)

    study = sys.argv[1]
    splits_dir, results = run_complete_nested_cpcg(study)

    print(f"\n📁 输出目录:")
    print(f"   划分: {splits_dir}")
    print(f"   CPCG结果:")
    for r in results:
        print(f"      Fold {r['fold']}: {r['stage2_dir']}")

    print(f"\n🎯 接下来:")
    print(f"   整合CPCG结果")
    print(f"   运行嵌套CV训练")
