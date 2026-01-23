import pandas as pd
import argparse
import os
import sys

def merge_data(study):
    # 1. 定义文件路径
    base_dir = "/root/autodl-tmp/newcfdemo/CFdemo/datasets_csv"
    clinical_path = os.path.join(base_dir, "clinical_data", f"tcga_{study}_clinical.csv")
    metadata_path = os.path.join(base_dir, "metadata", f"tcga_{study}.csv")

    print(f"[*] 正在处理 Study: {study}")
    print(f"    - Clinical文件: {clinical_path}")
    print(f"    - Metadata文件: {metadata_path}")

    # 2. 检查文件是否存在
    if not os.path.exists(clinical_path) or not os.path.exists(metadata_path):
        print("[!] 错误: 文件不存在，请检查路径。")
        return

    # 3. 读取数据
    df_clinical = pd.read_csv(clinical_path)
    df_meta = pd.read_csv(metadata_path)

    # 统一 case_id 格式
    if 'case_id' not in df_clinical.columns or 'case_id' not in df_meta.columns:
        print("[!] 错误: 两个文件都必须包含 'case_id' 列。")
        return

    df_clinical['case_id'] = df_clinical['case_id'].astype(str).str.strip()
    df_meta['case_id'] = df_meta['case_id'].astype(str).str.strip()

    print(f"[*] 读取成功:")
    print(f"    - Clinical 行数: {len(df_clinical)}")
    print(f"    - Metadata 行数: {len(df_meta)}")

    # 4. 验证重叠列
    common_cols = list(set(df_clinical.columns) & set(df_meta.columns))
    if 'case_id' in common_cols: common_cols.remove('case_id')

    if common_cols:
        print(f"[*] 检测到重叠列: {len(common_cols)} 个")
        # 这里仅做提示，不再赘述详细对比，逻辑保持不变
    else:
        print("[*] 没有发现重叠列。")

    # 5. 执行合并
    new_cols = [c for c in df_meta.columns if c not in df_clinical.columns]
    
    # === [修改点] 即使没有新列，也把 df_clinical 赋值给 df_merged，以便进行后续清洗 ===
    if not new_cols:
        print("[*] 提示: Metadata 中没有 Clinical 缺失的新列，跳过合并步骤，直接进入清洗检查。")
        df_merged = df_clinical
    else:
        print(f"[*] 准备合并下列新数据: {new_cols}")
        df_meta_subset = df_meta[['case_id'] + new_cols]
        df_merged = pd.merge(df_clinical, df_meta_subset, on='case_id', how='left')
    
    # ============================================================
    # 6. 数据清洗：剔除没有 slide_id 的样本 (无论是否合并都要执行)
    # ============================================================
    if 'slide_id' in df_merged.columns:
        print("-" * 40)
        print(f"[*] 正在执行数据清洗: 检查缺失 slide_id 的样本...")
        
        original_count = len(df_merged)
        
        # 剔除 slide_id 为 NaN (空) 的行
        df_merged_cleaned = df_merged.dropna(subset=['slide_id'])
        
        dropped_count = original_count - len(df_merged_cleaned)
        
        if dropped_count > 0:
            print(f"    ⚠️  警告: 已剔除 {dropped_count} 名病人，原因: 缺少 slide_id (为空或NaN)。")
            print(f"    -> 原始样本数: {original_count}")
            print(f"    -> 清洗后样本数: {len(df_merged_cleaned)}")
            df_merged = df_merged_cleaned
        else:
            print("    [√] 所有临床样本均拥有对应的 slide_id，数据完整。")
        print("-" * 40)
    else:
        print("[!] 警告: 数据中未找到 'slide_id' 列，跳过清洗步骤！")

    # ============================================================

    # 7. 覆盖保存
    backup_path = clinical_path + ".bak"
    # 如果原文件就是坏的，备份一下也无妨
    df_clinical.to_csv(backup_path, index=False)
    print(f"[*] 原文件已备份至: {backup_path}")

    df_merged.to_csv(clinical_path, index=False)
    print(f"[*] 成功！已覆盖保存至: {clinical_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Clinical and Metadata CSVs and Clean Missing Slides")
    parser.add_argument('--study', type=str, required=True, help="Study name (e.g., brca, blca, hnsc)")
    args = parser.parse_args()
    
    merge_data(args.study)