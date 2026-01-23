import os

import pandas as pd

# 定义基本列
basic_columns = ['case_id', 'slide_id', 'site', 'is_female', 'oncotree_code', 'age', 'survival_months', 'censorship',
                 'train']

try:
    # 读取筛选后的 CSV 文件以获取基因列名
    df_filtered = pd.read_csv('result_m2m3_base_0916_n100/tcga_gbmlgg/tcga_gbmlgg_M2M3base_0916.csv', sep=',')  # 如果是制表符，改为 sep='\t'
    print("筛选文件原始列名：", df_filtered.columns.tolist())

    # 获取基因列名（排除 'Unnamed: 0' 和 'OS'）
    gene_columns = [col for col in df_filtered.columns if col not in ['Unnamed: 0', 'OS']]
    print("筛选文件基因列：", gene_columns)

    # 为基因列添加 '_rnaseq' 后缀以匹配原始文件
    gene_columns_rnaseq = [col + '_rnaseq' for col in gene_columns]
    print("原始文件中所需基因列：", gene_columns_rnaseq)

    # 组合所有需要的列（基本列 + 基因列）
    required_columns = basic_columns + gene_columns_rnaseq
    print("所有所需列：", required_columns)

    # 读取原始 CSV 文件，仅加载所需列
    df_original = pd.read_csv('tcga_gbmlgg_all_clean.csv', usecols=required_columns)
    print("原始文件加载列：", df_original.columns.tolist())

    # 保存到新的 CSV 文件
    df_original.to_csv('tcga_gbmlgg_M2M3base_n100_0916_all_clean.csv', index=False)
    print("done")

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e}")
except KeyError as e:
    print(f"错误：原始文件中缺失列 {e}. 请检查筛选文件中的基因名称是否与原始文件匹配（需带 '_rnaseq' 后缀）。")
except Exception as e:
    print(f"未知错误：{e}")
