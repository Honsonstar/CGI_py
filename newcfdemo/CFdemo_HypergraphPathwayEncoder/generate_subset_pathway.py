import pandas as pd
import os
import argparse
import sys

def generate_custom_comps(study):
    # ================= 配置区域 =================
    # 基础目录 (根据你之前的上下文，根目录似乎是 CFdemo)
    base_dir = "/root/autodl-tmp/newcfdemo/CFdemo_HypergraphPathwayEncoder"
    
    # 1. 输入数据文件路径
    # 注意：这里根据你的要求拼写路径
    data_file_path = os.path.join(
        base_dir, 
        "preprocessing/CPCG_algo/raw_data",
        f"finalstage_result_/tcga_{study}",
        f"tcga_{study}_M2M3base_0916.csv"
    )
    
    # 2. 原始通路文件路径
    original_pathway_file = os.path.join(
        base_dir, 
        "datasets_csv/pathway_compositions/combine_comps.csv"
    )
    
    # 3. 输出文件路径
    output_path = os.path.join(
        base_dir, 
        "datasets_csv/pathway_compositions", 
        f"custom_{study}_comps.csv"
    )
    # ===========================================

    print(f"[*] 正在处理 Study: {study}")
    print(f"    - 读取基因数据源: {data_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_file_path):
        print(f"❌ 错误: 找不到数据文件: {data_file_path}")
        return
    if not os.path.exists(original_pathway_file):
        print(f"❌ 错误: 找不到原始通路文件: {original_pathway_file}")
        return

    # --- 步骤 1: 读取数据并提取基因列表 ---
    try:
        # 只读取前几行来获取列名即可，不需要读取整个大文件
        df_data = pd.read_csv(data_file_path, nrows=5)
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        return

    # 根据你的描述：第一列 case_id, 第二列 OS, 第三列开始是基因
    # df.columns 的索引: 0->case_id, 1->OS, 2->Gene1, 3->Gene2...
    all_columns = df_data.columns.tolist()
    
    if len(all_columns) < 3:
        print(f"❌ 错误: 数据文件列数少于3列，无法提取基因。当前列: {all_columns}")
        return

    # 提取基因 (从第3列开始，即索引2)
    kept_genes = all_columns[2:]
    print(f"    - 识别到的前2列 (将被忽略): {all_columns[:2]}")
    print(f"    - 提取到的基因数量: {len(kept_genes)} (例如: {kept_genes[:3]} ...)")

    # --- 步骤 2: 读取原始通路文件并筛选 ---
    print(f"    - 读取原始通路库: {original_pathway_file}")
    try:
        df_comps = pd.read_csv(original_pathway_file)
    except Exception as e:
        print(f"❌ 读取通路文件失败: {e}")
        return

    # 假设 combine_comps.csv 的第一列是 Gene Symbol
    gene_col_name = df_comps.columns[0]
    
    # 筛选：保留 df_comps 中那些“基因名”存在于 kept_genes 列表里的行
    # 注意：这里我们取交集。如果数据里的基因在通路库里找不到，就会被丢弃（这是正常的）
    df_subset = df_comps[df_comps[gene_col_name].isin(kept_genes)].copy()
    
    matched_count = len(df_subset)
    print(f"    - 匹配结果: 在 {len(kept_genes)} 个输入基因中，有 {matched_count} 个在通路库中找到了对应信息。")
    
    if matched_count == 0:
        print("⚠️ 警告: 没有匹配到任何基因！请检查数据文件和通路文件的基因命名是否一致 (例如大小写)。")
        return

    # --- 步骤 3: 保存结果 ---
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_subset.to_csv(output_path, index=False)
    print(f"✅ 成功生成定制通路文件: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subset pathway file based on study data.")
    parser.add_argument('--study', type=str, required=True, help="Study name (e.g., brca, blca, hnsc)")
    
    args = parser.parse_args()
    generate_custom_comps(args.study)