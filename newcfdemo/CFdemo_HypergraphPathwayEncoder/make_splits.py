import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

# ================= 配置区域 =================
#blca,brca,coadread,hnsc,stad
study = 'stad'

# 1. 锁定那个包含生存数据的 clinical 文件
# 根据你的截图，文件在 datasets_csv/reports_clean/ 目录下
source_csv_path = os.path.join('datasets_csv', 'metadata', f'tcga_{study}.csv')

# 2. 输出目录
output_root = 'splits'
output_dir = os.path.join(output_root, f'tcga_{study}')

# ================= 执行逻辑 =================

if not os.path.exists(source_csv_path):
    print(f"❌ 错误：找不到文件 {source_csv_path}")
    exit()

print(f"📂 读取临床数据: {source_csv_path}")
df = pd.read_csv(source_csv_path)

# 3. 提取 ID 和 标签
# 根据你的截图 image_7cb104.png，ID列名是 'case_id'
if 'case_id' not in df.columns:
    print("❌ 错误：在 CSV 中找不到 'case_id' 列！请检查列名。")
    print(f"   现有列名: {list(df.columns)}")
    exit()

patient_ids = df['case_id'].values

# 4. 尝试获取生存状态用于分层 (Stratification)
# 优先找 'censorship' 或 'censorship_dss'
if 'censorship' in df.columns:
    labels = df['censorship'].values
    print("✅ 使用 'censorship' 列进行分层划分 (Stratified Split)")
elif 'censorship_dss' in df.columns:
    labels = df['censorship_dss'].values
    print("✅ 使用 'censorship_dss' 列进行分层划分")
else:
    print("⚠️ 未找到 censorship 列，将退回随机划分 (Random Split)")
    labels = np.zeros(len(patient_ids)) # 全0意味着不做分层，只随机

# 5. 生成 5 折划分
os.makedirs(output_dir, exist_ok=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"🚀 开始为 {len(patient_ids)} 位病人生成划分文件...")

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
    train_ids = patient_ids[train_idx]
    test_ids = patient_ids[test_idx]
    
    # 补齐长度以便保存
    max_len = max(len(train_ids), len(test_ids))
    train_col = list(train_ids) + [''] * (max_len - len(train_ids))
    test_col = list(test_ids) + [''] * (max_len - len(test_ids))
    
    split_df = pd.DataFrame({'train': train_col, 'test': test_col})
    
    save_path = os.path.join(output_dir, f'splits_{fold}.csv')
    split_df.to_csv(save_path, index=False)
    print(f"   💾 已保存: {save_path}")

print(f"\n🎉 完美！splits 文件夹已生成，你可以运行 main.py 了！")