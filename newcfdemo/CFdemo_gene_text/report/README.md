# CFdemo_gene_text 项目报告

欢迎查看CFdemo_gene_text项目的完整实施报告！

## 📁 报告文件说明

### 🎯 快速开始

**查看项目总结**: [SUMMARY.md](SUMMARY.md)
**查看最终结论**: [FINAL_REPORT.md](FINAL_REPORT.md)

### 📊 完整报告文件

1. **[FINAL_REPORT.md](FINAL_REPORT.md)** (12KB)
   - 最终完整报告
   - 包含所有技术细节、结果分析和改进建议
   - **推荐首先阅读此文件**

2. **[SUMMARY.md](SUMMARY.md)** (4.7KB)
   - 项目实施总结
   - 快速了解项目概况和关键改进

3. **[report.txt](report.txt)** (19KB)
   - 详细操作记录
   - 包含16章的完整实施过程
   - 记录了每个步骤和修改

4. **[STATUS.txt](STATUS.txt)** (4.5KB)
   - 项目状态报告
   - 实时更新任务完成状态

5. **[compare_results.py](compare_results.py)** (6.8KB)
   - Python脚本：自动对比两次训练结果
   - 运行方法：`python report/compare_results.py`

## 🎯 项目亮点

### ✅ 成功实现
1. **差异化学习率配置** - 为文本和基因模块设置不同学习率
2. **自动化参数分组** - 根据参数名自动分类
3. **完整实验记录** - 系统性的文档和结果追踪
4. **对比分析** - 量化分析改进效果

### 📊 核心结果

| 指标 | 学习率调整前 | 学习率调整后 | 变化 |
|------|-------------|-------------|------|
| **C-index** | 0.6617 | 0.5803 | **-12.30%** |
| **IPCW C-index** | 0.6378 | 0.5794 | **-9.15%** |
| **IBS** | 0.2134 | 0.2750 | **+28.86%** |

### ⚠️ 发现问题
当前差异化学习率配置 (文本 lr=2e-5, 基因 lr=5e-4) **不适合此任务**，导致性能下降。

### 💡 改进建议
**方案2** (推荐):
```python
text_lr = 1e-4  # 文本模型
gene_lr = 5e-4  # 基因网络
```

## 🔧 技术实现

### 核心代码
```python
# utils/core_utils.py - _init_optim函数
text_lr = 2e-5  # 文本模型学习率
gene_lr = args.lr  # 基因网络学习率

# 根据参数名自动分组
for name, param in model.named_parameters():
    if 'clinical_bert' in name:
        text_params.append(param)
    elif any(keyword in name for keyword in ['fc_omic', 'omic', 'pathway_encoder']):
        gene_params.append(param)
```

### 运行训练

**学习率调整前的训练**:
```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text
conda run -n causal python main.py \
  --label_file datasets_csv/clinical_data/tcga_brca_clinical.csv \
  --study tcga_brca \
  --split_dir splits \
  --data_root_dir /root/autodl-tmp/newcfdemo/CFdemo_gene_text/datasets_csv/reports_clean \
  --task survival \
  --which_splits 5foldcv_ramdom \
  --omics_dir /root/autodl-tmp/newcfdemo/CFdemo_gene_text/preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv \
  --results_dir "case_results_dbiase001" \
  --batch_size 1 \
  --lr 0.0005 \
  --opt radam \
  --reg 0.0001 \
  --alpha_surv 0.5 \
  --weighted_sample \
  --max_epochs 20 \
  --label_col survival_months_dss \
  --k 5 \
  --bag_loss nll_surv \
  --type_of_path custom \
  --modality snn \
  --enable_multitask \
  --multitask_weight 0.12
```

> 提示：当前流程默认使用 **5 折交叉验证**（`which_splits=5foldcv_ramdom`）。如果 `which_splits` 包含 `5fold`，程序会自动强制 `k=5`，避免参数与 splits 目录不一致。

**对比结果**:
```bash
python report/compare_results.py
```

## 📂 项目结构

```
CFdemo_gene_text/
├── results/              # 训练结果
│   ├── case_results_dbiase001/          # 学习率调整前
│   └── case_results_dbiase001_lr_test/  # 学习率调整后
├── utils/               # ✅ 核心工具 (已修改)
│   ├── core_utils.py    # ✅ 差异化学习率实现
│   └── core_utils0.py   # ✅ 差异化学习率实现
├── scripts/             # 运行脚本
│   └── SNN.sh          # ✅ 已修改路径
└── report/             # 📝 报告目录
    ├── README.md        # 📝 本文件
    ├── FINAL_REPORT.md  # 📝 最终完整报告
    ├── SUMMARY.md       # 📝 项目总结
    ├── report.txt       # 📝 详细记录
    ├── STATUS.txt      # 📝 状态报告
    └── compare_results.py # 📝 结果对比脚本
```

## 🎓 学习要点

1. **超参数调优需要精细化** - 简单的大小调整可能适得其反
2. **验证机制的重要性** - 通过对比实验及时发现问题
3. **文档化的价值** - 完整的记录便于问题排查和经验总结
4. **差异化学习率** - 多模态模型优化需要考虑不同模块的特性

## 🚀 下一步行动

1. **立即行动**: 尝试方案2 (文本 lr=1e-4, 基因 lr=5e-4)
2. **短期目标**: 找到比统一学习率更好的配置
3. **长期目标**: 建立完整的超参数优化流程

---

**文档更新**: 2026-01-22 16:45
**项目状态**: ✅ 完成基础实施，需进一步调优
