# 三种运行模式使用指南

## 📋 概述

CFdemo_gene_text项目现已支持三种运行模式，可以通过`--ab_model`参数控制：

| 模式 | ab_model值 | 描述 | 输入数据 |
|------|-----------|------|----------|
| 仅文本 | 1 | 仅使用文本数据进行生存预测 | 文本报告（BioBERT） |
| 仅基因 | 2 | 仅使用基因数据进行生存预测 | 基因/组学数据 |
| 多模态融合 | 3 | 融合基因+文本数据进行生存预测 | 基因 + 文本 |

---

## 🚀 使用方法

### 方法1：命令行参数

```bash
# 模式1：仅文本模式
python main.py \
  --ab_model 1 \
  --results_dir case_results_text_only \
  ... 其他参数

# 模式2：仅基因模式
python main.py \
  --ab_model 2 \
  --results_dir case_results_gene_only \
  ... 其他参数

# 模式3：多模态融合模式（默认）
python main.py \
  --ab_model 3 \
  --results_dir case_results_multimodal \
  ... 其他参数
```

### 方法2：使用测试脚本

```bash
# 测试仅文本模式
python test_three_modes.py --mode 1

# 测试仅基因模式
python test_three_modes.py --mode 2

# 测试多模态融合模式
python test_three_modes.py --mode 3

# 测试所有三种模式
python test_three_modes.py --all
```

---

## 📊 完整示例

### 示例1：仅文本模式

```bash
conda run -n causal python main.py \
  --label_file datasets_csv/clinical_data/tcga_brca_clinical.csv \
  --study tcga_brca \
  --split_dir splits \
  --data_root_dir datasets_csv/reports_clean \
  --task survival \
  --which_splits 5foldcv_ramdom \
  --omics_dir preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv \
  --results_dir case_results_mode1_text_only \
  --batch_size 1 \
  --lr 0.0005 \
  --text_lr 0.0001 \
  --gene_lr 0.0003 \
  --ab_model 1 \
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

### 示例2：仅基因模式

```bash
conda run -n causal python main.py \
  --label_file datasets_csv/clinical_data/tcga_brca_clinical.csv \
  --study tcga_brca \
  --split_dir splits \
  --data_root_dir datasets_csv/reports_clean \
  --task survival \
  --which_splits 5foldcv_ramdom \
  --omics_dir preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv \
  --results_dir case_results_mode2_gene_only \
  --batch_size 1 \
  --lr 0.0005 \
  --text_lr 0.0001 \
  --gene_lr 0.0003 \
  --ab_model 2 \
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

### 示例3：多模态融合模式（推荐）

```bash
conda run -n causal python main.py \
  --label_file datasets_csv/clinical_data/tcga_brca_clinical.csv \
  --study tcga_brca \
  --split_dir splits \
  --data_root_dir datasets_csv/reports_clean \
  --task survival \
  --which_splits 5foldcv_ramdom \
  --omics_dir preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv \
  --results_dir case_results_mode3_multimodal \
  --batch_size 1 \
  --lr 0.0005 \
  --text_lr 0.0001 \
  --gene_lr 0.0003 \
  --ab_model 3 \
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

---

## 🔧 技术实现

### 修改文件

1. **`models/model_SNNOmics.py`**
   - 在`__init__`方法中添加`ab_model`参数
   - 支持三种模式配置

2. **`utils/process_args.py`**
   - 添加`--ab_model`命令行参数
   - 默认为3（多模态融合）

3. **`utils/core_utils.py`**
   - 在模型实例化时传递`ab_model`参数

### 代码示例

```python
# SNNOmics类初始化
def __init__(self, omic_input_dim: int, ..., ab_model: int = 3):
    """
    Args:
        ab_model: 运行模式控制
                  1 = 仅文本模式 (only_text)
                  2 = 仅基因模式 (only_omic)
                  3 = 多模态融合模式【默认】
    """
    self.ab_model = ab_model

# 命令行参数
parser.add_argument('--ab_model', type=int, default=3,
                   choices=[1, 2, 3],
                   help='运行模式: 1=仅文本, 2=仅基因, 3=多模态融合【默认】')
```

---

## 📈 性能比较

### 各模式特点

| 模式 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| 仅文本 | 简单、快速、无需基因数据 | 可能丢失基因信息 | 文本数据丰富、基因数据缺失 |
| 仅基因 | 快速、无需文本处理 | 可能丢失文本信息 | 基因数据质量高、文本数据缺失 |
| 多模态融合 | 综合两种模态信息、性能最佳 | 训练时间较长、复杂度高 | 基因和文本数据都可用 |

### 预期性能

- **多模态融合** > 仅基因 ≈ 仅文本（通常）
- 具体性能取决于数据集特性

---

## ⚠️ 注意事项

1. **数据依赖**：
   - 仅文本模式需要有效的文本数据
   - 仅基因模式需要有效的基因数据
   - 多模态模式需要两种数据都可用

2. **模型配置**：
   - 三种模式使用相同的基础架构
   - 仅在输入处理和融合方式上有所不同

3. **训练时间**：
   - 仅文本/仅基因：较快
   - 多模态融合：较慢但性能通常更好

---

## 🎯 推荐配置

### 最佳实践

1. **默认配置**：
   - 使用多模态融合模式（`--ab_model 3`）
   - 文本学习率：`--text_lr 0.0001`
   - 基因学习率：`--gene_lr 0.0003`

2. **快速验证**：
   - 测试所有三种模式
   - 比较性能差异
   - 选择最佳配置

3. **网格搜索**：
   - 可以对每种模式进行超参数优化
   - 找到每种模式的最优配置

---

## 📝 更新日志

### 2026-01-22
- ✅ 添加三种运行模式支持
- ✅ 实现`--ab_model`参数
- ✅ 创建测试脚本`test_three_modes.py`
- ✅ 完善文档和使用指南

---

## 🤝 支持

如有问题，请查看：
1. `report/UPDATE_LOG.txt` - 项目更新日志
2. `test_three_modes.py` - 测试脚本示例
3. 运行日志中的`[Model Config]`和`[Init Model]`信息

---

*文档版本：v1.0*
*最后更新：2026-01-22*
