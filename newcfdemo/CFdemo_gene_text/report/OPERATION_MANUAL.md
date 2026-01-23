# CFdemo_gene_text 中文操作手册

## 📋 目录

1. [项目简介](#1-项目简介)
2. [快速开始](#2-快速开始)
3. [运行模式](#3-运行模式)
4. [配置说明](#4-配置说明)
5. [命令示例](#5-命令示例)
6. [结果分析](#6-结果分析)
7. [常见问题](#7-常见问题)
8. [更新日志](#8-更新日志)

---

## 1. 项目简介

### 项目概述
CFdemo_gene_text 是一个多模态生存预测模型，支持：
- 基因/多组学数据
- 文本数据（病理报告）
- 多模态融合

### 主要特性
- 支持三种运行模式：多模态融合、仅基因、仅文本
- 差异化学习率支持
- 5折交叉验证
- 完整的评估指标（C-index、IPCW、IBS、IAUC）

---

## 2. 快速开始

### 环境准备
```bash
# 激活conda环境
conda activate causal

# 进入项目目录
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text
```

### 运行第一个实验
```bash
# 运行多模态融合模式（默认）
conda run -n causal python main.py \
  --label_file datasets_csv/clinical_data/tcga_brca_clinical.csv \
  --study tcga_brca \
  --split_dir splits \
  --data_root_dir datasets_csv/reports_clean \
  --omics_dir preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv \
  --results_dir case_results_basic \
  --max_epochs 20
```

### 2.1 生成 5 折交叉验证 splits（首次运行建议先做）
本项目默认使用 `splits/5foldcv_ramdom/<study>/splits_{0..4}.csv` 作为 5 折划分文件。

```bash
# 例：为 BRCA 生成可复现的 5-fold 划分
python make_splits.py --study brca --seed 1
```

---

## 3. 运行模式

### 3.1 多模态融合模式（ab_model=3）

**描述**：同时使用基因数据和文本数据进行预测

```bash
conda run -n causal python main.py \
  --ab_model 3 \
  --text_lr 1e-4 \
  --gene_lr 3e-4 \
  # ... 其他参数
```

### 3.2 仅基因模式（ab_model=2）

**描述**：仅使用基因数据进行分析

```bash
conda run -n causal python main.py \
  --ab_model 2 \
  # ... 其他参数
```

### 3.3 仅文本模式（ab_model=1）

**描述**：仅使用文本数据（病理报告）进行分析

```bash
conda run -n causal python main.py \
  --ab_model 1 \
  # ... 其他参数
```

---

## 4. 配置说明

### 4.1 学习率配置

#### 统一学习率（推荐）
```bash
--lr 5e-4
```
所有参数组使用相同的学习率，训练更稳定。

#### 差异化学习率（高级）
```bash
--text_lr 1e-4  # 文本模型学习率
--gene_lr 3e-4   # 基因网络学习率
```
- **优势**：在仅基因模式下可提升性能
- **劣势**：可能导致训练不稳定
- **推荐场景**：仅基因模式下的性能优化

### 4.2 核心参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--max_epochs` | 训练轮数 | 20 | 20 |
| `--batch_size` | 批次大小 | 1 | 1 |
| `--lr` | 学习率 | 5e-4 | 5e-4 |
| `--opt` | 优化器 | radam | radam |
| `--k` | 折数 | 5 | 5 |
| `--which_splits` | splits 目录名 | 5foldcv_ramdom | 5foldcv_ramdom |

> 说明：如果 `--which_splits` 包含 `5fold`（例如 `5foldcv_ramdom`），程序会自动**强制 `--k=5`**，避免出现“目录是5折但参数却跑成别的折数”的不一致问题。

---

## 5. 命令示例

### 5.1 标准实验

```bash
# 多模态融合，统一学习率
conda run -n causal python main.py \
  --label_file datasets_csv/clinical_data/tcga_brca_clinical.csv \
  --study tcga_brca \
  --split_dir splits \
  --data_root_dir datasets_csv/reports_clean \
  --task survival \
  --which_splits 5foldcv_ramdom \
  --omics_dir preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_brca/tcga_brca_M2M3base_0916.csv \
  --results_dir case_results_multi \
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
  --multitask_weight 0.12 \
  --ab_model 3
```

### 5.2 仅基因模式实验

```bash
# 仅基因，差异化学习率
conda run -n causal python main.py \
  --ab_model 2 \
  --text_lr 1e-4 \
  --gene_lr 3e-4 \
  # ... 其他参数同标准实验
```

### 5.3 仅文本模式实验

```bash
# 仅文本，统一学习率
conda run -n causal python main.py \
  --ab_model 1 \
  --lr 5e-4 \
  # ... 其他参数同标准实验
```

---

## 6. 结果分析

### 6.1 结果目录结构

```
results/
  └── case_results_xxx/
      └── tcga_brca__nll_surv_xxx/
          ├── summary.csv              # 汇总结果
          ├── fold_*.pkl              # 各折详细结果
          ├── splits_*.csv            # 分割数据
          └── TCGA_BRCA-Fold*.pkl     # 模型输出
```

### 6.2 关键指标

| 指标 | 说明 | 范围 | 目标 |
|------|------|------|------|
| C-index | 一致性指数 | 0-1 | 越高越好（>0.7优秀） |
| IPCW | 竞争风险指数 | 0-1 | 越高越好 |
| IBS | 综合Brier分数 | 0-1 | 越低越好 |
| IAUC | 时间依赖AUC | 0-1 | 越高越好 |

### 6.3 查看结果

```bash
# 查看汇总结果
cat results/case_results_xxx/*/summary.csv

# 使用Python分析
python
import pandas as pd
df = pd.read_csv('results/case_results_xxx/*/summary.csv')
print(df[['val_cindex', 'val_cindex_ipcw', 'val_IBS', 'val_iauc']].describe())
```

---

## 7. 常见问题

### 7.1 训练速度慢

**原因**：
- GPU显存不足
- 数据加载瓶颈

**解决方案**：
```bash
# 减少批处理大小
--batch_size 1

# 检查GPU使用率
nvidia-smi
```

### 7.2 内存不足

**解决方案**：
1. 降低batch_size
2. 关闭梯度检查点
3. 使用较小的模型

### 7.3 结果不理想

**优化方向**：
1. 调整学习率
2. 增加训练轮数
3. 尝试差异化学习率
4. 检查数据质量

---

## 8. 更新日志

### 2026-01-22
- ✅ 添加统一学习率支持
- ✅ 实现三种运行模式
- ✅ 优化差异化学习率策略
- ✅ 完成网格搜索实验
- ✅ 添加详细注释

### 核心修改点

#### 1. 学习率配置（core_utils.py）
```python
# 修改前：差异化学习率
text_lr = 1e-4
gene_lr = 3e-4

# 修改后：统一学习率（默认）
text_lr = args.lr
gene_lr = args.lr
```

#### 2. 三模式支持（model_SNNOmics.py）
```python
ab_model = 1  # 仅文本
ab_model = 2  # 仅基因
ab_model = 3  # 多模态融合（默认）
```

#### 3. 参数解析（process_args.py）
```python
parser.add_argument('--ab_model', type=int, default=3,
                   choices=[1, 2, 3],
                   help='运行模式: 1=仅文本, 2=仅基因, 3=多模态融合')
```

---

## 9. 联系信息

### 问题反馈
- 项目位置：`/root/autodl-tmp/newcfdemo/CFdemo_gene_text/`
- 报告目录：`/root/autodl-tmp/newcfdemo/CFdemo_gene_text/report/`

### 相关文档
- `README.md`：项目概述
- `THREE_MODES_GUIDE.md`：三模式使用指南
- `FINAL_REPORT.md`：最终实验报告

---

**最后更新**：2026-01-22
**版本**：v1.0
