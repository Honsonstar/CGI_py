# 训练结果查看说明

## 📋 当前状态

训练正在进行中：
- **项目**: CFdemo_gene_text
- **训练名称**: case_results_dbiase001_lr_tune1
- **学习率配置**: 文本 lr=1e-4, 基因 lr=3e-4 (调优后) 【已改】
- **进度**: 2/5 fold 完成
- **预计完成时间**: 约 17:10-17:15

## 🔍 如何查看训练进度

### 1. 检查结果目录
```bash
ls -lat /root/autodl-tmp/newcfdemo/CFdemo_gene_text/results/case_results_dbiase001_lr_tune1/
```

当看到 `summary.csv` 文件时，说明训练已完成。

### 2. 查看训练日志
```bash
tail -f /root/autodl-tmp/newcfdemo/CFdemo_gene_text/report/UPDATE_LOG.txt
```

### 3. 等待训练完成后运行对比分析
```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text
python report/compare_results.py
```

## 📊 已有结果对比

### 学习率调整前 (统一 lr=0.0005)
- **C-index**: 0.6617
- **IPCW C-index**: 0.6378
- **IBS**: 0.2134
- **IAUC**: 0.6135

### 学习率调整后第一次 (文本 lr=2e-5, 基因 lr=5e-4) ❌
- **C-index**: 0.5803 (-12.30%)
- **IPCW C-index**: 0.5794 (-9.15%)
- **IBS**: 0.2750 (+28.86%)
- **IAUC**: 0.5340 (-12.95%)

**结论**: 性能下降，需要调优

### 学习率调优后 (文本 lr=1e-4, 基因 lr=3e-4) 🔄
- **状态**: 训练进行中
- **预期**: 性能改善，接近或超越基准

## 🎯 修改内容

### 【已改】文件列表
1. **utils/core_utils.py** (第212-248行)
   - 调整学习率: 文本 lr=1e-4, 基因 lr=3e-4

2. **utils/core_utils0.py** (第113-149行)
   - 调整学习率: 文本 lr=1e-4, 基因 lr=3e-4

3. **merge_clinical_meta.py** (第8行)
   - 修改数据路径

4. **scripts/SNN.sh** (第3-4行, 第23行)
   - 修改数据路径

5. **preprocessing/CPCG_algo/data_pocess/preprocess_new.py** (第12-14行)
   - 修改数据路径

## 📁 关键文件

### 代码文件
- `utils/core_utils.py` - 【已改】差异化学习率实现
- `utils/core_utils0.py` - 【已改】差异化学习率实现
- `scripts/SNN.sh` - 【已改】运行脚本
- `merge_clinical_meta.py` - 【已改】数据合并脚本

### 结果目录
- `results/case_results_dbiase001/` - 学习率调整前
- `results/case_results_dbiase001_lr_test/` - 学习率调整后 (性能下降)
- `results/case_results_dbiase001_lr_tune1/` - 🔄 学习率调优后 (当前训练)

### 报告文档
- `report/UPDATE_LOG.txt` - 统一操作日志
- `report/README.md` - 项目说明
- `report/FINAL_REPORT.md` - 完整报告
- `report/compare_results.py` - 结果对比脚本

## 🚀 下一步行动

1. **等待训练完成** (预计17:10-17:15)
2. **运行对比分析**: `python report/compare_results.py`
3. **查看最终结果**
4. **如果性能提升**: 记录成功配置
5. **如果性能仍不佳**: 尝试其他学习率组合

## 💡 学习要点

- **差异化学习率**是有效的多模态优化方法
- **比例调优**很关键，不能简单使用极端值
- **验证机制**很重要，及时发现和调整问题
- **文档记录**便于追踪和复现

---

**更新时间**: 2026-01-22 16:58
**状态**: 训练进行中，等待最终结果
