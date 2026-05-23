# 数据预处理结果使用指南

## 📁 生成的文件

### 1. 主要数据集: `preprocessed_complete_dataset.csv`
- **数据规模**: 105,363 行 × 29 列
- **内容**: 包含完整19个疾病标签的高质量数据
- **用途**: 疾病选择分析、质量评估、主要模型训练

### 2. 补充数据集: `preprocessed_partial_dataset.csv`
- **数据规模**: 898 行 × 29 列
- **内容**: 部分疾病标签数据 (CheXpert填充)
- **用途**: 可作为补充训练数据使用

## 📊 数据质量摘要

- **原始数据**: 106,261 行
- **完整数据**: 105,363 行 (99.2% 保留率)
- **疾病标签**: 19/19 个可用
- **字段优化**: 34 → 29 列


## 📋 可用疾病标签

- No Finding
- Lung Opacity
- Cardiomegaly
- Atelectasis
- Pleural Effusion
- Support Devices
- Edema
- Pneumonia
- Pneumothorax
- Lung Lesion
- Fracture
- Enlarged Cardiomediastinum
- Consolidation
- Pleural Other
- Calcification of the Aorta
- Tortuous Aorta
- Pneumoperitoneum
- Subcutaneous Emphysema
- Pneumomediastinum
