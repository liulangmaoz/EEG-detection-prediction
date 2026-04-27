# 癫痫脑电信号处理系统

## 项目简介

本项目是一个专注于癫痫脑电信号处理的综合系统，旨在提供从数据预处理、特征提取到发作检测和预测的完整解决方案。系统支持多种分析方法，包括传统信号处理、机器学习和深度学习技术。

## 系统要求

- Python 3.7+
- CPU 或 GPU（推荐使用GPU以加速计算）

## 目录结构

```
├── eeg_processing/          # 核心处理模块
│   ├── __init__.py
│   ├── core.py              # 参数配置与数据加载
│   ├── detection.py         # 发作时间检测
│   ├── features.py          # 特征提取
│   ├── preprocessing.py     # 信号预处理
│   ├── utils.py             # 工具函数
│   ├── visualization.py     # 可视化图表生成
│   └── yolo_*.py            # YOLO视觉检测模块
├── yolo_dataset/            # YOLO数据集配置
│   └── eeg.yaml
├── LFP.py                   # 癫痫发作检测特征处理
├── LFP_non.py               # 正常脑电检测特征处理
├── SVM.py                   # 支持向量机训练
├── data_aggregator.py       # 数据聚合脚本
├── data_summary.py          # 数据汇总脚本
├── main_detect_train.py     # CNN-LSTM发作检测训练
├── main_predict_train.py    # DMSSTAN发作预测训练
├── prediction.py            # 预测后处理
├── yolov5su.pt              # YOLO模型文件
├── requirements.txt         # 依赖包列表
└── README.md                # 项目说明文档
```

## 安装步骤

1. **克隆项目**
   ```bash
   git clone <仓库地址>
   cd <项目目录>
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备

### 数据文件命名规则

- 文件格式：CSV
- 文件名格式：数字.csv（例如：1.csv, 2.csv, 3.csv）
- 系统会自动筛选文件名前缀为数字的文件

### 数据文件格式

CSV文件应包含脑电数据，格式如下：
- 行：时间点
- 列：12个脑电通道，顺序与CHANNEL_NAMES对应：
  1. R_SUB
  2. R_DG
  3. R_CA1
  4. R_CA3
  5. R_AMD
  6. R_ANT
  7. L_SUB
  8. L_DG
  9. L_CA1
  10. L_CA3
  11. L_AMD
  12. L_ANT

### 数据存放位置

- 癫痫发作数据：`data/ictal_epilepsy/`
- 正常脑电数据：`data/interictal_normal/`

## 使用方法

### 处理癫痫发作数据

1. 将癫痫发作CSV数据文件放入 `data/ictal_epilepsy/` 文件夹
2. 运行以下命令：
   ```bash
   python LFP.py
   ```

### 处理正常脑电数据

1. 将正常脑电CSV数据文件放入 `data/interictal_normal/` 文件夹
2. 运行以下命令：
   ```bash
   python LFP_non.py
   ```

### 训练发作检测模型

```bash
python main_detect_train.py
```

### 训练发作预测模型

```bash
python main_predict_train.py
```

## 预处理流程

系统采用以下预处理步骤：
1. 带通滤波（0.5-64Hz）
2. 工频带阻滤波（49-51Hz）
3. ICA去伪迹
4. 平均重参考
5. 基线校正

## 输出结果

每个数据文件的处理结果会生成一个独立的子文件夹，包含以下内容：

### HTML报告
- **01全局偏移脑电图.html** - 所有通道的全局偏移脑电图
- **02频域对比图.html** - 原始信号与滤波后信号的PSD对比
- **03棘波检测.html** - 棘波检测结果
- **04_大发作检测.html** - 癫痫大发作检测结果
- **05交互式脑电图.html** - 可交互的脑电图
- **06多通道时频分析.html** - 时频分析结果

### Excel文件
- **01数据查看.xlsx** - 数据基本信息
- **02癫痫发作检测.xlsx** - 发作检测结果
- **03发作各阶段能量占比.xlsx** - 频段能量分布
- **04发作各阶段主频功率.xlsx** - 主频功率
- **05发作各阶段功率谱熵.xlsx** - 功率谱熵

## 技术实现

- **信号处理**：使用SciPy和MNE库进行信号滤波和分析
- **特征提取**：实现了多种时域、频域和时频域特征
- **机器学习**：支持SVM等传统机器学习方法
- **深度学习**：实现了CNN-LSTM和DMSSTAN等深度学习模型
- **可视化**：生成交互式HTML报告和Excel分析结果

## 注意事项

- 数据文件必须严格按照规定格式命名和组织
- 对于不同数据库采集的电极配置，可能需要调整通道名称和处理参数
- 处理大量数据时，建议使用GPU以提高计算速度
- YOLO视觉检测模块作为辅助方法，可考虑使用短时能量分析作为替代

## 许可证

本项目仅供研究和学习使用。

## 联系方式

无。
