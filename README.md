# EEG数据处理系统使用指南

## 系统要求

- Python 3.8或更高版本
- 足够的内存（建议至少8GB，处理大数据时可能需要更多）

## 安装步骤

### 1. 下载代码

将以下文件和文件夹下载到您的本地计算机的某个目录中：

```
your_project_folder/
├── eeg_processing/          # 核心处理模块
│   ├── __init__.py
│   ├── core.py
│   ├── detection.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── visualization.py
├── LFP.py                   # 癫痫发作数据处理脚本
├── LFP_non.py               # 正常脑电数据处理脚本
└── requirements.txt         # 依赖包列表
```

### 2. 安装依赖

在命令行中进入您的项目文件夹，然后运行：

```bash
pip install -r requirements.txt
```

如果您使用conda环境：

```bash
conda install numpy matplotlib pandas scipy scikit-learn
pip install mne openpyxl
```

## 数据存储设置

### 文件夹结构

系统默认会在项目根目录下创建以下文件夹结构：

```
your_project_folder/
├── data/                    # 数据文件夹（系统会自动创建）
│   ├── ictal_epilepsy/     # 癫痫发作数据文件夹
│   └── interictal_normal/  # 正常脑电数据文件夹
└── results/                # 结果文件夹（系统会自动创建）
    ├── ictal_epilepsy/     # 癫痫发作处理结果
    └── interictal_normal/  # 正常脑电处理结果
```

### 数据文件命名规则

您的数据文件需要遵循以下命名规则：

- 文件格式：CSV
- 文件名格式：数字.csv（例如：1.csv, 2.csv, 3.csv）
- 这是因为代码会自动筛选文件名前缀为数字的文件

### 数据文件格式

您的CSV文件应该包含脑电数据，格式如下：

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

## 使用方法

### 处理癫痫发作数据

1. 将您的癫痫发作CSV数据文件放入 `data/ictal_epilepsy/` 文件夹中

2. 在命令行中运行：

```bash
python LFP.py
```

3. 处理结果会保存在 `results/ictal_epilepsy/` 文件夹中

### 处理正常脑电数据

1. 将您的正常脑电CSV数据文件放入 `data/interictal_normal/` 文件夹中

2. 在命令行中运行：

```bash
python LFP_non.py
```

3. 处理结果会保存在 `results/interictal_normal/` 文件夹中

## 自定义路径（可选）

如果您想使用不同的输入或输出路径，可以修改相应的脚本文件：

### 对于LFP.py

修改以下部分：

```python
# 输入文件夹：可根据需要修改
INPUT_CSV_DIR = os.path.join(SCRIPT_DIR, "data", "ictal_epilepsy")

# 输出根目录：可根据需要修改
OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, "results", "ictal_epilepsy")
```

例如改为：

```python
INPUT_CSV_DIR = "D:/your_custom_data_path/ictal_epilepsy"
OUTPUT_ROOT_DIR = "D:/your_custom_results_path/ictal_epilepsy"
```

### 对于LFP_non.py

修改以下部分：

```python
# 输入文件夹：可根据需要修改
INPUT_NORMAL_CSV_DIR = os.path.join(SCRIPT_DIR, "data", "interictal_normal")

# 输出根目录：可根据需要修改
OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, "results", "interictal_normal")
```

## 输出结果说明

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

## 常见问题

### Q1: 我的数据文件名不是数字开头的怎么办？

A: 您可以修改脚本中获取csv文件的部分，或者将文件名改为数字开头的格式（例如：data1.csv改为1.csv）。

### Q2: 系统提示找不到字体怎么办？

A: 这是警告信息，不会影响程序运行。如果您需要正确显示中文，可以修改eeg_processing/utils.py中的字体设置，使用您系统中已有的中文字体。

### Q3: 内存不足怎么办？

A: 如果您的数据文件很大，可以考虑：
1. 只处理部分数据
2. 增加系统虚拟内存
3. 使用更强大的硬件

### Q4: 如何查看生成的HTML报告？

A: 直接用浏览器（Chrome, Edge, Firefox等）打开HTML文件即可查看报告。

## 技术支持

如果您在使用过程中遇到问题，请检查：
1. Python版本是否满足要求
2. 所有依赖包是否正确安装
3. 数据文件格式是否正确
4. 文件路径是否正确设置

祝您使用愉快！
