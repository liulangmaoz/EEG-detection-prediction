# 癫痫脑电信号处理说明

## 系统要求
- Python 3 以上
- CPU / GPU 均可运行

---

## 安装步骤

### 1. 下载项目并解压
将项目压缩包下载后，解压到本地任意目录。

### 项目结构

========

eeg_processing/              # 核心处理模块

__init__.py

core.py                  # 参数配置 / 加载数据模块

detection.py             # 发作时间检测模块

 features.py              # 特征提取模块
 
 preprocessing.py         # 预处理模块
 
utils.py                 # 工具函数

visualization.py         # 可视化图表生成

yolo_*.py                # YOLO 视觉检测发作时间模块（不一定靠谱，可用短时能量替代）

LFP.py                       # 癫痫发作检测特征处理脚本

LFP_non.py                   # 正常脑电检测特征处理脚本

data_*.py                    # 整理特征数据的相关脚本

 SVM.py                       # 支持向量机训练发作检测
 
main_detect_train.py         # CNN-LSTM 发作检测训练脚本

 main_predict_train.py        # DMSSTAN 发作预测训练脚本
 
prediction.py                # 预测后处理模块

yolov5su.pt                  # YOLO 模型文件

 requirements.txt             # 依赖包列表（可能不全，运行时按需补充）

=========

### 2.安装依赖

终端换成本地
```bash
cd "D/你的存放文件夹"
pip install -r requirements.txt

### 数据文件命名规则

数据文件需要遵循以下命名规则：

- 文件格式：CSV
- 文件名格式：数字.csv（例如：1.csv, 2.csv, 3.csv）
- 这是因为代码会自动筛选文件名前缀为数字的文件
- 这样方便找

### 数据文件格式
不同数据库采集电极有差别
具体看着调整
CSV文件应该包含脑电数据，我的格式如下：

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

1. 将癫痫发作CSV数据文件放入 `data/ictal_epilepsy/` 文件夹中

2. 在命令行中运行：

```bash
python LFP.py
```

3. 处理结果会保存在 `results/ictal_epilepsy/` 文件夹中

### 处理正常脑电数据

1. 将正常脑电CSV数据文件放入 `data/interictal_normal/` 文件夹中

2. 在命令行中运行：

```bash
python LFP_non.py
```

3. 处理结果会保存在 `results/interictal_normal/` 文件夹中

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

处理代码仅供参考

代码太冗长本来想用trae整理的

结果没成功所以我把屎山代码传上来了

懒得整理了
