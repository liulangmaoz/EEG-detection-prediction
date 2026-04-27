# EEG Processing Utils Module
# 功能: 提供通用工具函数和配置

import os
from matplotlib.font_manager import FontProperties

# 保存目录设置
save_dir = os.path.join(os.path.expanduser("~"), "EEG_Results")
os.makedirs(save_dir, exist_ok=True)

# 字体设置
font_cn = FontProperties(family='Microsoft YaHei', size=12)
font_en = FontProperties(family='Arial', size=12)
