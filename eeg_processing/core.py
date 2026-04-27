import os
import numpy as np
import pandas as pd
import datetime
from matplotlib.font_manager import FontProperties

# 全局配置
DATA_LIMIT_CHART2 = 5000
DATE_STR = datetime.datetime.now().strftime("%Y-%m-%d")
HTML_OUTPUT_DIR = os.path.join(r"D:\python\pythonProject\figure", DATE_STR)
save_dir = HTML_OUTPUT_DIR
TRACK_SCALE = 200
CHANNEL_SPACING = 1.5
fs = 1000
CHANNEL_NAMES = ['R_SUB', 'R_DG', 'R_CA1', 'R_CA3', 'R_AMD', 'R_ANT', 'L_SUB', 'L_DG', 'L_CA1', 'L_CA3', 'L_AMD', 'L_ANT']

# 常量定义
PHASE_DURATION = 60.0
INTERVAL_DURATION = 60.0
ALPHA_BAND = (8.0, 12.0)
BETA_BAND = (12.0, 30.0)
ANALYSIS_BAND = (1.0, 100.0)
SAMPLING_FREQUENCY = 1000.0

EEG_BANDS = {
    'delta': (0.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0),
    'gamma': (30.0, 100.0)
}

font_cn = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
font_en = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf", size=12)

def load_nwb_data(file_path, fs, limit=None):
    """
    从CSV文件读取脑电数据
    :param file_path: CSV文件路径
    :param fs: 采样率
    :param limit: 数据点限制，None表示读取全部数据
    :return: 脑电数据和采样率
    """
    print(f"正在读取 CSV 脑电文件: {file_path} ...")
    
    try:
        df = pd.read_csv(file_path)
        
        CHANNEL_NAMES = [
            'R_SUB', 'R_DG', 'R_CA1', 'R_CA3', 'R_AMD', 'R_ANT',
            'L_SUB', 'L_DG', 'L_CA1', 'L_CA3', 'L_AMD', 'L_ANT'
        ]
        lfp_data = df[CHANNEL_NAMES].values
        
        if limit is not None:
            lfp_data = lfp_data[:limit]
        
        print(f"数据加载成功。形状: {lfp_data.shape}")
        
        for i in range(12):
            ch_min = np.min(lfp_data[:, i])
            ch_max = np.max(lfp_data[:, i])
            print(f"通道{i + 1}波动范围: {ch_min:.1f} ~ {ch_max:.1f} μV")
        
        print("\n===== 数据前20行（每行12个通道）=====")
        rows_to_print = min(20, lfp_data.shape[0])
        for row_idx in range(rows_to_print):
            row_data = lfp_data[row_idx][:12]
            channel_data = [f"{val:.2f}" for val in row_data]
            print(f"第{row_idx + 1:2d}行: {'  '.join(channel_data)}")
        
        return lfp_data, fs
        
    except Exception as e:
        print(f"读取CSV出错: {e}")
        return None, fs
