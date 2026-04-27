import os
import warnings
import datetime
from eeg_processing import *
# 导入新的特征计算模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
font_cn = font_cn
font_en = font_en

# ===================== 路径配置 =====================
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入文件夹：默认使用data/interictal_normal，可根据需要修改
INPUT_NORMAL_CSV_DIR = os.path.join(SCRIPT_DIR, "data", "interictal_normal")

# 输出根目录：默认保存到当前目录的results文件夹，或者使用系统默认
OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, "results", "interictal_normal")

# 确保目录存在
os.makedirs(INPUT_NORMAL_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

DATE_STR = datetime.datetime.now().strftime("%Y-%m-%d")
CHANNEL_NAMES = CHANNEL_NAMES
warnings.filterwarnings('ignore', category=UserWarning, module='pynwb.ecephys')
PHASE_DURATION = PHASE_DURATION
fs = fs

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='pynwb.ecephys')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # ===================== 获取 正常脑电 csv 文件 =====================
    csv_files = [f for f in os.listdir(INPUT_NORMAL_CSV_DIR) if f.endswith('.csv') and f[:-4].isdigit()]
    csv_files.sort(key=lambda x: int(x[:-4]))

    if not csv_files:
        print("未找到任何正常脑电 csv 文件！")
        exit()

    print(f"找到 {len(csv_files)} 个正常脑电文件：{csv_files}")

    # ===================== 批量循环处理 =====================
    for idx, csv_name in enumerate(csv_files, start=1):
        # 获取 CSV 纯文件名（不含后缀）→ 和癫痫数据完全一致
        file_base_name = csv_name[:-4]
        print(f"\n==============================================")
        print(f"           正在处理第 {idx} 个正常文件：{csv_name}")
        print(f"==============================================\n")

        # 1. 文件路径
        csv_file_path = os.path.join(INPUT_NORMAL_CSV_DIR, csv_name)
        print(f"正在读取 CSV 脑电文件: {csv_file_path}")

        # 2. 读取数据
        raw_data, fs = load_nwb_data(csv_file_path, fs, limit=None)

        # 3. 创建对应文件夹：使用原始CSV文件名命名，和癫痫数据命名逻辑完全统一
        save_sub_dir = file_base_name
        SAVE_DIR = os.path.join(OUTPUT_ROOT_DIR, save_sub_dir)
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"输出目录：{SAVE_DIR}")

        # ===================== 正常处理逻辑 =====================
        lfp_data = preprocess_signal(raw_data, fs)

        global_info, earliest_channel_info, channel_stats_info = detect_seizures_multichannel(lfp_data, fs)

        generate_chart1_html(raw_data, fs, SAVE_DIR)
        plot_psd_comparison_html(raw_data, lfp_data, fs, SAVE_DIR, channel_idx=0)

        generate_spike_detection_report(lfp_data, channel_stats_info, fs, SAVE_DIR, channel_idx=3, threshold_sd=3)
        generate_seizure_detection_report(channel_stats_info, global_info, lfp_data, fs, SAVE_DIR)
        InteractiveChartGenerator.generate_echarts_sliding_window_html(
            plot_data=lfp_data, fs=fs,
            directory=SAVE_DIR,
            filename="05交互式脑电图.html",
            channel_names=CHANNEL_NAMES
        )
        generate_multi_channel_timefreq_report(
            lfp_data, fs=fs,
            save_dir=os.path.join(SAVE_DIR, "06多通道时频分析.html"),
            channel_names=CHANNEL_NAMES,
            font_cn=font_cn,
            font_en=font_en
        )
        save_raw_data_view(raw_data, CHANNEL_NAMES, SAVE_DIR)

        channel_non_seizure_eeg_dict = batch_extract_non_seizure_eeg_data(lfp_data, fs, phase_duration=60,
                                                                            total_candidate_duration=300)
        channel_non_seizure_power_results = batch_calculate_non_seizure_phase_power(channel_non_seizure_eeg_dict, fs)
        channel_non_seizure_entropy_results = batch_calculate_non_seizure_spectral_entropy(
            channel_non_seizure_eeg_dict, fs)
        channel_non_seizure_band_results = batch_calculate_non_seizure_band_energy(channel_non_seizure_eeg_dict, fs)

        save_non_seizure_band_energy_results(channel_non_seizure_band_results, SAVE_DIR)
        save_non_seizure_spectral_entropy_results(channel_non_seizure_entropy_results, SAVE_DIR)
        save_non_seizure_phase_power_results(channel_non_seizure_power_results, SAVE_DIR)
        batch_calculate_seizure_features(channel_non_seizure_eeg_dict, fs, SAVE_DIR)

        print(f"\n===== 所有结果已保存至 Excel：{SAVE_DIR} =====")