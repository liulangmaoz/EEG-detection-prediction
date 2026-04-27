import os
import warnings
import datetime
from eeg_processing import *
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

# 输入文件夹：默认使用data/ictal_epilepsy，可根据需要修改
INPUT_CSV_DIR = os.path.join(SCRIPT_DIR, "data", "ictal_epilepsy")

# 输出根目录：默认保存到当前目录的results文件夹，或者使用系统默认
OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, "results", "ictal_epilepsy")

# 确保目录存在
os.makedirs(INPUT_CSV_DIR, exist_ok=True)
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

    # ===================== 获取 csv 文件 =====================
    csv_files = [f for f in os.listdir(INPUT_CSV_DIR) if f.endswith('.csv') and f[:-4].isdigit()]
    csv_files.sort(key=lambda x: int(x[:-4]))

    if not csv_files:
        print("未找到任何 csv 文件！")
        exit()

    print(f"找到 {len(csv_files)} 个待处理文件：{csv_files}")

    # ===================== 批量循环处理 =====================
    for idx, csv_name in enumerate(csv_files, start=1):
        # 获取 CSV 纯文件名（不含后缀）
        file_base_name = csv_name[:-4]
        print(f"\n==============================================")
        print(f"           正在处理第 {idx} 个文件：{csv_name}")
        print(f"==============================================\n")

        # 1. 构建当前文件完整路径
        csv_file_path = os.path.join(INPUT_CSV_DIR, csv_name)
        print(f"正在读取 CSV 脑电文件: {csv_file_path}")

        # 2. 读取数据
        raw_data, fs = load_nwb_data(csv_file_path, fs, limit=None)

        # 3. 输出文件夹：与 CSV 文件名完全对应
        save_sub_dir = file_base_name  # 使用原始文件名
        SAVE_DIR = os.path.join(OUTPUT_ROOT_DIR, save_sub_dir)
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"输出目录：{SAVE_DIR}")

        # =====================================================
        lfp_data = preprocess_signal(raw_data, fs, interactive=False)
        global_info, earliest_channel_info, channel_stats_info = detect_seizures_multichannel(lfp_data, fs)

        total_spikes, channel_spikes_list = detect_interictal_spikes_global(lfp_data,fs, global_info,channel_stats_info,threshold_sd=3)
        hfo_first_time = detect_first_hfo_time_all_channels(lfp_data, fs, channel_stats_info)

        channel_phase_power_results = batch_calculate_channel_phase_power(lfp_data, fs, channel_stats_info)
        channel_entropy_results = batch_calculate_phase_spectral_entropy(lfp_data, fs, channel_stats_info)
        channel_band_results = batch_calculate_phase_band_energy(lfp_data, fs, channel_stats_info)

        generate_chart1_html(lfp_data, fs,SAVE_DIR)
        plot_psd_comparison_html(raw_data, lfp_data, fs, SAVE_DIR, channel_idx=0)
        generate_spike_detection_report(lfp_data, channel_stats_info,fs, SAVE_DIR, channel_idx=3, threshold_sd=3)
        generate_seizure_detection_report(channel_stats_info, global_info, lfp_data, fs, SAVE_DIR)
        InteractiveChartGenerator.generate_echarts_sliding_window_html(
            plot_data=lfp_data, fs=fs,
            directory=SAVE_DIR,
            filename="05交互式脑电图.html",
            channel_names=CHANNEL_NAMES
        )
        generate_multi_channel_timefreq_report(
            lfp_data,fs=fs,
            save_dir=os.path.join(SAVE_DIR, "06多通道时频分析.html"),
            channel_names=CHANNEL_NAMES,
            font_cn=font_cn,
            font_en=font_en)
        save_raw_data_view(raw_data, CHANNEL_NAMES, SAVE_DIR)
        save_seizure_detection(channel_stats_info,
                               channel_spikes_list, earliest_channel_info,
                               hfo_first_time, global_info, SAVE_DIR,fs,
                               channel_names=CHANNEL_NAMES)

        # 保存发作各阶段能量占比
        save_band_energy_results(channel_band_results, SAVE_DIR)

        # 保存发作各阶段主频功率
        save_phase_power_results(channel_phase_power_results, SAVE_DIR)

        # 保存发作各阶段功率谱熵
        save_spectral_entropy_results(channel_entropy_results, SAVE_DIR)

        channel_seizure_eeg_dict = batch_extract_seizure_eeg_data(lfp_data, fs, channel_stats_info, phase_duration=60)
        batch_calculate_seizure_features(channel_seizure_eeg_dict, fs, SAVE_DIR)

        print(f"\n===== 所有结果已保存至 Excel：{SAVE_DIR} =====")