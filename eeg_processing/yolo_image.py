import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ======================== 路径定位 =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 数据根目录
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "clean_data")
ICTAL_DIR = os.path.join(CLEAN_DATA_DIR, "ictal_epilepsy")
NORMAL_DIR = os.path.join(CLEAN_DATA_DIR, "interictal_normal")

# 通道名称（你指定的）
CHANNEL_NAMES = [
    'R_SUB', 'R_DG', 'R_CA1', 'R_CA3', 'R_AMD', 'R_ANT',
    'L_SUB', 'L_DG', 'L_CA1', 'L_CA3', 'L_AMD', 'L_ANT'
]

# 采样率
FS = 1000

# ======================== 绘图函数 =========================
def plot_eeg_to_jpg(data, save_path, fs=FS):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    n_samples, n_channels = data.shape

    plt.figure(figsize=(18, 8))
    offset_step = 10000

    for i in range(n_channels):
        plt.plot(data[:, i] - offset_step * i, linewidth=0.5)

    plt.title("全局偏移脑电图 (Global Offset View)", fontsize=14)
    plt.xlabel(f"Time (s) | Total: {n_samples / fs:.1f}s", fontsize=12)

    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel("")

    for i in range(n_channels):
        y_pos = -offset_step * i
        plt.text(-n_samples * 0.02, y_pos, CHANNEL_NAMES[i],
                 fontsize=10, ha='right', va='center')

    plt.xlim(-n_samples * 0.02, n_samples)
    plt.tight_layout()

    plt.savefig(save_path, format='jpg', dpi=100, bbox_inches='tight')
    plt.close()

# ======================== 批量转换 CSV → JPG =========================
def process_folder(csv_folder):
    """
    批量把一个文件夹下的所有 CSV → 绘图
    直接保存到 CSV 所在的同一个文件夹内
    1.csv → 1.jpg
    """
    for fname in os.listdir(csv_folder):
        if not fname.endswith(".csv"):
            continue

        csv_path = os.path.join(csv_folder, fname)
        jpg_name = os.path.splitext(fname)[0] + ".jpg"
        jpg_path = os.path.join(csv_folder, jpg_name)

        print(f"正在处理: {fname} -> {jpg_name}")

        df = pd.read_csv(csv_path)
        data = df.iloc[:, 1:].values

        plot_eeg_to_jpg(data, jpg_path)

# ======================== 主执行 =========================
if __name__ == "__main__":
    print("===== EEG CSV → JPG 批量转换工具 =====")
    print(f"脚本根目录: {BASE_DIR}")
    print(f"癫痫数据目录: {ICTAL_DIR}")
    print(f"正常数据目录: {NORMAL_DIR}")
    print("=" * 60)

    print("\n【1】处理癫痫数据 (ictal_epilepsy)...")
    process_folder(ICTAL_DIR)

    print("\n【2】处理正常数据 (interictal_normal)...")
    process_folder(NORMAL_DIR)

    print("\n 全部完成！图片已保存到 CSV 同文件夹下")