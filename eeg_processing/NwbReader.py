import os
import warnings
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

# 关闭警告
warnings.filterwarnings("ignore")

# ====================== 路径与参数（内部定义，100%不报错） ======================
ROOT_DIR = r"D:\python\pythonProject\EEG_CNN-LSTM"

NWB_DATA_DIR = os.path.join(ROOT_DIR, "Data", "nwb_data")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "Data", "raw_data")

SAVE_ICTAL = os.path.join(RAW_DATA_DIR, "ictal_epilepsy")
SAVE_NORMAL = os.path.join(RAW_DATA_DIR, "interictal_normal")

EEG_COLUMNS = [
    'time', 'R_SUB', 'R_DG', 'R_CA1', 'R_CA3', 'R_AMD', 'R_ANT',
    'L_SUB', 'L_DG', 'L_CA1', 'L_CA3', 'L_AMD', 'L_ANT'
]

SAMPLE_RATE = 1000
DURATION_POINTS = 10 * 60 * SAMPLE_RATE
# ================================================================================


def create_folders():
    os.makedirs(SAVE_ICTAL, exist_ok=True)
    os.makedirs(SAVE_NORMAL, exist_ok=True)


def get_all_nwb_files(nwb_dir):
    nwb_files = []
    for file in os.listdir(nwb_dir):
        if file.endswith(".nwb"):
            nwb_files.append((file, os.path.join(nwb_dir, file)))
    return nwb_files


def get_acquisition_names(nwb_path):
    try:
        with NWBHDF5IO(nwb_path, mode='r') as io:
            nwb = io.read()
            return list(nwb.acquisition.keys())
    except Exception:
        return []


def load_eeg_data(nwb_path, acq_name):
    with NWBHDF5IO(nwb_path, mode='r') as io:
        nwb = io.read()
        data = nwb.acquisition[acq_name].data[:]

    if data.shape[0] != DURATION_POINTS:
        data = data.T
    return data


def save_to_csv(eeg_data, save_path, filename):
    eeg_data = eeg_data[:DURATION_POINTS, :]
    time_col = np.arange(DURATION_POINTS).reshape(-1, 1)
    df = pd.DataFrame(np.hstack([time_col, eeg_data]), columns=EEG_COLUMNS)
    save_file = os.path.join(save_path, filename)
    df.to_csv(save_file, index=False)
    print(save_file)


def get_next_index(folder_path):
    """
    自动获取文件夹中已有的最大数字编号 + 1
    例如：已有1,2,3.csv → 返回4
    空文件夹 → 返回1
    不会覆盖任何旧文件
    """
    max_num = 0
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                try:
                    # 提取文件名中的数字（去掉.csv）
                    num = int(os.path.splitext(file)[0])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
    return max_num + 1


def batch_process_nwb_to_csv():
    create_folders()
    nwb_files = get_all_nwb_files(NWB_DATA_DIR)

    if not nwb_files:
        print("No NWB files found.")
        return

    # 自动从现有文件最大编号开始接续
    ictal_idx = get_next_index(SAVE_ICTAL)
    normal_idx = get_next_index(SAVE_NORMAL)

    print(f"接续编号：发作数据从 {ictal_idx} 开始，正常数据从 {normal_idx} 开始")
    print("=" * 60)

    for filename, filepath in nwb_files:
        print(f"Processing {filename}")
        acq_names = get_acquisition_names(filepath)

        if "ictal" in filename.lower():
            for acq in acq_names:
                data = load_eeg_data(filepath, acq)
                save_to_csv(data, SAVE_ICTAL, f"{ictal_idx}.csv")
                ictal_idx += 1

        elif "non" in filename.lower():
            for acq in acq_names:
                data = load_eeg_data(filepath, acq)
                save_to_csv(data, SAVE_NORMAL, f"{normal_idx}.csv")
                normal_idx += 1

    print("\nAll files processed.")


if __name__ == "__main__":
    batch_process_nwb_to_csv()