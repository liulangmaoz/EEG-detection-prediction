import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ====================== 【模式开关】 ======================
# 模式选择：epilepsy = 癫痫文件夹（应该预警），normal = 正常文件夹（不应该预警）
PREDICT_MODE = "epilepsy"

# 两个文件夹路径（自动匹配模式）
EPILEPSY_FOLDER = r"D:\python\pythonProject\EEG_CNN-LSTM\Data\clean_data\ictal_epilepsy"
NORMAL_FOLDER = r"D:\python\pythonProject\EEG_CNN-LSTM\Data\clean_data\interictal_normal"

# ====================== 核心参数 ======================
FS = 1000
SEG_SEC = 2
STEP_SEC = 1
N_CHANNELS = 12
SEG_LEN = SEG_SEC * FS
STEP_LEN = STEP_SEC * FS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\python\pythonProject\EEG_CNN-LSTM\best_eeg_model.pth"
THRESHOLD = 0.7

# ====================== K-of-N 规则 ======================
N1 = 6
K1 = 4
N2 = 10
K2 = 6
ALERT_COUNT = 2

# ====================== 模型结构 ======================
class DynamicTemporalExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
    def forward(self, x):
        return x * torch.sigmoid(self.conv(torch.mean(x, dim=-1, keepdim=True)))

class ChannelExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)

class DynamicTemporalAttention(nn.Module):
    def __init__(self, in_channels, embed_dim=64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, embed_dim, 3, padding=1)
        self.dte = DynamicTemporalExcitation(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.dte(self.conv(x)).transpose(1, 2)
        return self.norm(x + self.attn(x, x, x)[0])

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.c2 = nn.Conv1d(in_channels, 32, 5, padding=2)
        self.c3 = nn.Conv1d(in_channels, 32, 7, padding=3)
        self.cel = ChannelExcitation(96)
    def forward(self, x):
        return self.cel(torch.cat([self.c1(x), self.c2(x), self.c3(x)], 1)).mean(-1)

class DMSSTAN(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.dta = DynamicTemporalAttention(in_channels)
        self.mssa = MultiScaleSpatialAttention(in_channels)
        self.cls = nn.Sequential(
            nn.Linear(64 + 96, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.cls(torch.cat([self.dta(x).mean(1), self.mssa(x)], 1))

# ====================== 预处理 ======================
def clean_segment(seg):
    seg = seg.astype(str)
    mask = np.char.isdigit(seg) | np.char.startswith(seg, '-') | np.char.equal(seg, '.')
    seg[~mask] = '0.0'
    return seg.astype(np.float32)

def z_score_normalize(data):
    mean = np.mean(data, axis=-1, keepdim=True)
    std = np.std(data, axis=-1, keepdim=True)
    std[std == 0] = 1.0
    return (data - mean) / std

# ====================== K-of-N ======================
def k_of_n(pred_list, K, N):
    window = pred_list[-N:]
    return 1 if sum(window) >= K else 0

# ====================== 单文件预测 ======================
def predict_single_file(csv_path, model, device):
    all_binary = []
    level2_results = []
    consecutive_alert = 0
    alert_triggered = False
    alert_timestamp = 0.0

    df = pd.read_csv(csv_path)
    data = df.iloc[:, :N_CHANNELS].values.T
    data = clean_segment(data)
    total_len = data.shape[1]

    with torch.no_grad():
        for idx, start in enumerate(range(0, total_len - SEG_LEN + 1, STEP_LEN)):
            seg = data[:, start:start + SEG_LEN]
            seg = z_score_normalize(seg)

            tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            prob = torch.softmax(model(tensor), dim=1)[:, 1].item()
            binary = 1 if prob >= THRESHOLD else 0
            all_binary.append(binary)

            if len(all_binary) >= N1:
                level1 = k_of_n(all_binary, K1, N1)
                level2_results.append(level1)

                if len(level2_results) >= N2:
                    level2 = k_of_n(level2_results, K2, N2)

                    if level2 == 1:
                        consecutive_alert += 1
                        if consecutive_alert >= ALERT_COUNT:
                            alert_timestamp = start / FS
                            alert_triggered = True
                            break
                    else:
                        consecutive_alert = 0

    return alert_triggered, alert_timestamp

# ====================== 批量预测主程序 ======================
if __name__ == '__main__':
    # 自动匹配文件夹
    if PREDICT_MODE == "epilepsy":
        DATA_FOLDER = EPILEPSY_FOLDER
        sample_type = "癫痫样本"
    elif PREDICT_MODE == "normal":
        DATA_FOLDER = NORMAL_FOLDER
        sample_type = "正常样本"
    else:
        print("模式错误！请输入 epilepsy 或 normal")
        exit()

    # 加载模型
    print("=" * 65)
    print(" 正在加载模型...")
    model = DMSSTAN(in_channels=12).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(" 模型加载完成")
    print("=" * 65)

    # 获取文件
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        print(" 未找到CSV文件！")
        exit()

    print(f" 找到 {len(csv_files)} 个 {sample_type}，开始批量预测...")
    print("=" * 65)

    total_files = len(csv_files)
    correct_count = 0
    wrong_count = 0
    run_log = []

    # 遍历预测
    for i, file_name in enumerate(tqdm(csv_files, desc="预测进度"), 1):
        file_path = os.path.join(DATA_FOLDER, file_name)
        try:
            triggered, timestamp = predict_single_file(file_path, model, DEVICE)

            # ====================== 判断正确/错误 ======================
            if PREDICT_MODE == "epilepsy":
                is_correct = triggered
                if is_correct:
                    log = f"[{i}/{total_files}] {file_name} | 预警于 {timestamp:.1f}s | 预测：正确"
                else:
                    log = f"[{i}/{total_files}] {file_name} | 未预警 | 预测：错误"

            else:  # normal
                is_correct = not triggered
                if is_correct:
                    log = f"[{i}/{total_files}] {file_name} | 未预警 | 预测：正确"
                else:
                    log = f"[{i}/{total_files}] {file_name} | 预警于 {timestamp:.1f}s | 预测：错误(误报)"

            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            run_log.append(log)
            print(log)

        except Exception as e:
            err_log = f"[{i}/{total_files}] {file_name} | 失败：{str(e)}"
            run_log.append(err_log)
            print(err_log)

    # 统计结果
    accuracy = (correct_count / total_files) * 100
    print("\n" + "=" * 65)
    print(f" 【{sample_type}】预测完成 - 统计结果")
    print("=" * 65)
    print(f" 总样本：{total_files}")
    print(f" 预测正确：{correct_count}")
    print(f" 预测错误：{wrong_count}")
    print(f" 准确率：{accuracy:.2f}%")
    print("=" * 65)

    # 保存日志
    log_name = f"{PREDICT_MODE}_run_log.txt"
    log_save_path = os.path.join(DATA_FOLDER, log_name)
    with open(log_save_path, 'w', encoding='utf-8') as f:
        f.write(f"EEG癫痫预测 - {sample_type}日志\n")
        f.write("=" * 50 + "\n")
        f.write(f"总样本：{total_files}\n")
        f.write(f"正确：{correct_count}\n")
        f.write(f"错误：{wrong_count}\n")
        f.write(f"准确率：{accuracy:.2f}%\n")
        f.write("=" * 50 + "\n\n")
        for line in run_log:
            f.write(line + "\n")

    print(f" 日志已保存：{log_save_path}")
    print("=" * 65)