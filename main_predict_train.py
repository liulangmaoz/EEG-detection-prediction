import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
FS = 1000
N_CHANNELS = 12
TOTAL_SAMPLES = 600000
TOTAL_MINUTES = 10
IMG_WIDTH = 1000

# ===================== 路径配置 =====================
BASE_DIR = r"D:\python\pythonProject\EEG_CNN-LSTM"
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "Data", "clean_data")

EPILEPSY_CSV_DIR = os.path.join(CLEAN_DATA_DIR, "ictal_epilepsy")
EPILEPSY_TXT_DIR = os.path.join(BASE_DIR, "DataLoader", "time", "ictal_txt")

NORMAL_CSV_DIR = os.path.join(CLEAN_DATA_DIR, "interictal_normal")
NORMAL_TXT_DIR = os.path.join(BASE_DIR, "DataLoader", "time", "normal_txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "segmented_dataset")
EPILEPSY_OUTPUT = os.path.join(OUTPUT_DIR, "epilepsy_segments")
NORMAL_OUTPUT = os.path.join(OUTPUT_DIR, "normal_segments")

os.makedirs(EPILEPSY_OUTPUT, exist_ok=True)
os.makedirs(NORMAL_OUTPUT, exist_ok=True)

# ===================== 模型训练参数 =====================
SEGMENT_DIR = OUTPUT_DIR
WINDOW_SEC = 2
STEP_SEC = 1
WINDOW_LEN = WINDOW_SEC * FS
STEP_LEN = STEP_SEC * FS

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 8e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5
model_save_path = "best_eeg_model.pth"

# ===================== 工具函数：数据切割 =====================
def read_yolo_txt(txt_path, img_width=IMG_WIDTH, total_samples=TOTAL_SAMPLES):
    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        return None, None

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    x1_list = []
    x2_list = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        x_center, w = parts[1], parts[3]

        x1 = (x_center - w / 2) * img_width
        x2 = (x_center + w / 2) * img_width
        x1_list.append(x1)
        x2_list.append(x2)

    start_x = min(x1_list)
    end_x = max(x2_list)

    start_samp = int((start_x / img_width) * total_samples)
    end_samp = int((end_x / img_width) * total_samples)
    return start_samp, end_samp

def load_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    data = df.values.T
    return data

def extract_epilepsy_segments():
    epilepsy_segments = []
    filenames = sorted([f for f in os.listdir(EPILEPSY_CSV_DIR) if f.endswith(".csv")])

    for fname in filenames:
        base = os.path.splitext(fname)[0]
        csv_path = os.path.join(EPILEPSY_CSV_DIR, fname)
        txt_path = os.path.join(EPILEPSY_TXT_DIR, f"{base}.txt")

        onset_samp, _ = read_yolo_txt(txt_path)
        if onset_samp is None:
            continue

        pre_minutes = 5
        pre_samples = int(pre_minutes * 60 * FS)
        start = max(0, onset_samp - pre_samples)
        end = onset_samp

        eeg = load_csv(csv_path)
        segment = eeg[:, start:end]

        save_path = os.path.join(EPILEPSY_OUTPUT, f"epi_{base}.npy")
        np.save(save_path, segment)
        epilepsy_segments.append(segment.shape)

    return epilepsy_segments

def extract_normal_segments():
    normal_segments = []
    filenames = sorted([f for f in os.listdir(NORMAL_CSV_DIR) if f.endswith(".csv")])

    start_min = 3
    end_min = 8
    start_samp = int(start_min * 60 * FS)
    end_samp = int(end_min * 60 * FS)

    for fname in filenames:
        csv_path = os.path.join(NORMAL_CSV_DIR, fname)
        eeg = load_csv(csv_path)
        segment = eeg[:, start_samp:end_samp]

        save_path = os.path.join(NORMAL_OUTPUT, f"norm_{os.path.splitext(fname)[0]}.npy")
        np.save(save_path, segment)
        normal_segments.append(segment.shape)

    return normal_segments

def plot_statistics(epi_segs, norm_segs):
    epi_count = len(epi_segs)
    norm_count = len(norm_segs)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['癫痫发作前片段', '正常片段'], [epi_count, norm_count], color=['#ff4c4c', '#4c8cff'])
    plt.title('脑电数据片段统计', fontsize=14)
    plt.ylabel('片段数量', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', ha='center', fontsize=12)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("="*60)
    print("数据切割完成！")
    print(f"癫痫发作前 5 分钟片段：{epi_count} 个")
    print(f"正常 3-8 分钟片段：{norm_count} 个")
    print("="*60)

# ===================== 工具函数：模型训练 =====================
def z_score_normalize(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std

def clean_segment(seg):
    if seg.shape[0] == 13:
        seg = seg[:12, :]
    seg = seg.astype(str)
    mask = np.char.isdigit(seg) | np.char.startswith(seg, '-') | np.char.equal(seg, '.')
    seg[~mask] = 0.0
    return seg.astype(np.float32)

def cut_to_samples(segment):
    segment = clean_segment(segment)
    total_len = segment.shape[1]
    samples = []
    if total_len < WINDOW_LEN:
        return samples
    for i in range(0, total_len - WINDOW_LEN + 1, STEP_LEN):
        samples.append(segment[:, i:i+WINDOW_LEN])
    return samples

def load_all_segments(folder, label_name, max_count=None):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if max_count:
        files = np.random.choice(files, max_count, replace=False)

    samples = []
    print(f"正在加载 {label_name} 数据...")
    for idx, f in enumerate(files):
        path = os.path.join(folder, f)
        seg = np.load(path, allow_pickle=True)
        segs = cut_to_samples(seg)
        samples.extend(segs)
        print(f"  已处理：{idx+1}/{len(files)} | 累计片段：{len(samples)}")
    print(f"{label_name} 加载完成，总片段：{len(samples)}\n")
    return samples

# ===================== 模型结构 =====================
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
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels),
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
        x = self.dte(self.conv(x)).transpose(1,2)
        return self.norm(x + self.attn(x,x,x)[0])

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels,32,3,padding=1)
        self.c2 = nn.Conv1d(in_channels,32,5,padding=2)
        self.c3 = nn.Conv1d(in_channels,32,7,padding=3)
        self.cel = ChannelExcitation(96)
    def forward(self, x):
        return self.cel(torch.cat([self.c1(x),self.c2(x),self.c3(x)],1)).mean(-1)

class DMSSTAN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dta = DynamicTemporalAttention(in_channels)
        self.mssa = MultiScaleSpatialAttention(in_channels)
        self.cls = nn.Sequential(
            nn.Linear(64+96,128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,2)
        )
    def forward(self, x):
        return self.cls(torch.cat([self.dta(x).mean(1), self.mssa(x)],1))

# ===================== 主执行：先切割 → 再训练 =====================
if __name__ == "__main__":
    # ========== 第一步：切割数据 ==========
    print("【1/3】开始切割癫痫数据...")
    epi_segments = extract_epilepsy_segments()

    print("【2/3】开始切割正常数据...")
    norm_segments = extract_normal_segments()

    print("【3/3】数据统计...")
    plot_statistics(epi_segments, norm_segments)

    # ========== 第二步：加载片段并训练 ==========
    print("\n==================== 开始训练模型 ====================\n")

    X_neg = load_all_segments(NORMAL_OUTPUT, "正常", max_count=28)
    X_pos = load_all_segments(EPILEPSY_OUTPUT, "癫痫", max_count=28)

    n = min(len(X_pos), len(X_neg))
    X_pos = X_pos[:n]
    X_neg = X_neg[:n]

    print("="*60)
    print(f"正负样本 1:1 平衡完成")
    print(f"癫痫片段：{len(X_pos)} | 正常片段：{len(X_neg)}")
    print(f"总训练样本：{len(X_pos)+len(X_neg)}")
    print("="*60)

    X = [z_score_normalize(x) for x in X_pos + X_neg]
    y = [1]*n + [0]*n

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # 划分训练集/测试集
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    X_train = torch.tensor(X_train, device=DEVICE)
    X_test = torch.tensor(X_test, device=DEVICE)
    y_train = torch.tensor(y_train, device=DEVICE)
    y_test = torch.tensor(y_test, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # 模型初始化
    model = DMSSTAN(N_CHANNELS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_acc, counter = 0, 0
    print("\n开始训练...\n")

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.eval()
        pred, true = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                pred.extend(model(bx).argmax(1).cpu().numpy())
                true.extend(by.cpu().numpy())

        acc = accuracy_score(true, pred)
        print(f"Epoch {epoch+1:2d} | Loss: {loss_sum:.2f} | Acc: {acc:.4f}")

        if acc > best_acc + 1e-4:
            best_acc = acc
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\n连续{PATIENCE}轮无提升，提前停止训练")
                break

    # 评估
    print("\n训练完成，加载最优模型评估...\n")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    yt, yp, ysc = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx)
            yt.extend(by.cpu().numpy())
            yp.extend(out.argmax(1).cpu().numpy())
            ysc.extend(torch.softmax(out,1)[:,1].cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()

    print("="*50)
    print("最终模型指标")
    print("="*50)
    print(f"ACC  : {((tp+tn)/(tn+fp+fn+tp)):.4f}")
    print(f"SEN  : {(tp/(tp+fn+1e-8)):.4f}")
    print(f"SPEC : {(tn/(tn+fp+1e-8)):.4f}")
    print(f"AUC  : {roc_auc_score(yt, ysc):.4f}")
    print("="*50)