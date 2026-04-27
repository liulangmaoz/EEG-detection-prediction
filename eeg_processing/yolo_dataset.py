import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# CSV 数据路径
CSV_ICTAL_DIR = os.path.join(PROJECT_ROOT, "Data", "clean_data", "ictal_epilepsy")
CSV_NORMAL_DIR = os.path.join(PROJECT_ROOT, "Data", "clean_data", "interictal_normal")

# 输出图片路径
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "Data", "image")
IMAGE_ICTAL = os.path.join(IMAGE_ROOT, "ictal_epilepsy")
IMAGE_NORMAL = os.path.join(IMAGE_ROOT, "interictal_normal")
os.makedirs(IMAGE_ICTAL, exist_ok=True)
os.makedirs(IMAGE_NORMAL, exist_ok=True)

# YOLO 数据集输出
YOLO_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset")
for subdir in ["images/train", "images/test", "images/val",
               "labels/train", "labels/test", "labels/val"]:
    os.makedirs(os.path.join(YOLO_DIR, subdir), exist_ok=True)

# ===================== 【新增核心】CSV → EEG 波形图 =====================
def generate_eeg_image_from_csv(csv_path, save_path, fs=1000):
    """
    从 CSV 读取 LFP 信号 → 生成 EEG 波形图
    命名完全对应：1.csv → 1.jpg
    """
    df = pd.read_csv(csv_path)
    lfp = df.values  # shape: [n_samples, 12]

    plt.figure(figsize=(16, 8))
    for ch in range(min(12, lfp.shape[1])):
        offset = ch * 10
        plt.plot(lfp[:, ch] + offset, linewidth=0.8)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

# ===================== 批量生成：CSV → JPG（一一对应） =====================
def generate_all_images():
    print("正在生成癫痫数据图片...")
    for fname in os.listdir(CSV_ICTAL_DIR):
        if fname.endswith(".csv"):
            base = os.path.splitext(fname)[0]
            csv_file = os.path.join(CSV_ICTAL_DIR, fname)
            jpg_file = os.path.join(IMAGE_ICTAL, f"{base}.jpg")
            generate_eeg_image_from_csv(csv_file, jpg_file)

    print("正在生成正常数据图片...")
    for fname in os.listdir(CSV_NORMAL_DIR):
        if fname.endswith(".csv"):
            base = os.path.splitext(fname)[0]
            csv_file = os.path.join(CSV_NORMAL_DIR, fname)
            jpg_file = os.path.join(IMAGE_NORMAL, f"{base}.jpg")
            generate_eeg_image_from_csv(csv_file, jpg_file)

    print("✅ 所有 CSV 已生成对应 JPG 图片，命名完全对应！\n")

# ===================== 随机划分 + 复制 =====================
def split_and_copy(
    img_folder, lbl_folder,
    val_num=3, test_num=5,
    prefix="",
    seed=42
):
    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))])
    random.seed(seed)
    random.shuffle(img_files)

    val_files = img_files[:val_num]
    test_files = img_files[val_num : val_num + test_num]
    train_files = img_files[val_num + test_num :]

    def safe_copy(files, prefix, img_dst, lbl_dst):
        for i, f in enumerate(files, 1):
            base = os.path.splitext(f)[0]
            new_name = f"{prefix}_{i}"
            ext = os.path.splitext(f)[1]

            shutil.copy(os.path.join(img_folder, f), os.path.join(img_dst, new_name + ext))
            shutil.copy(os.path.join(lbl_folder, base + ".txt"), os.path.join(lbl_dst, new_name + ".txt"))

    img_train = os.path.join(YOLO_DIR, "images/train")
    img_test = os.path.join(YOLO_DIR, "images/test")
    img_val = os.path.join(YOLO_DIR, "images/val")

    lbl_train = os.path.join(YOLO_DIR, "labels/train")
    lbl_test = os.path.join(YOLO_DIR, "labels/test")
    lbl_val = os.path.join(YOLO_DIR, "labels/val")

    safe_copy(train_files, prefix, img_train, lbl_train)
    safe_copy(test_files, prefix, img_test, lbl_test)
    safe_copy(val_files, prefix, img_val, lbl_val)

    return len(train_files), len(test_files), len(val_files)

# ===================== 执行 =====================
if __name__ == "__main__":
    # 第一步：CSV → 对应图片
    generate_all_images()

    # 第二步：划分 YOLO 数据集
    train_sz, test_sz, val_sz = split_and_copy(
        img_folder=IMAGE_ICTAL,
        lbl_folder=os.path.join(BASE_DIR, "time", "ictal_txt"),
        val_num=3, test_num=5, prefix="seizure"
    )

    train_nm, test_nm, val_nm = split_and_copy(
        img_folder=IMAGE_NORMAL,
        lbl_folder=os.path.join(BASE_DIR, "time", "normal_txt"),
        val_num=3, test_num=5, prefix="normal"
    )

    print("随机数据集构建完成（纯随机、无命名冲突、标签一一对应）")
    print("=" * 65)
    print(f"训练集：癫痫 {train_sz} 张 | 正常 {train_nm} 张")
    print(f"测试集：癫痫 {test_sz} 张 | 正常 {test_nm} 张")
    print(f"验证集：癫痫 {val_sz} 张 | 正常 {val_nm} 张")
    print("=" * 65)