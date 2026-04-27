import os

# ==================== 你指定的路径 ====================
TARGET_DIR = r"D:\python\pythonProject\EEG_CNN-LSTM\DataLoader\time\normal_txt"

# 自动创建文件夹（如果不存在）
os.makedirs(TARGET_DIR, exist_ok=True)

# ==================== 创建 1.txt ~ 28.txt ====================
for i in range(1, 29):  # 1 到 28
    txt_path = os.path.join(TARGET_DIR, f"{i}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        pass  # 空白文件

# ==================== 完成提示 ====================
print(" 成功创建 28 个空白 txt 文件！")
print(f"位置：{TARGET_DIR}")
print(" 文件名：1.txt ~ 28.txt")