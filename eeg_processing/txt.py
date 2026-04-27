import os
import cv2
from ultralytics import YOLO
from labels_detection import load_yolo_seizure_model

# ===================== 【你的项目路径 完全一致】 =====================
BASE_DIR = r"D:\python\pythonProject\EEG_CNN-LSTM"

# 波形图路径（yolo_dataset.py生成的）
IMAGE_ICTAL_DIR = os.path.join(BASE_DIR, "Data", "clean_data", "ictal_epilepsy")

# 输出txt路径（你空着的那个文件夹）
TXT_OUTPUT_DIR = os.path.join(BASE_DIR, "DataLoader", "time", "ictal_txt")
os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

# 固定参数
IMG_WIDTH = 1000
TOTAL_SAMPLES = 600000
SEIZURE_CLASS = 0

# ===================== 核心：批量生成 YOLO txt 标注 =====================
def batch_generate_ictal_txt():
    # 加载你训练好的YOLO模型
    model = load_yolo_seizure_model()
    print("YOLO模型加载成功！开始检测癫痫发作并生成txt...\n")

    # 遍历所有癫痫图片
    img_files = sorted([f for f in os.listdir(IMAGE_ICTAL_DIR) if f.endswith(('.jpg', '.png'))])

    count = 0
    for img_name in img_files:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(IMAGE_ICTAL_DIR, img_name)
        txt_path = os.path.join(TXT_OUTPUT_DIR, f"{base}.txt")

        # YOLO检测
        img = cv2.imread(img_path)
        results = model(img_path, verbose=False)

        x1_list = []
        x2_list = []

        for res in results:
            for box in res.boxes:
                if int(box.cls[0]) == SEIZURE_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1_list.append(float(x1))
                    x2_list.append(float(x2))

        if len(x1_list) == 0:
            print(f"⚠️  {img_name} 未检测到发作")
            continue

        # 计算YOLO格式
        start_x = min(x1_list)
        end_x = max(x2_list)

        x_center = (start_x + end_x) / 2.0 / IMG_WIDTH
        width = (end_x - start_x) / IMG_WIDTH

        # 写入txt（和你原来的代码完全兼容）
        yolo_line = f"{SEIZURE_CLASS} {x_center:.6f} 0.5 {width:.6f} 1.0\n"
        with open(txt_path, 'w') as f:
            f.write(yolo_line)

        count += 1
        print(f"✅ 已生成：{txt_path}")

    print(f"\n🎉 全部完成！共生成 {count} 个 txt 标注")

if __name__ == "__main__":
    batch_generate_ictal_txt()