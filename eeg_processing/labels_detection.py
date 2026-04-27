import numpy as np
from ultralytics import YOLO
import cv2

def load_yolo_seizure_model():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "runs/detect/train4/weights/best.pt")
    return YOLO(model_path)

def generate_3phase_labels_yolo(lfp_data, model, image_path, fs=1000, total_samples=600000):
    n_samples = lfp_data.shape[0]
    time_axis = np.arange(n_samples) / fs

    img = cv2.imread(image_path)
    img_width = img.shape[1]

    results = model(image_path, verbose=False)
    x1_all = []
    x2_all = []

    for res in results:
        for box in res.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1_all.append(float(x1))
                x2_all.append(float(x2))

    if len(x1_all) == 0:
        return np.zeros(n_samples, dtype=int), time_axis

    start_x = min(x1_all)
    end_x = max(x2_all)

    start_samp = (start_x / img_width) * total_samples
    end_samp = (end_x / img_width) * total_samples

    global_start = start_samp / fs
    global_end = end_samp / fs

    labels = np.zeros(n_samples, dtype=int)
    labels[(time_axis >= global_start) & (time_axis <= global_end)] = 1

    return labels, time_axis

def generate_normal_label(lfp_data, fs=1000):
    length = lfp_data.shape[0]
    labels = np.zeros(length, dtype=int)
    time_axis = np.arange(length) / fs
    return labels, time_axis

# cd D:\python\pythonProject\EEG_CNN-LSTM
# python main.py --model lstm
# python main.py --model cnn
# python main.py --model cnn_lstm
# ctrl + c