# -*- coding: utf-8 -*-
# EEG Epilepsy Two-Class Classification
# Professional Training Script for CNN / LSTM / CNN-BiLSTM
# 重复保存scaler报错
# ROC曲线 + AUC + 保存预测结果 + 顶刊绘图

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, roc_auc_score
)

# ============================ Path Setting ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "DataLoader"))

# ============================ Parser ============================
parser = argparse.ArgumentParser(description='EEG Epilepsy Classification (CNN, LSTM, CNN-LSTM)')

parser.add_argument('--model', type=str, default='cnn_lstm', help='model: [cnn, lstm, cnn_lstm]')
parser.add_argument('--data', type=str, default='eeg_all', help='dataset name')
parser.add_argument('--root_path', type=str, default=BASE_DIR, help='root path of project')
parser.add_argument('--ictal_dir', type=str, default='Data/clean_data/ictal_epilepsy', help='epilepsy data')
parser.add_argument('--normal_dir', type=str, default='Data/clean_data/interictal_normal', help='normal data')
parser.add_argument('--fs', type=int, default=1000, help='sampling rate')
parser.add_argument('--seq_len_cnn', type=int, default=2000, help='sequence length for CNN')
parser.add_argument('--seq_len_lstm', type=int, default=2000, help='sequence length for LSTM')
parser.add_argument('--seq_len_cnnlstm', type=int, default=2000, help='sequence length for CNN-LSTM')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs_cnn', type=int, default=15, help='training epochs for CNN')
parser.add_argument('--epochs_lstm', type=int, default=12, help='training epochs for LSTM')
parser.add_argument('--epochs_cnnlstm', type=int, default=12, help='training epochs for CNN-LSTM')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--img_size', nargs='+', type=int, default=[224, 224], help='input image size')
parser.add_argument('--image_folder', type=str, default='image_dataset', help='folder to save images')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
parser.add_argument('--channels', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11], help='used channels')
parser.add_argument('--samples_per_class', type=int, default=3600000, help='max samples per class')

args = parser.parse_args()

# ============================ Auto Path Completion ============================
args.ictal_dir = os.path.join(args.root_path, args.ictal_dir)
args.normal_dir = os.path.join(args.root_path, args.normal_dir)
args.image_folder = os.path.join(args.root_path, args.image_folder)
args.train_folder = os.path.join(args.image_folder, 'train')
args.val_folder = os.path.join(args.image_folder, 'val')
args.img_size = tuple(args.img_size)

data_save_path = os.path.join(args.root_path, "eeg_combined_data.npz")

os.makedirs(os.path.join(args.root_path, "saved_models"), exist_ok=True)
os.makedirs(os.path.join(args.root_path, "training_logs"), exist_ok=True)
os.makedirs(os.path.join(args.root_path, "plots"), exist_ok=True)
os.makedirs(os.path.join(args.root_path, "predictions"), exist_ok=True)

# ============================ Print Args ============================
print('=' * 80)
print('           Args in Experiment')
print('=' * 80)
print(args)

# ============================ Load Data & Tools ============================
from DataLoader.labels_detection import load_yolo_seizure_model, generate_3phase_labels_yolo
from DataLoader.labels_detection import generate_normal_label
from DataLoader.prepOwnData import prepOwnData

model_yolo = load_yolo_seizure_model()

# ============================ Load or Create Combined Data ============================
if os.path.exists(data_save_path):
    print("\n Loading saved combined data...")
    data = np.load(data_save_path)
    lfp_data = data['lfp']
    labels = data['labels']
else:
    print("\n Creating and saving combined data...")
    all_lfp = []
    all_labels = []

    for fname in os.listdir(args.ictal_dir):
        if fname.endswith('.csv'):
            path = os.path.join(args.ictal_dir, fname)
            img_name = os.path.splitext(fname)[0] + ".jpg"
            img_path = os.path.join(args.ictal_dir, img_name)
            print(f'Loading ictal: {path}')
            df = pd.read_csv(path)
            lfp = df.values
            lab, _ = generate_3phase_labels_yolo(lfp_data=lfp, model=model_yolo, image_path=img_path, fs=args.fs)
            all_lfp.append(lfp)
            all_labels.append(lab)

    for fname in os.listdir(args.normal_dir):
        if fname.endswith('.csv'):
            path = os.path.join(args.normal_dir, fname)
            print(f'Loading normal: {path}')
            df = pd.read_csv(path)
            lfp = df.values
            lab, _ = generate_normal_label(lfp, fs=args.fs)
            all_lfp.append(lfp)
            all_labels.append(lab)

    lfp_data = np.concatenate(all_lfp, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.savez_compressed(data_save_path, lfp=lfp_data, labels=labels)
    print(f" Data saved to {data_save_path}")

print(f"\n Total data length: {lfp_data.shape[0]}")
print(f" Label distribution: {np.unique(labels, return_counts=True)}")

# ============================ Balanced Sampling ============================
indices = []
for c in [0, 1]:
    idx = np.where(labels == c)[0][:args.samples_per_class]
    indices.append(idx)
indices = np.concatenate(indices)
lfp_data = lfp_data[indices]
labels = labels[indices]

print(f"\n After balanced sampling: {lfp_data.shape[0]}")
print(f" Balanced labels: {np.unique(labels, return_counts=True)}")

# ============================ Training Callbacks ============================
def get_callbacks(model_name):
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.root_path, "saved_models", f"best_{model_name}.keras"),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    return [early_stop, checkpoint]

# ============================ 绘图函数：ROC 曲线 ============================
def plot_roc(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='#A2A3AA', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    plt.title(f'ROC Curve - {model_name.upper()}', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(args.root_path, "plots", f"{model_name}_roc_curve.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nROC 曲线已保存至: {save_path}")
    return roc_auc

# ==============================================================================================
# ======================================= CNN Model ============================================
# ==============================================================================================
if args.model == 'cnn':
    from model.CNN_ImageModel import build_cnn_image_model
    from model.ImageGenerator import generate_tf_images
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print('\n' + '=' * 80)
    print('               CNN Training Start')
    print('=' * 80)

    train_X, train_y, val_X, val_y = prepOwnData(lfp_data, labels, args.seq_len_cnn, args.channels)
    generate_tf_images(train_X, train_y, args.train_folder, fs=args.fs)
    generate_tf_images(val_X, val_y, args.val_folder, fs=args.fs)

    datagen = ImageDataGenerator()
    train_gen = datagen.flow_from_directory(args.train_folder, target_size=args.img_size, batch_size=args.batch_size, class_mode='categorical')
    val_gen = datagen.flow_from_directory(args.val_folder, target_size=args.img_size, batch_size=args.batch_size, class_mode='categorical')

    model = build_cnn_image_model(args.img_size, args.num_classes)
    model.summary()
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs_cnn, callbacks=get_callbacks("cnn"), verbose=1)

    y_pred_prob = model.predict(val_gen, verbose=1)[:,1]
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen), axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc_score = plot_roc(y_true, y_pred_prob, "cnn")

    print("\n" + "="*60)
    print("             CNN 模型综合评价指标")
    print("="*60)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"F1 分数            : {f1:.4f}")
    print(f"AUC                : {auc_score:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("="*60)

# ==============================================================================================
# ======================================= LSTM Model ===========================================
# ==============================================================================================
elif args.model == 'lstm':
    from model.LSTM_model import extract_features, build_bilstm

    print('\n' + '=' * 80)
    print('               LSTM Training Start')
    print('=' * 80)

    train_X, train_y, val_X, val_y = prepOwnData(lfp_data, labels, args.seq_len_lstm, args.channels)
    XTrain = extract_features(train_X)
    XVal = extract_features(val_X)

    mu = np.mean(np.concatenate(XTrain, axis=0), axis=0)
    sg = np.std(np.concatenate(XTrain, axis=0), axis=0) + 1e-12
    XTrain = (XTrain - mu) / sg
    XVal = (XVal - mu) / sg

    np.savez(os.path.join(BASE_DIR, "saved_models", "scaler_lstm.npz"), mu=mu, sg=sg)

    model = build_bilstm(sequence_len=args.seq_len_lstm, num_classes=args.num_classes, lr=args.learning_rate)
    model.summary()
    history = model.fit(XTrain, train_y, validation_data=(XVal, val_y), epochs=args.epochs_lstm, batch_size=args.batch_size, callbacks=get_callbacks("lstm"), verbose=1)

    y_pred_prob = model.predict(XVal, verbose=1).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = val_y.flatten()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc_score = plot_roc(y_true, y_pred_prob, "lstm")

    print("\n" + "="*60)
    print("             LSTM 模型综合评价指标")
    print("="*60)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"F1 分数            : {f1:.4f}")
    print(f"AUC                : {auc_score:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("="*60)

# ==============================================================================================
# ===================================== CNN-LSTM Model =========================================
# ==============================================================================================
elif args.model == 'cnn_lstm':
    from model.CNN_LSTM_model import extract_features, build_cnn_bilstm

    print('\n' + '=' * 80)
    print('               CNN-LSTM Training Start')
    print('=' * 80)

    train_X, train_y, val_X, val_y = prepOwnData(lfp_data, labels, args.seq_len_cnnlstm, args.channels)
    XTrain = extract_features(train_X)
    XVal = extract_features(val_X)

    mu = np.mean(np.concatenate(XTrain, axis=0), axis=0)
    sg = np.std(np.concatenate(XTrain, axis=0), axis=0) + 1e-12
    XTrain = (XTrain - mu) / sg
    XVal = (XVal - mu) / sg

    # ===================== 只保存一次，修复BUG =====================
    np.savez(os.path.join(BASE_DIR, "saved_models", "scaler_cnn_lstm.npz"), mu=mu, sg=sg)

    model = build_cnn_bilstm(input_length=args.seq_len_cnnlstm, learning_rate=args.learning_rate)
    model.summary()

    history = model.fit(
        XTrain, train_y,
        validation_data=(XVal, val_y),
        epochs=args.epochs_cnnlstm,
        batch_size=args.batch_size,
        callbacks=get_callbacks("cnn_lstm"),
        verbose=1
    )

    y_pred_prob = model.predict(XVal, verbose=1).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = val_y.flatten()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    cm = confusion_matrix(y_true, y_pred)
    auc_score = plot_roc(y_true, y_pred_prob, "cnn_lstm")

    print("\n" + "="*60)
    print("             CNN-LSTM 模型综合评价指标")
    print("="*60)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"F1 分数            : {f1:.4f}")
    print(f"AUC                : {auc_score:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("="*60)

# ============================ Save Log & Plot ============================
if args.model in ['lstm', 'cnn_lstm', 'cnn']:
    log_df = pd.DataFrame(history.history)
    log_df.to_csv(os.path.join(args.root_path, "training_logs", f"{args.model}_log.csv"), index=False)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.root_path, "plots", f"{args.model}_curve.png"), dpi=300)

    # ===================== 保存预测结果 ======================
    np.savez(os.path.join(args.root_path, "predictions", f"{args.model}_val_results.npz"),
             y_true=y_true, y_pred_prob=y_pred_prob, y_pred=y_pred)

print('\nAll training completed successfully ')
print(f"ROC 曲线、预测数据、训练曲线均已保存")

# cd D:\python\pythonProject\EEG_CNN-LSTM
# python main.py --model lstm
# python main.py --model cnn
# python main.py --model cnn_lstm
# ctrl + c