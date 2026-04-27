# -*- coding: utf-8 -*-
# BiLSTM Model for EEG Classification (IF + Spectral Entropy)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from scipy.signal import hilbert, stft
from tqdm import tqdm


# ====================== 瞬时频率 ======================
def compute_instfreq(x, fs=1000):
    analytic = hilbert(x)
    phase = np.angle(analytic)
    phase = np.unwrap(phase)
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])
    return inst_freq


# ====================== 谱熵 ======================
def compute_pentropy(x, fs=1000, nperseg=256):
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg)
    power = np.abs(Zxx) ** 2
    eps = 1e-10
    entropy = []
    for frame in power.T:
        frame = frame / (np.sum(frame) + eps)
        ent = -np.sum(frame * np.log2(frame + eps))
        entropy.append(ent)
    entropy = np.array(entropy)
    entropy = np.interp(np.linspace(0, 1, len(x)),
                        np.linspace(0, 1, len(entropy)), entropy)
    return entropy


# ====================== 特征提取 ======================
def extract_features(signals, fs=1000):
    features = []
    for sig in tqdm(signals, desc="Extracting IF+SE"):
        if_feat = compute_instfreq(sig, fs=fs)
        se_feat = compute_pentropy(sig, fs=fs)
        feat = np.stack([if_feat, se_feat], axis=1)
        features.append(feat)
    return np.array(features)


# ====================== BiLSTM 模型 ======================
def build_bilstm(sequence_len=1000, input_features=2, num_classes=3, lr=0.001, clipnorm=1.0):
    inputs = Input(shape=(sequence_len, input_features))
    x = Bidirectional(LSTM(100, return_sequences=False))(inputs)
    x = Dense(num_classes)(x)
    outputs = Softmax()(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=clipnorm),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model