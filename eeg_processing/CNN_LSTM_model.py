# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def extract_features(signals):
    features = []
    for sig in tqdm(signals):
        features.append(sig.reshape(-1, 1))
    return np.array(features)

def build_cnn_bilstm(input_length=2000, learning_rate=0.001):
    inputs = Input(shape=(input_length, 1))

    # 强CNN
    x = Conv1D(64, 10, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 10, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 10, activation='relu')(x)
    x = MaxPooling1D(2)(x)

    # 强LSTM
    x = Bidirectional(LSTM(128, return_sequences=False))(x)

    # 分类头
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model