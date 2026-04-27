# -*- coding: utf-8 -*-
# Time-Frequency Image Generator

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import stft

def generate_tf_images(signals, labels, save_dir, fs=1000):
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(signals)), desc="Generating TF Images"):
        sig = signals[i]
        lbl = int(labels[i])
        class_dir = os.path.join(save_dir, str(lbl))
        os.makedirs(class_dir, exist_ok=True)

        f, t, Zxx = stft(sig, fs=fs, nperseg=224)
        Zxx = np.abs(Zxx)

        plt.figure(figsize=(2.24, 2.24))
        plt.pcolormesh(Zxx, cmap="jet")
        plt.axis("off")
        plt.savefig(
            os.path.join(class_dir, f"{i}.png"),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close("all")