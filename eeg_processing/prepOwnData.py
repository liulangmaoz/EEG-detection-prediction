import numpy as np
from segSignals import segSignals

def prepOwnData(lfp_data, labels, targetLength, channels, fs=1000):
    EEG = lfp_data[:, 1:13].T
    n_channels, n_samples = EEG.shape

    def get_segments(signal, label_vec, target_label):
        mask = label_vec == target_label
        return signal[:, mask]

    seg_N = get_segments(EEG, labels, 0)
    seg_I = get_segments(EEG, labels, 1)

    long_signals = [seg_N, seg_I]
    long_labels = [0, 1]

    def split_5to1(signal_list, label_list):
        train_sig, val_sig = [], []
        train_lbl, val_lbl = [], []

        for sig, lbl in zip(signal_list, label_list):
            n_train = int(sig.shape[1] * 5 / 6)
            train_sig.append(sig[:, :n_train])
            val_sig.append(sig[:, n_train:])
            train_lbl.append(lbl)
            val_lbl.append(lbl)

        return train_sig, train_lbl, val_sig, val_lbl

    train_long, train_lbl, val_long, val_lbl = split_5to1(long_signals, long_labels)

    train_X, train_y = segSignals(train_long, train_lbl, targetLength, channels)
    val_X, val_y = segSignals(val_long, val_lbl, targetLength, channels)

    print("=" * 60)
    print("Data split completed")
    print("=" * 60)
    print(f"Train: {train_X.shape} | 0={(train_y==0).sum()} 1={(train_y==1).sum()}")
    print(f"Val:   {val_X.shape} | 0={(val_y==0).sum()} 1={(val_y==1).sum()}")
    print("=" * 60)

    return train_X, train_y, val_X, val_y