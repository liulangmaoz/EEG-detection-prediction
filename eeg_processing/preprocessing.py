import numpy as np
import mne
from scipy import signal

def preprocess_signal(lfp_data, fs, interactive=False):
    """
    完整预处理流水线：
    1. 带通滤波 0.5-64Hz
    2. 49-51Hz 工频带阻滤波
    3. ICA去伪迹 + 平均重参考
    4. 基线校正
    :param lfp_data: 脑电数据 (n_samples, n_channels)
    :param fs: 采样率
    :param interactive: 是否启用交互式ICA成分选择
    :return: 预处理后的脑电数据
    """
    print("\n===== 开始信号预处理 (0.5-64Hz) =====")
    print(f"原始数据形状: {lfp_data.shape}")
    
    n_samples, n_channels = lfp_data.shape
    processed_data = np.zeros_like(lfp_data)
    nyq = fs / 2
    low_cut = 0.5
    high_cut = 64.0
    
    # 带通滤波 + 工频带阻滤波
    sos_bandpass = signal.butter(4, [low_cut / nyq, high_cut / nyq], btype='bandpass', output='sos')
    sos_bandstop = signal.butter(4, [49/nyq, 51/nyq], btype='bandstop', output='sos')
    
    print(f"应用带通滤波: {low_cut}Hz - {high_cut}Hz (SOS滤波)")
    print(f"应用工频带阻滤波: 49Hz - 51Hz")
    
    for ch_idx in range(n_channels):
        clean_signal = signal.sosfiltfilt(sos_bandpass, lfp_data[:, ch_idx])
        clean_signal = signal.sosfiltfilt(sos_bandstop, clean_signal)
        processed_data[:, ch_idx] = clean_signal
        
        from .core import CHANNEL_NAMES
        if ch_idx < len(CHANNEL_NAMES):
            print(f"通道{ch_idx + 1}({CHANNEL_NAMES[ch_idx]}) 预处理完成")
    
    print("===== 滤波完成 =====\n")
    
    # ICA 去伪迹与重参考处理
    print("===== 开始 ICA 去伪迹与重参考处理 =====")
    from .core import CHANNEL_NAMES
    ch_names = CHANNEL_NAMES
    n_samples_ica, n_channels_ica = processed_data.shape
    ch_types = ['eeg'] * n_channels_ica
    
    # 构建MNE对象
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(processed_data.T.astype(np.float64), info)
    
    # 虚拟电极位置
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='warn')
    
    # 重参考
    raw.set_eeg_reference(ref_channels=[], verbose=False)
    try:
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
        print("平均重参考完成")
    except:
        print("重参考失败")
    
    # ICA 拟合
    print("\n运行 ICA 拟合...")
    ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter='auto')
    ica.fit(raw, verbose=False)
    print("ICA 拟合完成！")
    
    # 画图
    if interactive:
        print("\n打开 ICA 成分拓扑图")
        try:
            ica.plot_components(title="ICA 成分拓扑图", show=True)
        except:
            print("拓扑图绘制失败，跳过")
        
        print("打开 ICA 成分时间序列")
        ica.plot_sources(raw, block=True, show=True)
        
        # 输入剔除成分
        exclude_ics = input("\n>>> 请输入要排除的 ICA 成分编号（逗号分隔），无则回车：")
        if exclude_ics.strip():
            ica.exclude = [int(ic.strip()) for ic in exclude_ics.split(',')]
            print(f"已剔除成分：{ica.exclude}")
            ica.apply(raw, verbose=False)
    else:
        print("非交互式模式：跳过ICA成分可视化和手动选择")
    
    # 转回numpy
    clean_lfp_mne, _ = raw[:]
    clean_lfp = clean_lfp_mne.T
    
    # 基线校正
    print("\n执行基线校正（每个通道独立去均值）...")
    clean_lfp = clean_lfp - np.mean(clean_lfp, axis=0, keepdims=True)
    
    print("\n===== 【全部预处理完成】 =====\n")
    return clean_lfp
