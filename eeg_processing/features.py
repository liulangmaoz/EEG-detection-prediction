import numpy as np
from scipy import signal
from scipy.integrate import simpson
import os
import datetime
import pandas as pd
from tqdm import tqdm
import pyinform as pi
from scipy.signal import argrelextrema, resample,correlate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KDTree
import nolds

# 特征提取函数
def calculate_main_frequency_power(eeg_segment, fs):
    """
    计算脑电片段的主频和对应功率值
    :param eeg_segment: 脑电片段
    :param fs: 采样率
    :return: 主频和功率值
    """
    nperseg = int(fs * 2)
    noverlap = int(fs * 1)
    freq, psd = signal.welch(eeg_segment, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density')

    valid_freq_mask = (freq >= 0.5) & (freq <= 64.0)
    valid_freq = freq[valid_freq_mask]
    valid_psd = psd[valid_freq_mask]

    if len(valid_freq) == 0 or len(valid_psd) == 0:
        return None, None

    max_power_idx = np.argmax(valid_psd)
    main_frequency = round(valid_freq[max_power_idx], 2)
    main_power = round(valid_psd[max_power_idx], 6)

    return main_frequency, main_power


def calculate_alpha_beta_main_power(eeg_segment, fs):
    """
    计算α波和β波的主频与功率
    :param eeg_segment: 脑电片段
    :param fs: 采样率
    :return: α波、β波和联合频段的功率信息
    """
    nperseg = int(fs * 2)
    noverlap = int(fs * 1)
    freq, psd = signal.welch(eeg_segment, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density', window='hann')

    def _extract_band_info(freq_band):
        low, high = freq_band
        band_mask = (freq >= low) & (freq <= high)
        band_freq = freq[band_mask]
        band_psd = psd[band_mask]

        if len(band_freq) == 0 or len(band_psd) == 0:
            return None, None, 0.0

        max_power_idx = np.argmax(band_psd)
        main_freq = round(band_freq[max_power_idx], 2)
        main_power = round(band_psd[max_power_idx], 6)
        total_band_power = round(np.sum(band_psd) * (band_freq[1] - band_freq[0]), 6)

        return main_freq, main_power, total_band_power

    from .core import ALPHA_BAND, BETA_BAND
    alpha_freq, alpha_power, alpha_total = _extract_band_info(ALPHA_BAND)
    beta_freq, beta_power, beta_total = _extract_band_info(BETA_BAND)
    combined_band = (ALPHA_BAND[0], BETA_BAND[1])
    combined_freq, combined_power, combined_total = _extract_band_info(combined_band)

    return {
        'alpha': {'freq': alpha_freq, 'power': alpha_power, 'total_power': alpha_total},
        'beta': {'freq': beta_freq, 'power': beta_power, 'total_power': beta_total},
        'combined': {'freq': combined_freq, 'power': combined_power, 'total_power': combined_total},
        'freq_bands': {'alpha': ALPHA_BAND, 'beta': BETA_BAND}
    }


def calculate_power_spectral_entropy(eeg_segment, fs, freq_band=None):
    """
    计算功率谱熵
    :param eeg_segment: 脑电片段
    :param fs: 采样率
    :param freq_band: 频率范围
    :return: 功率谱熵信息
    """
    from .core import ANALYSIS_BAND, ALPHA_BAND, BETA_BAND
    if freq_band is None:
        freq_band = ANALYSIS_BAND

    nperseg = int(fs * 2)
    noverlap = int(fs * 1)
    freq, psd = signal.welch(eeg_segment, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='density')

    band_mask = (freq >= freq_band[0]) & (freq <= freq_band[1])
    valid_freq = freq[band_mask]
    valid_psd = psd[band_mask]

    if len(valid_psd) == 0 or np.sum(valid_psd) == 0:
        return {
            'total_entropy': np.nan,
            'total_entropy_norm': np.nan,
            'alpha_entropy': np.nan,
            'alpha_entropy_norm': np.nan,
            'beta_entropy': np.nan,
            'beta_entropy_norm': np.nan,
            'freq_band': freq_band,
            'valid_freq_num': len(valid_freq)
        }

    total_power = np.sum(valid_psd)
    p = valid_psd / total_power
    p = p + 1e-10
    spectral_entropy = -np.sum(p * np.log(p))

    N = len(valid_freq)
    if N <= 1:
        total_entropy_norm = np.nan
    else:
        max_possible_entropy = np.log(N)
        total_entropy_norm = round(spectral_entropy / max_possible_entropy, 6)

    alpha_entropy = np.nan
    alpha_entropy_norm = np.nan
    alpha_mask = (valid_freq >= ALPHA_BAND[0]) & (valid_freq <= ALPHA_BAND[1])
    alpha_freq = valid_freq[alpha_mask]
    alpha_psd = valid_psd[alpha_mask]
    if np.sum(alpha_psd) > 0:
        p_alpha = alpha_psd / np.sum(alpha_psd)
        p_alpha += 1e-10
        alpha_entropy = -np.sum(p_alpha * np.log(p_alpha))
        N_alpha = len(alpha_freq)
        if N_alpha <= 1:
            alpha_entropy_norm = np.nan
        else:
            max_alpha_entropy = np.log(N_alpha)
            alpha_entropy_norm = round(alpha_entropy / max_alpha_entropy, 6)

    beta_entropy = np.nan
    beta_entropy_norm = np.nan
    beta_mask = (valid_freq >= BETA_BAND[0]) & (valid_freq <= BETA_BAND[1])
    beta_freq = valid_freq[beta_mask]
    beta_psd = valid_psd[beta_mask]
    if np.sum(beta_psd) > 0:
        p_beta = beta_psd / np.sum(beta_psd)
        p_beta += 1e-10
        beta_entropy = -np.sum(p_beta * np.log(p_beta))
        N_beta = len(beta_freq)
        if N_beta <= 1:
            beta_entropy_norm = np.nan
        else:
            max_beta_entropy = np.log(N_beta)
            beta_entropy_norm = round(beta_entropy / max_beta_entropy, 6)

    return {
        'total_entropy': round(spectral_entropy, 6),
        'total_entropy_norm': total_entropy_norm,
        'alpha_entropy': round(alpha_entropy, 6) if not np.isnan(alpha_entropy) else np.nan,
        'alpha_entropy_norm': alpha_entropy_norm,
        'beta_entropy': round(beta_entropy, 6) if not np.isnan(beta_entropy) else np.nan,
        'beta_entropy_norm': beta_entropy_norm,
        'freq_band': freq_band,
        'valid_freq_num': len(valid_freq)
    }


def calculate_eeg_band_energy(eeg_segment, fs=1000, full_band=(0.5, 64.0)):
    """
    计算脑电片段的频段能量
    :param eeg_segment: 脑电片段
    :param fs: 采样率
    :param full_band: 全频段范围
    :return: 频段能量信息
    """
    # 数据预处理
    sos = signal.butter(4, 0.5, btype='high', fs=fs, output='sos')
    eeg_filtered = signal.sosfiltfilt(sos, eeg_segment)

    # 计算功率谱密度
    nperseg = int(fs * 1.0)
    noverlap = int(fs * 0.5)
    freq, psd = signal.welch(eeg_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='density',
                             detrend='constant')

    # 筛选全频段内的PSD
    full_band_mask = (freq >= full_band[0]) & (freq <= full_band[1])
    valid_freq = freq[full_band_mask]
    valid_psd = psd[full_band_mask]

    if len(valid_psd) == 0 or np.sum(valid_psd) < 1e-10:
        from .core import EEG_BANDS
        return {
            'band_energy': {band: np.nan for band in EEG_BANDS.keys()},
            'total_full_band_energy': np.nan,
            'band_relative_ratio': {band: np.nan for band in EEG_BANDS.keys()},
            'freq_resolution': np.nan
        }

    # 各频段能量计算
    from .core import EEG_BANDS
    band_energy = {}
    freq_resolution = freq[1] - freq[0]
    for band_name, (band_low, band_high) in EEG_BANDS.items():
        band_mask = (valid_freq >= band_low) & (valid_freq <= band_high)
        band_psd = valid_psd[band_mask]
        band_freq = valid_freq[band_mask]

        if len(band_psd) == 0:
            band_energy[band_name] = 0.0
            continue

        band_total_energy = simpson(band_psd, band_freq)
        band_energy[band_name] = round(band_total_energy, 6)

    # 全频段总能量计算
    total_full_band_energy = sum(band_energy.values())
    total_full_band_energy = round(total_full_band_energy, 6)

    # 相对占比计算
    band_relative_ratio = {}
    if total_full_band_energy > 1e-10:
        for band_name, energy in band_energy.items():
            ratio = (energy / total_full_band_energy) * 100.0
            band_relative_ratio[band_name] = round(ratio, 4)
    else:
        band_relative_ratio = {band: 0.0 for band in EEG_BANDS.keys()}

    return {
        'band_energy': band_energy,
        'total_full_band_energy': total_full_band_energy,
        'band_relative_ratio': band_relative_ratio,
        'freq_resolution': round(freq_resolution, 6),
        'full_band': full_band,
        'is_valid': total_full_band_energy > 1e-10
    }


# 批量处理函数
def batch_extract_seizure_eeg_data(lfp_data, fs, channel_stats_info, phase_duration=60):
    """
    批量提取发作期脑电数据
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param channel_stats_info: 通道统计信息
    :param phase_duration: 发作期目标时长
    :return: 发作期数据字典
    """
    channel_seizure_eeg_dict = {}
    n_samples, n_channels = lfp_data.shape

    try:
        channel_valid_events = channel_stats_info['channel_valid_events']
    except KeyError as e:
        raise ValueError(f"channel_stats_info 缺少关键键值 {e}，请检查发作检测结果")

    if len(channel_valid_events) != n_channels:
        raise ValueError("通道有效发作事件列表长度与脑电数据通道数不匹配！")

    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        try:
            ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        except:
            ch_name = f"通道{ch_idx}"

        ch_local_events = channel_valid_events[ch_idx]
        if len(ch_local_events) == 0:
            channel_seizure_eeg_dict[ch_name] = []
            continue

        ch_seizure_eeg_list = []
        total_duration_sec = n_samples / fs
        prev_seizure_end_sec = None

        for seizure_event in ch_local_events:
            orig_seizure_start = seizure_event['start_time']
            orig_seizure_end = seizure_event['end_time']

            from .detection import extend_seizure_to_1min
            extended_seizure = extend_seizure_to_1min(
                seizure_start_sec=orig_seizure_start,
                seizure_end_sec=orig_seizure_end,
                fs=fs,
                total_duration_sec=total_duration_sec,
                prev_seizure_end_sec=prev_seizure_end_sec
            )

            if not extended_seizure['is_valid']:
                prev_seizure_end_sec = extended_seizure['end_sec']
                continue

            start_sample = int(extended_seizure['start_sec'] * fs)
            end_sample = int(extended_seizure['end_sec'] * fs)

            start_sample = max(0, start_sample)
            end_sample = min(n_samples, end_sample)
            if start_sample >= end_sample or (end_sample - start_sample) < (fs * 10):
                prev_seizure_end_sec = extended_seizure['end_sec']
                continue

            seizure_eeg_segment = lfp_data[start_sample:end_sample, ch_idx].astype(np.float64)

            ch_seizure_eeg_list.append({
                'seizure_index': len(ch_seizure_eeg_list) + 1,
                'start_sec': extended_seizure['start_sec'],
                'end_sec': extended_seizure['end_sec'],
                'duration_sec': extended_seizure['duration_sec'],
                'original_start_sec': orig_seizure_start,
                'original_end_sec': orig_seizure_end,
                'eeg_data': seizure_eeg_segment
            })

            prev_seizure_end_sec = extended_seizure['end_sec']

        channel_seizure_eeg_dict[ch_name] = ch_seizure_eeg_list
        print(f"===== 通道{ch_name} 发作期数据提取完成 =====")
        print(f"有效发作期数量：{len(ch_seizure_eeg_list)} 个，每个发作期约1分钟\n")

    return channel_seizure_eeg_dict


def batch_extract_full_duration_eeg_data(lfp_data, fs, channel_stats_info=None, phase_duration=None):
    """
    批量提取全时长脑电数据
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param channel_stats_info: 通道统计信息
    :param phase_duration: 相位时长
    :return: 全时长数据字典
    """
    channel_full_eeg_dict = {}
    n_samples, n_channels = lfp_data.shape
    total_duration_sec = n_samples / fs

    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        try:
            ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        except:
            ch_name = f"通道{ch_idx}"

        full_eeg_segment = lfp_data[:, ch_idx].astype(np.float64)

        full_duration_record = {
            'seizure_index': 1,
            'start_sec': 0.0,
            'end_sec': total_duration_sec,
            'duration_sec': total_duration_sec,
            'original_start_sec': 0.0,
            'original_end_sec': total_duration_sec,
            'eeg_data': full_eeg_segment
        }

        channel_full_eeg_dict[ch_name] = [full_duration_record]
        print(f"===== 通道{ch_name} 全时长数据提取完成 =====")
        print(f"数据总时长：{total_duration_sec:.2f} 秒，采样点数量：{len(full_eeg_segment)}\n")

    return channel_full_eeg_dict


def batch_extract_non_seizure_eeg_data(lfp_data, fs, phase_duration=60, total_candidate_duration=300):
    """
    批量提取无癫痫发作数据
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param phase_duration: 目标提取时长
    :param total_candidate_duration: 候选总时长
    :return: 无发作数据字典
    """
    if not isinstance(lfp_data, np.ndarray):
        raise TypeError("lfp_data必须是numpy.ndarray类型的脑电数据")
    if lfp_data.ndim != 2:
        raise ValueError("lfp_data必须是二维数组，形状为[n_samples, n_channels]")
    n_samples, n_channels = lfp_data.shape

    phase_duration = int(phase_duration)
    total_candidate_duration = int(total_candidate_duration)
    if phase_duration != 60:
        print("警告：推荐保持phase_duration=60秒（1分钟），与后续特征计算逻辑一致")
        phase_duration = 60
    if total_candidate_duration < 300 or total_candidate_duration > 360:
        print("警告：total_candidate_duration推荐设置为300-360秒（5-6分钟）")
        total_candidate_duration = 300 if total_candidate_duration < 300 else 360

    total_data_duration = n_samples / fs
    if total_data_duration < phase_duration:
        raise ValueError(f"脑电数据总时长({total_data_duration:.2f}秒)小于目标提取时长({phase_duration}秒)，无法提取")
    if total_data_duration < total_candidate_duration:
        print(
            f"警告：脑电数据总时长({total_data_duration:.2f}秒)小于候选总时长({total_candidate_duration}秒)，将从全部数据中选取1分钟片段")
        total_candidate_duration = int(total_data_duration)

    channel_non_seizure_eeg_dict = {}

    # 截取第5-6分钟的数据
    start_sec = 5 * 60  # 5分钟
    end_sec = 6 * 60  # 6分钟
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)

    # 确保采样点在有效范围内
    start_sample = max(0, start_sample)
    end_sample = min(n_samples, end_sample)

    print("===== 开始提取无癫痫发作数据（每个通道1分钟脑电片段） =====")
    print(f"提取配置：候选时长{total_candidate_duration / 60:.1f}分钟，目标片段1分钟，采样率{fs}Hz")
    print(f"提取时间区间：{start_sec:.2f}秒 ~ {end_sec:.2f}秒\n")

    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        non_seizure_eeg_segment = lfp_data[start_sample:end_sample, ch_idx].astype(np.float64)
        is_valid = len(non_seizure_eeg_segment) >= (phase_duration * fs * 0.9)

        ch_non_seizure_list = []
        if is_valid:
            ch_non_seizure_list.append({
                'seizure_index': 0,
                'start_sec': round(start_sec, 2),
                'end_sec': round(end_sec, 2),
                'duration_sec': round(end_sec - start_sec, 2),
                'original_start_sec': 0.0,
                'original_end_sec': 0.0,
                'eeg_data': non_seizure_eeg_segment
            })
            print(f"通道{ch_name}：提取成功，数据长度{len(non_seizure_eeg_segment)}个采样点")
        else:
            print(f"警告：通道{ch_name}：提取失败，数据长度不足，返回空列表")

        channel_non_seizure_eeg_dict[ch_name] = ch_non_seizure_list

    print(f"\n===== 无发作脑电数据提取完成 =====")
    print(f"有效通道数量：{len([k for k, v in channel_non_seizure_eeg_dict.items() if len(v) > 0])}/{n_channels}")

    return channel_non_seizure_eeg_dict


def batch_calculate_channel_phase_power(lfp_data, fs, channel_stats_info):
    """
    批量计算通道阶段功率
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param channel_stats_info: 通道统计信息
    :return: 通道阶段功率结果
    """
    n_samples, n_channels = lfp_data.shape
    channel_phase_power_results = {}

    channel_valid_events = channel_stats_info['channel_valid_events']
    if len(channel_valid_events) != n_channels:
        raise ValueError("通道有效发作事件列表长度与通道数不匹配！")

    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        ch_local_events = channel_valid_events[ch_idx]

        if len(ch_local_events) == 0:
            continue

        from .detection import divide_seizure_phases
        ch_seizure_phases = divide_seizure_phases(
            channel_stats_info=channel_stats_info,
            fs=fs,
            total_samples=n_samples,
            ch_idx=ch_idx
        )

        if len(ch_seizure_phases) == 0:
            continue

        ch_phase_results = []
        for seizure_phase in ch_seizure_phases:
            seizure_id = seizure_phase['seizure_id']
            current_seizure_results = {'seizure_id': seizure_id, 'phases': []}

            def _process_phase(phase_info, phase_type):
                if phase_info is None or not phase_info['is_valid']:
                    return None

                start_sample = int(phase_info['start_sec'] * fs)
                end_sample = int(phase_info['end_sec'] * fs)

                if start_sample >= n_samples or end_sample <= 0 or start_sample >= end_sample:
                    return None

                eeg_segment = lfp_data[start_sample:end_sample, ch_idx]
                if len(eeg_segment) < fs * 10:
                    return None

                power_info = calculate_alpha_beta_main_power(eeg_segment, fs)

                return {
                    'phase_name': phase_info['phase_name'],
                    'phase_type': phase_type,
                    'start_sec': phase_info['start_sec'],
                    'end_sec': phase_info['end_sec'],
                    'duration_sec': phase_info['duration_sec'],
                    'power_info': power_info
                }

            pre_phase_result = _process_phase(seizure_phase['pre_phase'], '前期')
            if pre_phase_result:
                current_seizure_results['phases'].append(pre_phase_result)

            attack_phase_result = _process_phase(seizure_phase['attack_phase'], '发作期')
            if attack_phase_result:
                current_seizure_results['phases'].append(attack_phase_result)

            post_phase_result = _process_phase(seizure_phase['post_phase'], '后期')
            if post_phase_result:
                current_seizure_results['phases'].append(post_phase_result)

            if len(current_seizure_results['phases']) > 0:
                ch_phase_results.append(current_seizure_results)

        if len(ch_phase_results) > 0:
            channel_phase_power_results[ch_name] = ch_phase_results
            print(f"\n===== 通道{ch_name} （α/β波）主频功率计算完成 =====")

    return channel_phase_power_results


def batch_calculate_non_seizure_phase_power(channel_non_seizure_eeg_dict, fs):
    """
    批量计算无癫痫发作数据的功率
    :param channel_non_seizure_eeg_dict: 无发作数据字典
    :param fs: 采样率
    :return: 无发作功率结果
    """
    channel_non_seizure_power_results = {}

    print("===== 开始计算无癫痫发作数据（α/β波）主频功率 =====")

    for ch_name, non_seizure_list in channel_non_seizure_eeg_dict.items():
        if len(non_seizure_list) == 0:
            continue

        non_seizure_info = non_seizure_list[0]
        eeg_segment = non_seizure_info['eeg_data']
        start_sec = non_seizure_info['start_sec']
        end_sec = non_seizure_info['end_sec']
        duration_sec = non_seizure_info['duration_sec']

        if len(eeg_segment) < fs * 10:
            print(f"警告：通道{ch_name} 数据长度不足10秒，跳过计算")
            continue

        try:
            power_info = calculate_alpha_beta_main_power(eeg_segment, fs)
        except Exception as e:
            print(f"警告：通道{ch_name} 功率计算失败: {str(e)}，跳过该通道")
            continue

        ch_non_seizure_power = {
            'channel_name': ch_name,
            'is_seizure': False,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': duration_sec,
            'power_info': power_info
        }

        channel_non_seizure_power_results[ch_name] = ch_non_seizure_power
        print(f"通道{ch_name}：无发作数据主频功率计算完成")

    print(f"\n===== 无发作数据主频功率计算全部完成 =====")
    print(f"有效计算通道数量：{len(channel_non_seizure_power_results)}/{len(channel_non_seizure_eeg_dict)}")

    return channel_non_seizure_power_results


def batch_calculate_phase_spectral_entropy(lfp_data, fs, channel_stats_info):
    """
    批量计算相位谱熵
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param channel_stats_info: 通道统计信息
    :return: 相位谱熵结果
    """
    n_samples, n_channels = lfp_data.shape
    channel_entropy_results = {}

    channel_valid_events = channel_stats_info['channel_valid_events']

    if len(channel_valid_events) != n_channels:
        raise ValueError("通道有效发作事件列表长度与通道数不匹配！")

    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        ch_local_events = channel_valid_events[ch_idx]

        if len(ch_local_events) == 0:
            continue

        from .detection import divide_seizure_phases
        ch_seizure_phases = divide_seizure_phases(
            channel_stats_info=channel_stats_info,
            fs=fs,
            total_samples=n_samples,
            ch_idx=ch_idx
        )
        if len(ch_seizure_phases) == 0:
            continue

        ch_phase_entropy = []
        for seizure_phase in ch_seizure_phases:
            seizure_id = seizure_phase['seizure_id']
            current_seizure_entropy = {'seizure_id': seizure_id, 'phases': []}

            def _process_phase(phase_info, phase_type):
                if phase_info is None or not phase_info['is_valid']:
                    return None
                start_sample = int(phase_info['start_sec'] * fs)
                end_sample = int(phase_info['end_sec'] * fs)
                if start_sample >= n_samples or end_sample <= 0 or start_sample >= end_sample:
                    return None
                eeg_segment = lfp_data[start_sample:end_sample, ch_idx]
                if len(eeg_segment) < fs * 10:
                    return None
                entropy_info = calculate_power_spectral_entropy(eeg_segment, fs)
                return {
                    'phase_name': phase_info['phase_name'],
                    'phase_type': phase_type,
                    'start_sec': phase_info['start_sec'],
                    'end_sec': phase_info['end_sec'],
                    'duration_sec': phase_info['duration_sec'],
                    'entropy_info': entropy_info
                }

            pre_entropy = _process_phase(seizure_phase['pre_phase'], '前期')
            if pre_entropy:
                current_seizure_entropy['phases'].append(pre_entropy)

            attack_entropy = _process_phase(seizure_phase['attack_phase'], '发作期')
            if attack_entropy:
                current_seizure_entropy['phases'].append(attack_entropy)

            post_entropy = _process_phase(seizure_phase['post_phase'], '后期')
            if post_entropy:
                current_seizure_entropy['phases'].append(post_entropy)

            if len(current_seizure_entropy['phases']) > 0:
                ch_phase_entropy.append(current_seizure_entropy)

        channel_entropy_results[ch_name] = ch_phase_entropy
        print(f"\n===== 通道{ch_name} 功率谱熵计算完成 =====")

    return channel_entropy_results


def batch_calculate_non_seizure_spectral_entropy(channel_non_seizure_eeg_dict, fs):
    """
    批量计算无癫痫发作数据的谱熵
    :param channel_non_seizure_eeg_dict: 无发作数据字典
    :param fs: 采样率
    :return: 无发作谱熵结果
    """
    channel_non_seizure_entropy_results = {}

    print("===== 开始计算无癫痫发作数据功率谱熵 =====")

    for ch_name, non_seizure_list in channel_non_seizure_eeg_dict.items():
        if len(non_seizure_list) == 0:
            continue

        non_seizure_info = non_seizure_list[0]
        eeg_segment = non_seizure_info['eeg_data']
        start_sec = non_seizure_info['start_sec']
        end_sec = non_seizure_info['end_sec']
        duration_sec = non_seizure_info['duration_sec']

        if len(eeg_segment) < fs * 10:
            print(f"警告：通道{ch_name} 数据长度不足10秒，跳过功率谱熵计算")
            continue

        try:
            # 修复：直接调用calculate_power_spectral_entropy
            entropy_info = calculate_power_spectral_entropy(eeg_segment, fs)
        except Exception as e:
            print(f"警告：通道{ch_name} 功率谱熵计算失败: {str(e)}，跳过该通道")
            continue

        ch_non_seizure_entropy = {
            'channel_name': ch_name,
            'is_seizure': False,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': duration_sec,
            'entropy_info': entropy_info
        }

        channel_non_seizure_entropy_results[ch_name] = ch_non_seizure_entropy
        print(f"通道{ch_name}：无发作数据功率谱熵计算完成")

    print(f"\n===== 无发作数据功率谱熵计算全部完成 =====")
    print(f"有效计算通道数量：{len(channel_non_seizure_entropy_results)}/{len(channel_non_seizure_eeg_dict)}")

    return channel_non_seizure_entropy_results


def batch_calculate_phase_band_energy(lfp_data, fs, channel_stats_info):
    """
    批量计算相位频段能量
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param channel_stats_info: 通道统计信息
    :return: 相位频段能量结果
    """
    n_samples, n_channels = lfp_data.shape
    channel_band_results = {}

    channel_valid_events = channel_stats_info['channel_valid_events']

    if len(channel_valid_events) != n_channels:
        raise ValueError("通道有效发作事件列表长度与通道数不匹配！")

    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        ch_local_events = channel_valid_events[ch_idx]

        if len(ch_local_events) == 0:
            continue

        from .detection import divide_seizure_phases
        ch_seizure_phases = divide_seizure_phases(
            channel_stats_info=channel_stats_info,
            fs=fs,
            total_samples=n_samples,
            ch_idx=ch_idx
        )
        if len(ch_seizure_phases) == 0:
            continue

        ch_phase_band = []
        for seizure_phase in ch_seizure_phases:
            seizure_id = seizure_phase['seizure_id']
            current_seizure_band = {'seizure_id': seizure_id, 'phases': []}

            def _process_phase(phase_info, phase_type):
                if phase_info is None or not phase_info['is_valid']:
                    return None
                start_sample = int(phase_info['start_sec'] * fs)
                end_sample = int(phase_info['end_sec'] * fs)

                if start_sample >= n_samples or end_sample <= 0 or start_sample >= end_sample:
                    return None
                if (end_sample - start_sample) < fs * 10:
                    return None

                eeg_segment = lfp_data[start_sample:end_sample, ch_idx]
                band_info = calculate_eeg_band_energy(eeg_segment, fs=fs)

                return {
                    'phase_name': phase_info['phase_name'],
                    'phase_type': phase_type,
                    'start_sec': phase_info['start_sec'],
                    'end_sec': phase_info['end_sec'],
                    'duration_sec': phase_info['duration_sec'],
                    'band_info': band_info
                }

            pre_band = _process_phase(seizure_phase['pre_phase'], '前期')
            if pre_band:
                current_seizure_band['phases'].append(pre_band)

            attack_band = _process_phase(seizure_phase['attack_phase'], '发作期')
            if attack_band:
                current_seizure_band['phases'].append(attack_band)

            post_band = _process_phase(seizure_phase['post_phase'], '后期')
            if post_band:
                current_seizure_band['phases'].append(post_band)

            if len(current_seizure_band['phases']) > 0:
                ch_phase_band.append(current_seizure_band)

        channel_band_results[ch_name] = ch_phase_band
        print(f"\n===== 通道{ch_name} 各频段能量与相对占比计算完成 =====")

    return channel_band_results


def batch_calculate_non_seizure_band_energy(channel_non_seizure_eeg_dict, fs):
    """
    批量计算无癫痫发作数据的频段能量
    :param channel_non_seizure_eeg_dict: 无发作数据字典
    :param fs: 采样率
    :return: 无发作频段能量结果
    """
    channel_non_seizure_band_results = {}

    print("===== 开始计算无癫痫发作数据频段能量与相对占比 =====")

    for ch_name, non_seizure_list in channel_non_seizure_eeg_dict.items():
        if len(non_seizure_list) == 0:
            continue

        non_seizure_info = non_seizure_list[0]
        eeg_segment = non_seizure_info['eeg_data']
        start_sec = non_seizure_info['start_sec']
        end_sec = non_seizure_info['end_sec']
        duration_sec = non_seizure_info['duration_sec']

        if len(eeg_segment) < fs * 10:
            print(f"警告：通道{ch_name} 数据长度不足10秒，跳频段能量计算")
            continue

        try:
            band_info = calculate_eeg_band_energy(eeg_segment, fs=fs)
        except Exception as e:
            print(f"警告：通道{ch_name} 频段能量计算失败: {str(e)}，跳过该通道")
            continue

        ch_non_seizure_band = {
            'channel_name': ch_name,
            'is_seizure': False,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': duration_sec,
            'band_info': band_info
        }

        channel_non_seizure_band_results[ch_name] = ch_non_seizure_band
        print(f"通道{ch_name}：无发作数据频段能量计算完成")

    print(f"\n===== 无发作数据频段能量计算全部完成 =====")
    print(f"有效计算通道数量：{len(channel_non_seizure_band_results)}/{len(channel_non_seizure_eeg_dict)}")

    return channel_non_seizure_band_results


# 混沌指数计算函数
def calculate_r2(y_true, y_pred):
    """计算线性拟合的决定系数R²，评估拟合效果"""
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    return r2

def find_scaling_region(log_r, log_Cr, min_window_size=5, r2_threshold=0.95):
    """
    滑动窗口筛选无标度区（满足线性拟合R²阈值的最长连续区间）
    返回：无标度区的log_r、log_Cr索引
    """
    n_points = len(log_r)
    if n_points < min_window_size:
        return np.array([]), np.array([])

    best_r2 = 0.0
    best_indices = np.array([])

    # 滑动窗口遍历所有可能的区间
    for start in range(n_points - min_window_size + 1):
        for end in range(start + min_window_size - 1, n_points):
            window_log_r = log_r[start:end + 1]
            window_log_Cr = log_Cr[start:end + 1]

            # 线性拟合计算R²
            slope, intercept = np.polyfit(window_log_r, window_log_Cr, 1)
            y_pred = slope * window_log_r + intercept
            r2 = calculate_r2(window_log_Cr, y_pred)

            # 更新最优无标度区
            if r2 >= r2_threshold and (end - start + 1) > len(best_indices):
                best_r2 = r2
                best_indices = np.arange(start, end + 1)
            elif r2 >= best_r2 and r2 >= r2_threshold and (end - start + 1) == len(best_indices):
                best_r2 = r2
                best_indices = np.arange(start, end + 1)

    # 若未找到满足阈值的区间，返回R²最高的最小窗口区间
    if len(best_indices) == 0 and n_points >= min_window_size:
        max_r2 = 0.0
        best_window = np.arange(0, min_window_size)
        for start in range(n_points - min_window_size + 1):
            end = start + min_window_size - 1
            window_log_r = log_r[start:end + 1]
            window_log_Cr = log_Cr[start:end + 1]
            slope, intercept = np.polyfit(window_log_r, window_log_Cr, 1)
            y_pred = slope * window_log_r + intercept
            r2 = calculate_r2(window_log_Cr, y_pred)
            if r2 > max_r2:
                max_r2 = r2
                best_window = np.arange(start, end + 1)
        best_indices = best_window

    return best_indices

def estimate_tau(signal, max_tau=50, bins=20):
    """
    使用平均互信息(AMMI)的第一极小值估计最佳延迟 tau
    优化：增加噪声鲁棒性，返回合理范围内的tau
    """
    N = len(signal)
    if N < 500:
        return 1  # 数据太短直接返回1，避免计算报错

    taus = np.arange(1, max_tau + 1)
    mis = []

    # 为了速度和鲁棒性，截取中间部分数据（避免首尾噪声）
    limit_N = min(N, 5000)
    start_idx = max(0, (N - limit_N) // 2)
    sig_part = signal[start_idx:start_idx + limit_N]

    # 预先计算直方图箱子，保证离散化一致性
    bins_edges = np.histogram_bin_edges(sig_part, bins=bins)

    for tau in taus:
        x = sig_part[:-tau]
        y = sig_part[tau:]

        # 离散化（基于预计算的箱子边缘）
        x_disc = np.digitize(x, bins_edges)
        y_disc = np.digitize(y, bins_edges)

        # 计算互信息（增加异常值处理）
        try:
            mi = mutual_info_score(x_disc, y_disc)
        except:
            mi = np.inf
        mis.append(mi)

    mis = np.array(mis)
    # 去除无穷大异常值
    mis = np.where(np.isinf(mis), np.nanmax(mis), mis)

    # 找第一个局部极小值（更符合Fraser & Swinney理论）
    local_minima = argrelextrema(mis, np.less)[0]
    if len(local_minima) > 0:
        tau_opt = taus[local_minima[0]]
    else:
        tau_opt = taus[np.nanargmin(mis)]  # 如果没局部极小，取全局最小作为备选

    # 确保tau在合理范围（1~max_tau的1/2，避免过大）
    tau_opt = np.clip(tau_opt, 1, max_tau // 2)
    return int(tau_opt)

def robust_correlation_dimension(data, emb_dim, tau, r_vals=None):
    """
    基于 KDTree 的鲁棒关联维数计算（O(N log N) 提速，抗噪性更强）
    优化：修复关联积分归一化、动态生成r范围、筛选无标度区、增加拟合优度验证
    """
    N = len(data)
    # 1. 相空间重构：创建嵌入矩阵 (shape: (n_vectors, emb_dim))
    n_vectors = N - (emb_dim - 1) * tau
    if n_vectors < 100:  # 嵌入向量数量过少，无法有效计算
        return np.nan, ([], []), 0.0

    # 高效构建嵌入向量（向量化操作，避免循环提速）
    indices = np.arange(n_vectors)[:, None] + np.arange(emb_dim)[None, :] * tau
    embedded_data = data[indices].astype(np.float64)  # 强制转换为float64，避免精度问题

    # 2. 使用 KDTree 加速距离查询
    try:
        tree = KDTree(embedded_data, leaf_size=40)
    except:
        return np.nan, ([], []), 0.0

    # 3. 随机选择参考点（平衡速度与精度，最多 1000 个参考点）
    n_ref = min(n_vectors, 1000)
    rng = np.random.RandomState(42)  # 固定随机种子，保证结果可复现
    ref_indices = rng.choice(n_vectors, n_ref, replace=False)
    ref_points = embedded_data[ref_indices]

    # 4. 动态生成半径 r 的范围（优化：基于嵌入向量的距离分布，避免硬编码）
    if r_vals is None:
        # 计算部分两两距离，确定合理的r范围
        sample_size = min(500, n_vectors)
        sample_indices = rng.choice(n_vectors, sample_size, replace=False)
        sample_vectors = embedded_data[sample_indices]

        # 计算样本内的距离分位数
        dists = []
        for vec in sample_vectors[:100]:  # 仅计算前100个样本，提升速度
            dist, _ = tree.query(vec.reshape(1, -1), k=min(50, n_vectors))
            dists.extend(dist.flatten().tolist())

        if len(dists) > 0:
            dists = np.array(dists)
            r_min = max(0.01, np.percentile(dists, 5))  # 5%分位数
            r_max = min(10.0, np.percentile(dists, 95))  # 95%分位数
        else:
            r_min, r_max = 0.1, 2.0

        # 确保r_min < r_max，避免生成空数组
        if r_min >= r_max:
            r_min, r_max = 0.1, 2.0

        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), 15)
    else:
        r_vals = np.array(r_vals, dtype=np.float64)

    # 5. 计算关联积分 C(r)（统计每个半径内的邻居数量）
    counts = []
    for r in r_vals:
        # 每个半径单独调用 query_radius，r 为标量，格式合法
        try:
            cnt = tree.query_radius(ref_points, r=r, count_only=True)
        except:
            cnt = np.zeros(n_ref)
        counts.append(cnt)

    # 转换为数组，形状 (n_radii, n_ref)，再转置为 (n_ref, n_radii) 匹配原有逻辑
    counts = np.array(counts).T

    # 6. 关联积分归一化（修复：分母改为n_vectors，符合G-P算法理论）
    # 排除自身点（减1），归一化处理
    denominator = n_vectors  # 修复核心：将 n_vectors - 1 改为 n_vectors
    correlation_sum = np.mean((counts - 1) / denominator, axis=0)

    # 7. 筛选有效数据，移除C(r)=0或C(r)=1的点（避免饱和）
    valid_idx = (correlation_sum > 0) & (correlation_sum < 1)
    if np.sum(valid_idx) < 5:  # 有效数据点过少，无法拟合（提升至5个，保证无标度区筛选）
        return np.nan, ([], []), 0.0

    log_r = np.log10(r_vals[valid_idx])
    log_Cr = np.log10(correlation_sum[valid_idx])

    # 8. 筛选无标度区（优化：仅在无标度区内拟合，保证D2的有效性）
    scaling_indices = find_scaling_region(log_r, log_Cr, min_window_size=5, r2_threshold=0.95)
    if len(scaling_indices) < 5:
        return np.nan, ([], []), 0.0

    scaling_log_r = log_r[scaling_indices]
    scaling_log_Cr = log_Cr[scaling_indices]

    # 9. 线性拟合：计算D2（关联维数）和拟合优度R²
    try:
        slope, intercept = np.polyfit(scaling_log_r, scaling_log_Cr, 1)
        y_pred = slope * scaling_log_r + intercept
        r2 = calculate_r2(scaling_log_Cr, y_pred)
    except:
        return np.nan, ([], []), 0.0

    # 10. 验证D2的合理性（理论：D2 <= 嵌入维数emb_dim，且>=0）
    if slope < 0 or slope > emb_dim:
        return np.nan, ([], []), r2

    return slope, (scaling_log_r, scaling_log_Cr), r2

def compute_d2(signal, fs=1000, segment_duration=60, m_candidates=[4, 6, 8]):
    """
    分段计算癫痫脑电信号的关联维数 D2（整合改进后的鲁棒算法）
    优化：嵌入维数饱和性验证、分段独立估计tau、合理过滤异常值
    参数：
        signal: 1D 脑电信号数组
        fs: 采样率
        segment_duration: 分段时长(秒)
        m_candidates: 嵌入维数候选列表，用于饱和性验证
    返回：
        mean_d2: 有效段的平均 D2 值
        d2_list: 各分段的 D2 值列表
    """
    # 从外部环境获取全局变量
    global save_dir, DATE_STR

    # 1. 数据预处理：去趋势 + 标准化（提高计算鲁棒性）
    signal = signal - np.mean(signal)  # 简单去均值去趋势
    signal = signal / (np.std(signal) + 1e-8)  # 额外归一化，避免标准化后方差为0
    scaler = StandardScaler()
    try:
        signal_norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    except:
        signal_norm = signal.flatten()

    # 2. 降采样（至 200Hz，减少计算量，保留癫痫核心特征）
    target_fs = 200
    if fs > target_fs:
        try:
            num_samples = int(len(signal_norm) * target_fs / fs)
            signal_proc = resample(signal_norm, num_samples)
        except:
            signal_proc = signal_norm
        print(f"数据降采样: {fs}Hz -> {target_fs}Hz (点数: {len(signal_proc)})")
    else:
        signal_proc = signal_norm
        target_fs = fs

    # 3. 分段计算 D2（优化：每个分段独立估计tau，嵌入维数饱和性验证）
    seg_len = int(segment_duration * target_fs)
    n_segs = len(signal_proc) // seg_len
    if seg_len < 100:  # 分段过短，无法计算
        print("错误：分段时长过短，无法进行有效计算")
        return 0.0, []

    d2_list = []
    sample_curve = None
    sample_r2 = 0.0

    print(f"\n===== 开始计算关联维数 D2 (总段数: {n_segs}, 嵌入维数候选: {m_candidates}) =====")

    for i in range(n_segs):
        start = i * seg_len
        end = start + seg_len
        segment = signal_proc[start:end]

        # 跳过噪声过大的分段（标准差接近0）
        if np.std(segment) < 1e-6:
            print(f"段 {i + 1}: 信号噪声过大，标准差接近0 -> 标记为 NaN")
            d2_list.append(np.nan)
            continue

        try:
            # 步骤1：分段独立估计最佳tau（优化：避免全局tau的局限性）
            tau = estimate_tau(segment)

            # 步骤2：嵌入维数饱和性验证（选择稳定的D2值）
            d2_candidates = []
            r2_candidates = []
            for m in m_candidates:
                d2, curve, r2 = robust_correlation_dimension(segment, emb_dim=m, tau=tau)
                if not np.isnan(d2) and r2 >= 0.8:  # 仅保留拟合优度较好的结果
                    d2_candidates.append(d2)
                    r2_candidates.append(r2)

            # 步骤3：确定最终D2值（取候选值的均值，若稳定）
            if len(d2_candidates) >= 2:
                # 计算候选值的变异系数，判断是否稳定
                cv = np.std(d2_candidates) / np.mean(d2_candidates)
                if cv < 0.1:  # 变异系数<10%，认为稳定
                    final_d2 = np.mean(d2_candidates)
                    final_r2 = np.mean(r2_candidates)
                    # 提取示例曲线（取中间嵌入维数的结果）
                    m_mid = m_candidates[len(m_candidates) // 2]
                    _, final_curve, _ = robust_correlation_dimension(segment, emb_dim=m_mid, tau=tau)
                else:
                    final_d2 = d2_candidates[np.argmax(r2_candidates)]
                    final_r2 = np.max(r2_candidates)
                    final_curve = ([], [])
            elif len(d2_candidates) == 1:
                final_d2 = d2_candidates[0]
                final_r2 = r2_candidates[0]
                m_mid = m_candidates[0]
                _, final_curve, _ = robust_correlation_dimension(segment, emb_dim=m_mid, tau=tau)
            else:
                final_d2 = np.nan
                final_r2 = 0.0
                final_curve = ([], [])

            # 步骤4：异常值过滤（优化：严格遵循理论，D2<=嵌入维数，且>=0）
            max_m = max(m_candidates)
            if np.isnan(final_d2) or final_d2 < 0 or final_d2 > max_m or final_r2 < 0.8:
                print(f"段 {i + 1}: 计算异常 (D2={final_d2:.2f}, R²={final_r2:.2f}) -> 标记为 NaN")
                d2_list.append(np.nan)
            else:
                d2_list.append(final_d2)
                print(f"段 {i + 1}: D2 = {final_d2:.3f}, R² = {final_r2:.2f} (tau={tau})")
                # 保存示例曲线（非NaN的有效曲线）
                if sample_curve is None and len(final_curve[0]) > 0:
                    sample_curve = final_curve
                sample_r2 = max(sample_r2, final_r2)

        except Exception as e:
            print(f"段 {i + 1} 报错: {str(e)} -> 标记为 NaN")
            d2_list.append(np.nan)

    # 4. 结果统计与提示
    d2_valid = [x for x in d2_list if not np.isnan(x)]
    mean_d2 = np.mean(d2_valid) if d2_valid else 0.0

    print(f"\n计算完成。有效段数: {len(d2_valid)}/{n_segs}")
    print(f"平均 D2: {mean_d2:.3f}")
    if len(d2_valid) > 0:
        if mean_d2 < 4:
            print(">> 提示: D2值较低，符合癫痫发作期特征 (高同步性/低复杂度)")
        else:
            print(">> 提示: D2值正常，符合非发作期脑电特征 (低同步性/高复杂度)")
    else:
        print(">> 提示: 无有效D2结果，无法进行临床特征判断")

    # ========== 以下为注释掉的可视化相关代码 ==========
    # # 5. 结果可视化（带中文支持，双图布局，优化无有效数据的展示）
    # plt.figure(figsize=(14, 6))

    # # 左图：D2 随时间变化趋势
    # plt.subplot(1, 2, 1)
    # t_axis = np.arange(n_segs) * segment_duration
    # plt.plot(t_axis, d2_list, 'o-', linewidth=2, color='#2c3e50', markersize=4)
    # plt.axhline(y=3.5, color='red', linestyle='--', alpha=0.7, label='发作期/非发作期阈值')
    # plt.title(f'关联维数 D2 随时间变化 (嵌入维数: {m_candidates})', fontsize=12)
    # plt.xlabel('时间 (秒)', fontsize=10)
    # plt.ylabel('关联维数 (D2)', fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(loc='best')

    # # 右图：双对数坐标关联积分拟合示例（无标度区）
    # plt.subplot(1, 2, 2)
    # if sample_curve and len(sample_curve[0]) > 0:
    #     lx, ly = sample_curve
    #     plt.scatter(lx, ly, s=20, c='blue', alpha=0.6, label='无标度区数据')
    #     # 绘制拟合直线
    #     poly = np.polyfit(lx, ly, 1)
    #     y_fit = np.polyval(poly, lx)
    #     plt.plot(lx, y_fit, 'r--', lw=2, label=f'拟合斜率 (D2)={poly[0]:.2f}\nR²={sample_r2:.2f}')
    #     plt.title('双对数坐标关联积分 (无标度区拟合)', fontsize=12)
    #     plt.xlabel('log10(半径 r)', fontsize=10)
    #     plt.ylabel('log10(关联积分 C(r))', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.grid(True, linestyle='--', alpha=0.6)
    # else:
    #     plt.text(0.5, 0.5, "无有效拟合数据\n（请检查信号质量或分段时长）", ha='center', va='center', fontsize=11)

    # plt.tight_layout()
    # # 保存图表（避免路径不存在报错，增加异常处理）
    # try:
    #     os.makedirs(save_dir, exist_ok=True)
    #     plot_path = os.path.join(save_dir, f'd2_analysis_optimized_{DATE_STR}.png')
    #     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #     print(f"\n图表已保存至: {plot_path}")
    # except Exception as e:
    #     print(f"\n图表保存失败: {str(e)}")
    # plt.show()
    # ========== 可视化代码注释结束 ==========

    # # 6. 保存结果到 Excel（增加异常处理，避免路径错误）
    # try:
    #     df_res = pd.DataFrame({
    #         'Segment_ID': range(1, n_segs + 1),
    #         'Start_Time_s': np.arange(n_segs) * segment_duration,
    #         'D2_Value': d2_list
    #     })
    #     excel_path = os.path.join(save_dir, f'd2_results_optimized_{DATE_STR}.xlsx')
    #     df_res.to_excel(excel_path, index=False)
    #     print(f"结果已保存至 Excel: {excel_path}")
    # except Exception as e:
    #     print(f"Excel 保存失败: {str(e)}")

    return mean_d2, d2_list

def calculate_delay(signal, max_delay=100):
    """自相关法确定延迟时间τ"""
    autocorr = correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # 取正延迟部分

    # 找第一个自相关值下降到最大值1/e的位置
    epsilon = np.exp(-1)
    for tau in range(1, max_delay):
        if autocorr[tau] < autocorr[0] * epsilon:
            return tau
    return 10  # 默认值（1000Hz下对应10ms）

def embed_signal(signal, m, tau):
    """相空间重构（时间延迟嵌入）"""
    n = len(signal)
    embed_len = n - (m - 1) * tau
    if embed_len <= 0:
        raise ValueError("嵌入维度或延迟过大，需减小m或tau")
    embed_mat = np.zeros((embed_len, m))
    for i in range(embed_len):
        embed_mat[i, :] = signal[i:i + m * tau:tau]
    return embed_mat

def calculate_local_cor_dim(embed_mat, r_range=None):
    """计算单个窗口的点关联维数PD2"""
    n = embed_mat.shape[0]
    if n < 20:  # 窗口点数过少，返回NaN避免无效拟合
        return np.nan

    # 计算所有点对的欧氏距离（向量化计算，提升效率）
    dists = np.sqrt(np.sum((embed_mat[:, None, :] - embed_mat[None, :, :]) ** 2, axis=-1))
    dists = dists[np.triu_indices(n, k=1)]  # 取上三角，避免重复计算

    # 确定半径范围（文献常用：数据标准差的0.1~2倍，对数分布）
    if r_range is None:
        sigma = dists.std()
        if sigma < 1e-8:  # 避免标准差为0导致计算错误
            return np.nan
        r_range = np.logspace(np.log10(0.1 * sigma), np.log10(2 * sigma), 20)

    # 计算关联积分C(r)
    C = np.zeros_like(r_range)
    total_pairs = len(dists)  # 总点对数（n*(n-1)/2）
    for i, r in enumerate(r_range):
        C[i] = np.sum(dists < r) / total_pairs  # 关联积分归一化

    # 筛选有效数据（排除C(r)=0或1的饱和情况）
    valid_idx = (C > 1e-5) & (C < 0.99)
    if not np.any(valid_idx):
        return np.nan
    r_valid = r_range[valid_idx]
    C_valid = C[valid_idx]

    # 双对数线性拟合求斜率（即PD2值）
    try:
        slope, _ = np.polyfit(np.log10(r_valid), np.log10(C_valid), 1)
    except:
        slope = np.nan
    return slope

def compute_pd2_eeg(channel_data, fs, win_len=1, m_range=(3, 5), max_delay=100):
    """计算脑电信号的点关联维数PD2序列（核心主函数）"""
    win_size = int(win_len * fs)
    n_win = len(channel_data) // win_size
    if n_win == 0:
        raise ValueError("窗口长度过大或数据过短，无法分窗")

    pd2_seq = np.zeros((n_win, len(m_range)))
    print(f"\n开始计算PD2序列：共 {n_win} 个窗口，窗口长度 {win_len} s，嵌入维度 {m_range}")

    tau = calculate_delay(channel_data[:int(10 * fs)], max_delay=max_delay)
    print(f"确定延迟时间 τ = {tau}（对应 {tau / fs * 1000:.1f} ms）")

    # 无进度条
    for win_idx in range(n_win):
        win_start = win_idx * win_size
        win_end = win_start + win_size
        win_data = channel_data[win_start:win_end]
        win_data = win_data - win_data.mean()

        for m_idx, m in enumerate(m_range):
            try:
                embed_mat = embed_signal(win_data, m, tau)
                pd2_seq[win_idx, m_idx] = calculate_local_cor_dim(embed_mat)
            except:
                pd2_seq[win_idx, m_idx] = np.nan

    pd2_seq = np.nanmean(pd2_seq, axis=1)
    print("PD2序列计算完成")
    return pd2_seq

def estimate_tau_ami(data, max_tau=50, bins=32):
    """ 基于平均互信息估计 Tau """
    n = len(data)
    if n > 5000:
        data = data[:5000]

    mis = []
    bins_edges = np.histogram_bin_edges(data, bins=bins)

    for tau in range(1, max_tau + 1):
        p1 = data[:-tau]
        p2 = data[tau:]
        p1_disc = np.digitize(p1, bins_edges)
        p2_disc = np.digitize(p2, bins_edges)
        mi = mutual_info_score(p1_disc, p2_disc)
        mis.append(mi)

    mis = np.array(mis)
    local_minima = argrelextrema(mis, np.less)[0]
    if len(local_minima) > 0:
        tau_opt = local_minima[0] + 1
    else:
        tau_opt = np.argmin(mis) + 1

    print(f"  -> [参数估计] 最佳时间延迟 (Tau): {tau_opt}")
    return int(tau_opt)

def compute_le_rosenstein(signal_data, fs, window_size_sec=10, emb_dim=6):
    print(f"\n===== 开始计算最大李雅普诺夫指数 (LLE) =====")

    if signal_data.ndim == 2:
        signal_single = signal_data[:, 0]
    else:
        signal_single = signal_data

    target_fs = 200
    if fs > target_fs:
        n_new = int(len(signal_single) * target_fs / fs)
        signal_proc = resample(signal_single, n_new)
        print(f"数据降采样: {fs}Hz -> {target_fs}Hz")
    else:
        signal_proc = signal_single
        target_fs = fs

    scaler = StandardScaler()
    signal_proc = scaler.fit_transform(signal_proc.reshape(-1, 1)).flatten()
    tau = estimate_tau_ami(signal_proc)

    points_per_window = int(window_size_sec * target_fs)
    num_windows = len(signal_proc) // points_per_window
    lle_values = []
    time_points = []

    print(f"计算配置: 窗口={window_size_sec}s, m={emb_dim}, Tau={tau}")

    for i in range(num_windows):
        idx_start = i * points_per_window
        idx_end = idx_start + points_per_window
        segment = signal_proc[idx_start:idx_end]

        try:
            le = nolds.lyap_r(segment, emb_dim=emb_dim, lag=tau, min_tsep=tau, trajectory_len=30, fit='poly')
            lle_values.append(le)
        except:
            lle_values.append(np.nan)
        time_points.append(idx_start / target_fs)

    print("LE计算完成!")
    return np.array(time_points), np.array(lle_values)

def _discretize_signal(signal, num_bins):
    """
    内部辅助函数：将连续信号基于分位数离散化为符号序列（等频分箱，适配EEG数据）
    :param signal: 输入连续信号（np.ndarray）
    :param num_bins: 分箱数（符号表大小）
    :return: 离散化后的符号序列（np.ndarray，非负整数）
    """
    if num_bins <= 1:
        raise ValueError("分箱数num_bins必须大于1")

    # 计算分位数阈值
    quantiles = np.quantile(signal, np.linspace(0, 1, num_bins + 1))
    # 去重阈值（避免数据分布集中导致阈值重复）
    quantiles = np.unique(quantiles)
    if len(quantiles) < 2:
        raise RuntimeError("信号数据分布过于集中，无法进行有效分箱")

    # 信号离散化
    symbols = np.digitize(signal, quantiles, right=False) - 1
    # 确保符号在[0, num_bins-1]范围内
    symbols = np.clip(symbols, 0, num_bins - 1)

    return symbols.astype(np.int32)


def _check_convergence(diff_entropies, threshold=1e-4, consecutive=3):
    """
    内部辅助函数：判断ΔH(k)的收敛性（是否进入平台期）
    :param diff_entropies: ΔH(k)差值列表
    :param threshold: 相邻值波动阈值
    :param consecutive: 连续稳定的点数要求
    :return: 收敛标记（bool）、收敛起始索引（int）
    """
    if len(diff_entropies) < consecutive:
        return False, 0

    # 计算相邻ΔH的绝对差值
    diff_fluctuation = np.abs(np.diff(diff_entropies))

    # 查找连续consecutive个波动小于阈值的区间
    for i in range(len(diff_fluctuation) - consecutive + 1):
        if all(fluct < threshold for fluct in diff_fluctuation[i:i + consecutive]):
            return True, i + 1  # 返回收敛起始索引（对应diff_entropies的索引）

    return False, 0


def compute_kolmogorov_entropy(channel_data,fs,save_dir,num_bins=4,max_k=8,plot_fig=True):
    """
    封装优化版Kolmogorov Entropy (KE/KS Entropy) 计算流程，针对EEG/LFP脑电数据
    修复原代码缺陷：完善数据校验、优化KE估计方法、修正逻辑漏洞、提升鲁棒性
    :param channel_data: 必选入参 - 单通道脑电数据（np.ndarray，一维浮点型）
    :param fs: 必选入参 - 采样率（Hz），如256、512
    :param num_bins: 可选入参 - 符号分箱数，默认4（文献推荐3-5，需为整数且≥3）
    :param max_k: 可选入参 - 最大块长，默认8（平衡精度与计算量，4^8=65536状态可行）
    :param save_dir: 可选入参 - 结果保存目录，默认创建当前目录下ke_results文件夹
    :return: 结构化计算结果字典（包含KE估计值、块熵数据等）
    """
    # 初始化返回结果
    result_dict = {
        "ke_nats": np.nan,
        "ke_bits": np.nan,
        "ke_rate_per_sec": np.nan,
        "block_entropies": {},
        "per_symbol_entropies": {},
        "diff_entropies": [],
        "status": "failed",
        "message": ""
    }

    try:
        # ===================== 步骤1：全链路数据有效性校验 =====================
        # 1. 校验channel_data类型与结构
        if not isinstance(channel_data, np.ndarray):
            raise TypeError("channel_data必须是numpy.ndarray类型")
        if channel_data.ndim != 1:
            # 扁平化二维数组（兼容偶发的二维输入）
            channel_data = channel_data.flatten()
            print("警告：输入数据为二维数组，已自动扁平化为一维")
        if len(channel_data) == 0:
            raise ValueError("channel_data不能为空数组")
        if not np.issubdtype(channel_data.dtype, np.floating):
            raise TypeError("channel_data必须是浮点型数组（建议np.float64）")
        if not np.isfinite(channel_data).all():
            raise ValueError("channel_data包含无穷大或NaN值，无法进行有效计算")

        # 2. 校验采样率fs
        if not (isinstance(fs, (int, float)) and fs > 0):
            raise ValueError("采样率fs必须是大于0的数字（如256、512）")
        if fs < 100 or fs > 2000:
            print("警告：采样率fs超出100-2000Hz的常见EEG范围，结果可能存在偏差")

        # 3. 校验可选入参合法性
        if not (isinstance(num_bins, int) and num_bins >= 3):
            raise ValueError("num_bins必须是大于等于3的整数（文献推荐3-5）")
        if not (isinstance(max_k, int) and max_k >= 1):
            raise ValueError("max_k必须是大于等于1的整数")

        # 提前创建保存目录（兼容原有逻辑）
        os.makedirs(save_dir, exist_ok=True)
        DATE_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # ===================== 步骤2：信号符号化（带中间结果校验） =====================
        print("\n===== 开始计算 Kolmogorov Entropy (KE) =====")
        print(f"信号长度: {len(channel_data)} 点 ({len(channel_data) / fs:.1f}s @ {fs:.0f}Hz)")

        # 信号离散化
        symbols = _discretize_signal(channel_data, num_bins)

        # 校验符号序列有效性
        if len(symbols) == 0:
            raise RuntimeError("离散化失败，符号序列为空")
        if not np.issubdtype(symbols.dtype, np.integer):
            raise RuntimeError("符号序列必须为整数类型")
        if not (0 <= symbols.min() and symbols.max() < num_bins):
            raise RuntimeError(f"符号序列超出[0, {num_bins - 1}]的合法范围")

        print(f"符号化完成: {num_bins} 符号 (0-{num_bins - 1}), 符号分布: {np.bincount(symbols)}")

        # ===================== 步骤3：块熵计算（异常处理） =====================
        block_entropies = {}
        per_symbol_entropies = {}
        diff_entropies = []
        bits_to_nats = np.log(2)  # 单位转换因子：bits → nats

        print("\n计算块熵 H(k) for k=1 to", max_k, "(base=e, natural log)...")

        for k in range(1, max_k + 1):
            try:
                # 计算块熵（pyinform默认返回以2为底的bits值）
                H_k_bits = pi.block_entropy(symbols, k=k)

                # 手动转换为nats单位（匹配物理逻辑）
                H_k = H_k_bits * bits_to_nats

                # 存储结果
                block_entropies[k] = H_k
                per_symbol_entropies[k] = H_k / k

                # 计算ΔH(k) = H(k) - H(k-1)
                if k > 1:
                    h_k = block_entropies[k] - block_entropies[k - 1]
                    diff_entropies.append(h_k)
                    print(f"k={k}: H({k})={H_k:.4f} nats, H(k)/k={per_symbol_entropies[k]:.4f}, ΔH={h_k:.4f}")
                else:
                    print(f"k={k}: H({k})={H_k:.4f} nats, H(k)/k={per_symbol_entropies[k]:.4f}")

            except ValueError as ve:
                print(f"警告: k={k} 计算失败（数据量不足/格式错误）: {ve}，停止计算更高阶熵")
                break
            except RuntimeError as re:
                print(f"警告: k={k} 计算失败（PyInform接口错误）: {re}，停止计算更高阶熵")
                break
            except Exception as e:
                print(f"警告: k={k} 计算失败（未知错误）: {e}，停止计算更高阶熵")
                break

        # ===================== 步骤4：KE估计（优化收敛判断与鲁棒估计） =====================
        ke_estimate = np.nan
        ke_bits = np.nan
        ke_rate_per_sec = np.nan

        if len(diff_entropies) > 0:
            # 收敛性判断（核心优化：动态判断平台期，替代固定截取）
            is_converged, conv_start_idx = _check_convergence(diff_entropies)

            if is_converged:
                # 取平台期数据的中位数（抗异常值干扰，优于均值）
                plateau_data = diff_entropies[conv_start_idx:]
                ke_estimate = np.median(plateau_data)
                print(f"\n检测到ΔH(k)收敛（平台期起始k={conv_start_idx + 2}），采用平台期中位数作为KE估计")
            else:
                # 未收敛时，采用k>3的滑动平均（备选方案，保留兼容性）
                if len(diff_entropies) >= 4:
                    ke_estimate = np.mean(diff_entropies[3:])
                else:
                    ke_estimate = np.mean(diff_entropies)
                print(f"\n未检测到ΔH(k)收敛，采用均值作为KE估计（仅供参考）")

            # 单位转换与熵率计算
            ke_bits = ke_estimate / np.log(2)
            symbol_rate = fs  # 1:1映射（注释：若后续采用降采样，需修改为实际符号生成率）
            ke_rate_per_sec = ke_bits * symbol_rate

            # 打印结果汇总
            print(f"\n===== KE 估计结果 =====")
            print(f"符号字母表大小 D={num_bins}")
            print(f"KS Entropy (nats/symbol): {ke_estimate:.4f}")
            print(f"KS Entropy (bits/symbol): {ke_bits:.4f}")
            print(f"解释: 值~1-3 bits 表示中等复杂度 (正常EEG); <1 可能表示癫痫同步。")
            print(f"熵率 (bits/s): {ke_rate_per_sec:.2f} (全信号信息产生率，基于1:1符号-采样点映射)")

        else:
            raise RuntimeError("无法计算有效块熵，检查数据长度或减小max_k")

        # ===================== 步骤5：更新返回结果字典 =====================
        result_dict.update({
            "ke_nats": ke_estimate,
            "ke_bits": ke_bits,
            "ke_rate_per_sec": ke_rate_per_sec,
            "block_entropies": block_entropies,
            "per_symbol_entropies": per_symbol_entropies,
            "diff_entropies": diff_entropies,
            "status": "success",
            "message": "KE计算完成"
        })

    except Exception as e:
        # 捕获全局异常，返回错误信息
        error_msg = f"KE计算失败: {str(e)}"
        print(f"\n错误: {error_msg}")
        result_dict["message"] = error_msg

    return result_dict


def batch_calculate_seizure_features(channel_seizure_eeg_dict, fs, save_dir):
    """
    批量遍历所有通道的发作期脑电数据，计算D2、PD2、LE、KE四个特征值
    按发作次数（0/1/2次）存储结果，最终汇总所有有效特征的平均值
    无额外图表输出，仅保留核心数据计算与Excel存储

    :param channel_seizure_eeg_dict: 通道发作期脑电数据字典（来自 batch_extract_seizure_eeg_data）
    :param fs: 采样率（Hz），如256、512
    :param save_dir: 结果保存目录（已定义的 HTML_OUTPUT_DIR）
    :return: 结构化结果字典（通道详细结果 + 全局平均值）
    """
    # 1. 初始化结果存储结构
    # 1.1 通道详细结果（按通道→发作次数→四个特征）
    channel_detail_results = {}
    # 1.2 全局特征列表（用于计算平均值）
    global_d2_list = []
    global_pd2_mean_list = []
    global_le_mean_list = []
    global_ke_bits_list = []

    # 2. 遍历所有通道，逐个处理发作期数据
    print("===== 开始批量计算所有通道发作期特征（D2/PD2/LE/KE） =====")
    for ch_name, seizure_list in tqdm(channel_seizure_eeg_dict.items(), desc="通道处理进度"):
        ch_seizure_results = []

        # 2.1 无有效发作期的通道
        if len(seizure_list) == 0:
            channel_detail_results[ch_name] = {
                "seizure_count": 0,
                "seizure_details": [],
                "valid_features": False
            }
            continue

        # 2.2 有有效发作期的通道，遍历每个发作期
        for seizure_idx, seizure_info in enumerate(seizure_list, 1):
            seizure_eeg = seizure_info['eeg_data']
            seizure_start = seizure_info['start_sec']
            seizure_end = seizure_info['end_sec']

            # 初始化单个发作期的特征结果
            single_seizure_features = {
                "seizure_index": seizure_idx,
                "start_sec": seizure_start,
                "end_sec": seizure_end,
                "duration_sec": seizure_end - seizure_start,
                "D2": np.nan,
                "D2_list": [],
                "PD2_mean": np.nan,
                "PD2_sequence": [],
                "LE_mean": np.nan,
                "LE_sequence": [],
                "KE_bits": np.nan,
                "KE_result": None
            }

            try:
                # ===================== 计算 D2 关联维数 =====================
                mean_d2, d2_list = compute_d2(
                    signal=seizure_eeg,
                    fs=fs,
                    segment_duration=60,  # 发作期已为1分钟，整段计算
                    m_candidates=[4, 6, 8]
                )
                single_seizure_features["D2"] = mean_d2
                single_seizure_features["D2_list"] = d2_list
                if not np.isnan(mean_d2) and mean_d2 > 0:
                    global_d2_list.append(mean_d2)

                # ===================== 计算 PD2 点关联维数 =====================
                pd2_seq = compute_pd2_eeg(
                    channel_data=seizure_eeg,
                    fs=fs,
                    win_len=1,
                    m_range=(3, 5),
                    max_delay=100
                )
                pd2_mean = np.nanmean(pd2_seq) if len(pd2_seq) > 0 else np.nan
                single_seizure_features["PD2_mean"] = pd2_mean
                single_seizure_features["PD2_sequence"] = pd2_seq
                if not np.isnan(pd2_mean):
                    global_pd2_mean_list.append(pd2_mean)

                # ===================== 计算 LE 李雅普诺夫指数 =====================
                t_axis, lle_series = compute_le_rosenstein(
                    signal_data=seizure_eeg,
                    fs=fs,
                    window_size_sec=10,
                    emb_dim=6
                )
                le_mean = np.nanmean(lle_series) if len(lle_series) > 0 else np.nan
                single_seizure_features["LE_mean"] = le_mean
                single_seizure_features["LE_sequence"] = lle_series
                if not np.isnan(le_mean):
                    global_le_mean_list.append(le_mean)

                # ===================== 计算 KE 科尔莫戈罗夫熵 =====================
                ke_result = compute_kolmogorov_entropy(
                    channel_data=seizure_eeg,
                    fs=fs,
                    save_dir=os.path.join(save_dir, f"KE_{ch_name}_seizure{seizure_idx}"),
                    num_bins=4,
                    max_k=8,
                    plot_fig=False  # 关闭可视化图表，满足需求
                )
                ke_bits = ke_result["ke_bits"] if ke_result["status"] == "success" else np.nan
                single_seizure_features["KE_bits"] = ke_bits
                single_seizure_features["KE_result"] = ke_result
                if not np.isnan(ke_bits):
                    global_ke_bits_list.append(ke_bits)

            except Exception as e:
                print(f"\n警告：{ch_name} 发作{seizure_idx} 特征计算失败: {str(e)}")
                continue

            # 添加单个发作期结果到通道列表
            ch_seizure_results.append(single_seizure_features)

        # 2.3 保存当前通道的所有发作期结果
        channel_detail_results[ch_name] = {
            "seizure_count": len(ch_seizure_results),
            "seizure_details": ch_seizure_results,
            "valid_features": len([s for s in ch_seizure_results if
                                   not all(np.isnan([s['D2'], s['PD2_mean'], s['LE_mean'], s['KE_bits']]))]) > 0
        }

    # 3. 计算全局特征平均值
    global_feature_summary = {
        "D2": {
            "valid_count": len(global_d2_list),
            "mean_value": np.mean(global_d2_list) if len(global_d2_list) > 0 else np.nan
        },
        "PD2": {
            "valid_count": len(global_pd2_mean_list),
            "mean_value": np.mean(global_pd2_mean_list) if len(global_pd2_mean_list) > 0 else np.nan
        },
        "LE": {
            "valid_count": len(global_le_mean_list),
            "mean_value": np.mean(global_le_mean_list) if len(global_le_mean_list) > 0 else np.nan
        },
        "KE": {
            "valid_count": len(global_ke_bits_list),
            "mean_value": np.mean(global_ke_bits_list) if len(global_ke_bits_list) > 0 else np.nan
        }
    }

    # 4. 结果保存到Excel（单文件，包含通道详情与全局汇总）
    try:
        # 4.1 创建Excel工作簿
        with pd.ExcelWriter(os.path.join(save_dir, "Seizure_Features_Summary.xlsx"), engine="openpyxl") as writer:
            # 4.2 工作表1：通道发作期特征详情
            detail_data = []
            for ch_name, ch_info in channel_detail_results.items():
                for seizure_detail in ch_info["seizure_details"]:
                    detail_data.append({
                        "通道名称": ch_name,
                        "发作编号": seizure_detail["seizure_index"],
                        "发作起始时间(s)": seizure_detail["start_sec"],
                        "发作结束时间(s)": seizure_detail["end_sec"],
                        "发作时长(s)": seizure_detail["duration_sec"],
                        "D2关联维数": seizure_detail["D2"],
                        "PD2平均点关联维数": seizure_detail["PD2_mean"],
                        "LE平均李雅普诺夫指数": seizure_detail["LE_mean"],
                        "KE熵(bits/symbol)": seizure_detail["KE_bits"]
                    })
            df_detail = pd.DataFrame(detail_data)
            df_detail.to_excel(writer, sheet_name="通道发作特征详情", index=False)

            # 4.3 工作表2：全局特征平均值汇总
            summary_data = [
                {
                    "特征名称": "D2关联维数",
                    "有效样本数": global_feature_summary["D2"]["valid_count"],
                    "全局平均值": global_feature_summary["D2"]["mean_value"]
                },
                {
                    "特征名称": "PD2点关联维数",
                    "有效样本数": global_feature_summary["PD2"]["valid_count"],
                    "全局平均值": global_feature_summary["PD2"]["mean_value"]
                },
                {
                    "特征名称": "LE李雅普诺夫指数",
                    "有效样本数": global_feature_summary["LE"]["valid_count"],
                    "全局平均值": global_feature_summary["LE"]["mean_value"]
                },
                {
                    "特征名称": "KE科尔莫戈罗夫熵",
                    "有效样本数": global_feature_summary["KE"]["valid_count"],
                    "全局平均值": global_feature_summary["KE"]["mean_value"]
                }
            ]
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="全局特征平均值汇总", index=False)

        print(f"\n===== 所有结果已保存至 Excel：{os.path.join(save_dir, 'Seizure_Features_Summary.xlsx')} =====")
    except Exception as e:
        raise RuntimeError(f"Excel文件保存失败: {str(e)}")

    # 5. 整理返回结果
    return {
        "channel_detail_results": channel_detail_results,
        "global_feature_summary": global_feature_summary,
        "excel_path": os.path.join(save_dir, "Seizure_Features_Summary.xlsx"),
        "status": "success"
    }


def batch_calculate_non_seizure_features(channel_non_seizure_eeg_dict, fs, save_dir):
    """
    批量计算无发作期的混沌指数特征
    :param channel_non_seizure_eeg_dict: 无发作数据字典
    :param fs: 采样率
    :param save_dir: 保存目录
    :return: 结构化结果字典
    """
    # 1. 初始化结果存储结构
    channel_detail_results = {}
    global_d2_list = []
    global_pd2_mean_list = []
    global_le_mean_list = []
    global_ke_bits_list = []

    # 2. 遍历所有通道，处理无发作期数据
    print("===== 开始批量计算所有通道无发作期特征（D2/PD2/LE/KE） =====")
    for ch_name, non_seizure_list in tqdm(channel_non_seizure_eeg_dict.items(), desc="通道处理进度"):
        ch_non_seizure_results = []

        # 2.1 无有效数据的通道
        if len(non_seizure_list) == 0:
            channel_detail_results[ch_name] = {
                "seizure_count": 0,
                "seizure_details": [],
                "valid_features": False
            }
            continue

        # 2.2 有有效数据的通道，处理数据
        for non_seizure_info in non_seizure_list:
            non_seizure_eeg = non_seizure_info['eeg_data']
            start_sec = non_seizure_info['start_sec']
            end_sec = non_seizure_info['end_sec']

            # 初始化特征结果
            single_non_seizure_features = {
                "seizure_index": 0,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "D2": np.nan,
                "D2_list": [],
                "PD2_mean": np.nan,
                "PD2_sequence": [],
                "LE_mean": np.nan,
                "LE_sequence": [],
                "KE_bits": np.nan,
                "KE_result": None
            }

            try:
                # ===================== 计算 D2 关联维数 =====================
                mean_d2, d2_list = compute_d2(
                    signal=non_seizure_eeg,
                    fs=fs,
                    segment_duration=60,
                    m_candidates=[4, 6, 8]
                )
                single_non_seizure_features["D2"] = mean_d2
                single_non_seizure_features["D2_list"] = d2_list
                if not np.isnan(mean_d2) and mean_d2 > 0:
                    global_d2_list.append(mean_d2)

                # ===================== 计算 PD2 点关联维数 =====================
                pd2_seq = compute_pd2_eeg(
                    channel_data=non_seizure_eeg,
                    fs=fs,
                    win_len=1,
                    m_range=(3, 5),
                    max_delay=100
                )
                pd2_mean = np.nanmean(pd2_seq) if len(pd2_seq) > 0 else np.nan
                single_non_seizure_features["PD2_mean"] = pd2_mean
                single_non_seizure_features["PD2_sequence"] = pd2_seq
                if not np.isnan(pd2_mean):
                    global_pd2_mean_list.append(pd2_mean)

                # ===================== 计算 LE 李雅普诺夫指数 =====================
                t_axis, lle_series = compute_le_rosenstein(
                    signal_data=non_seizure_eeg,
                    fs=fs,
                    window_size_sec=10,
                    emb_dim=6
                )
                le_mean = np.nanmean(lle_series) if len(lle_series) > 0 else np.nan
                single_non_seizure_features["LE_mean"] = le_mean
                single_non_seizure_features["LE_sequence"] = lle_series
                if not np.isnan(le_mean):
                    global_le_mean_list.append(le_mean)

                # ===================== 计算 KE 科尔莫戈罗夫熵 =====================
                ke_result = compute_kolmogorov_entropy(
                    channel_data=non_seizure_eeg,
                    fs=fs,
                    save_dir=os.path.join(save_dir, f"KE_{ch_name}_non_seizure"),
                    num_bins=4,
                    max_k=8,
                    plot_fig=False
                )
                ke_bits = ke_result["ke_bits"] if ke_result["status"] == "success" else np.nan
                single_non_seizure_features["KE_bits"] = ke_bits
                single_non_seizure_features["KE_result"] = ke_result
                if not np.isnan(ke_bits):
                    global_ke_bits_list.append(ke_bits)

            except Exception as e:
                print(f"\n警告：{ch_name} 无发作期特征计算失败: {str(e)}")
                continue

            # 添加结果到通道列表
            ch_non_seizure_results.append(single_non_seizure_features)

        # 2.3 保存当前通道的结果
        channel_detail_results[ch_name] = {
            "seizure_count": len(ch_non_seizure_results),
            "seizure_details": ch_non_seizure_results,
            "valid_features": len([s for s in ch_non_seizure_results if
                                   not all(np.isnan([s['D2'], s['PD2_mean'], s['LE_mean'], s['KE_bits']]))]) > 0
        }

    # 3. 计算全局特征平均值
    global_feature_summary = {
        "D2": {
            "valid_count": len(global_d2_list),
            "mean_value": np.mean(global_d2_list) if len(global_d2_list) > 0 else np.nan
        },
        "PD2": {
            "valid_count": len(global_pd2_mean_list),
            "mean_value": np.mean(global_pd2_mean_list) if len(global_pd2_mean_list) > 0 else np.nan
        },
        "LE": {
            "valid_count": len(global_le_mean_list),
            "mean_value": np.mean(global_le_mean_list) if len(global_le_mean_list) > 0 else np.nan
        },
        "KE": {
            "valid_count": len(global_ke_bits_list),
            "mean_value": np.mean(global_ke_bits_list) if len(global_ke_bits_list) > 0 else np.nan
        }
    }

    # 4. 结果保存到Excel
    try:
        with pd.ExcelWriter(os.path.join(save_dir, "Non_Seizure_Features_Summary.xlsx"), engine="openpyxl") as writer:
            # 工作表1：通道无发作期特征详情
            detail_data = []
            for ch_name, ch_info in channel_detail_results.items():
                for non_seizure_detail in ch_info["seizure_details"]:
                    detail_data.append({
                        "通道名称": ch_name,
                        "起始时间(s)": non_seizure_detail["start_sec"],
                        "结束时间(s)": non_seizure_detail["end_sec"],
                        "时长(s)": non_seizure_detail["duration_sec"],
                        "D2关联维数": non_seizure_detail["D2"],
                        "PD2平均点关联维数": non_seizure_detail["PD2_mean"],
                        "LE平均李雅普诺夫指数": non_seizure_detail["LE_mean"],
                        "KE熵(bits/symbol)": non_seizure_detail["KE_bits"]
                    })
            df_detail = pd.DataFrame(detail_data)
            df_detail.to_excel(writer, sheet_name="通道无发作特征详情", index=False)

            # 工作表2：全局特征平均值汇总
            summary_data = [
                {
                    "特征名称": "D2关联维数",
                    "有效样本数": global_feature_summary["D2"]["valid_count"],
                    "全局平均值": global_feature_summary["D2"]["mean_value"]
                },
                {
                    "特征名称": "PD2点关联维数",
                    "有效样本数": global_feature_summary["PD2"]["valid_count"],
                    "全局平均值": global_feature_summary["PD2"]["mean_value"]
                },
                {
                    "特征名称": "LE李雅普诺夫指数",
                    "有效样本数": global_feature_summary["LE"]["valid_count"],
                    "全局平均值": global_feature_summary["LE"]["mean_value"]
                },
                {
                    "特征名称": "KE科尔莫戈罗夫熵",
                    "有效样本数": global_feature_summary["KE"]["valid_count"],
                    "全局平均值": global_feature_summary["KE"]["mean_value"]
                }
            ]
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="全局特征平均值汇总", index=False)

        print(
            f"\n===== 所有无发作期结果已保存至 Excel：{os.path.join(save_dir, 'Non_Seizure_Features_Summary.xlsx')} =====")
    except Exception as e:
        raise RuntimeError(f"Excel文件保存失败: {str(e)}")

    # 5. 整理返回结果
    return {
        "channel_detail_results": channel_detail_results,
        "global_feature_summary": global_feature_summary,
        "excel_path": os.path.join(save_dir, "Non_Seizure_Features_Summary.xlsx"),
        "status": "success"
    }