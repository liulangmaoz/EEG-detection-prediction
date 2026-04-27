import numpy as np
from scipy import signal

def calculate_short_time_energy(lfp_channel, fs, window_sec=0.5):
    """
    计算短时能量
    :param lfp_channel: 单通道脑电数据
    :param fs: 采样率
    :param window_sec: 窗口大小（秒）
    :return: 能量值
    """
    window_size = int(window_sec * fs)
    energy = np.sqrt(np.convolve(lfp_channel ** 2, np.ones(window_size) / window_size, mode='same'))
    return energy

def detect_seizures_multichannel(lfp_data, fs, baseline_window_sec=10, seizure_min_duration=5.0):
    """
    多通道大发作检测
    :param lfp_data: 脑电数据 (n_samples, n_channels)
    :param fs: 采样率
    :param baseline_window_sec: 基线窗口大小
    :param seizure_min_duration: 发作最小持续时间
    :return: 全局信息、最早通道信息、通道统计信息
    """
    print("\n===== 开始多通道大发作检测 =====")
    n_samples, n_channels = lfp_data.shape
    time_axis = np.arange(n_samples) / fs
    
    # 计算所有通道的短时能量
    energy_map = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        energy_map[:, ch] = calculate_short_time_energy(lfp_data[:, ch], fs, window_sec=0.5)
    
    # 定义全局能量
    global_energy = np.mean(energy_map, axis=1)
    
    # 确定基线窗口
    baseline_end_sample = int(0.1 * n_samples)
    if baseline_end_sample > 60000:
        baseline_end_sample = 60000
    print(f"基线窗口：前{baseline_end_sample / fs:.1f}秒（静息态）")
    
    # 计算全局基线阈值
    global_baseline_energy = global_energy[:baseline_end_sample]
    global_baseline_mean = np.mean(global_baseline_energy)
    global_baseline_std = np.std(global_baseline_energy)
    
    if np.isnan(global_baseline_mean) or np.isnan(global_baseline_std):
        print("警告：全局基线能量计算异常，将使用全局能量的均值和标准差替代")
        global_fallback_mean = np.nanmean(global_energy)
        global_fallback_std = np.nanstd(global_energy)
        global_threshold = global_fallback_mean + 3 * global_fallback_std
    else:
        global_threshold = global_baseline_mean + 3 * global_baseline_std
    
    print(f"全局基线能量统计: 均值={global_baseline_mean:.2f}, 标准差={global_baseline_std:.2f}")
    print(f"全局大发作检测阈值: {global_threshold:.2f} (均值 + 3σ)")
    
    # 为每个通道计算专属基线阈值
    print("\n正在为每个通道计算专属基线阈值...")
    channel_thresholds = np.zeros(n_channels)
    channel_baseline_means = np.zeros(n_channels)
    channel_baseline_stds = np.zeros(n_channels)
    
    for ch in range(n_channels):
        ch_baseline_energy = energy_map[:baseline_end_sample, ch]
        ch_baseline_mean = np.mean(ch_baseline_energy)
        ch_baseline_std = np.std(ch_baseline_energy)
        
        channel_baseline_means[ch] = ch_baseline_mean
        channel_baseline_stds[ch] = ch_baseline_std
        
        if np.isnan(ch_baseline_mean) or np.isnan(ch_baseline_std):
            print(f"警告：通道{ch}基线能量计算异常，将使用该通道全局能量的均值和标准差替代")
            ch_fallback_mean = np.nanmean(energy_map[:, ch])
            ch_fallback_std = np.nanstd(energy_map[:, ch])
            ch_threshold = ch_fallback_mean + 3 * ch_fallback_std
        else:
            ch_threshold = ch_baseline_mean + 3 * ch_baseline_std
        
        channel_thresholds[ch] = ch_threshold
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"通道{ch}"
        print(f"通道{ch_name}：基线均值={ch_baseline_mean:.2f}, 标准差={ch_baseline_std:.2f}, 专属阈值={ch_threshold:.2f}")
    
    # 全局能量验证
    print("\n正在进行全局能量验证（确证全脑大发作）...")
    global_above_thresh = global_energy > global_threshold
    global_edges = np.diff(global_above_thresh.astype(int))
    global_starts = np.where(global_edges == 1)[0] + 1
    global_ends = np.where(global_edges == -1)[0]
    
    if global_above_thresh[0]:
        global_starts = np.insert(global_starts, 0, 0)
    if global_above_thresh[-1]:
        global_ends = np.append(global_ends, len(global_above_thresh) - 1)
    
    valid_global_events = []
    min_samples = int(seizure_min_duration * fs)
    for s, e in zip(global_starts, global_ends):
        if (e - s) >= min_samples:
            valid_global_events.append({
                'start_idx': s,
                'end_idx': e,
                'start_time': time_axis[s],
                'end_time': time_axis[e],
                'duration': time_axis[e] - time_axis[s]
            })
    
    # 单通道自身阈值检测
    print("\n正在进行单通道局部早期发作检测（使用专属阈值）...")
    channel_first_seizure_time = np.full(n_channels, np.inf)
    channel_valid_events = []
    
    for ch in range(n_channels):
        ch_threshold = channel_thresholds[ch]
        ch_above_thresh = energy_map[:, ch] > ch_threshold
        
        ch_edges = np.diff(ch_above_thresh.astype(int))
        ch_starts = np.where(ch_edges == 1)[0] + 1
        ch_ends = np.where(ch_edges == -1)[0]
        
        if ch_above_thresh[0]:
            ch_starts = np.insert(ch_starts, 0, 0)
        if ch_above_thresh[-1]:
            ch_ends = np.append(ch_ends, len(ch_above_thresh) - 1)
        
        ch_valid_events_list = []
        for s, e in zip(ch_starts, ch_ends):
            if (e - s) >= min_samples:
                event_info = {
                    'start_idx': s,
                    'end_idx': e,
                    'start_time': time_axis[s],
                    'end_time': time_axis[e],
                    'duration': time_axis[e] - time_axis[s],
                    'channel_threshold': ch_threshold
                }
                ch_valid_events_list.append(event_info)
        
        channel_valid_events.append(ch_valid_events_list)
        
        if len(ch_valid_events_list) > 0:
            channel_first_seizure_time[ch] = ch_valid_events_list[0]['start_time']
    
    # 最早发作通道溯源
    print("\n正在溯源最早发作通道（定位起始灶）...")
    earliest_channel = np.argmin(channel_first_seizure_time)
    if channel_first_seizure_time[earliest_channel] == np.inf:
        print("未检测到任何通道的有效局部发作事件")
        earliest_channel_events = []
    else:
        earliest_channel_events = channel_valid_events[earliest_channel]
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[earliest_channel] if earliest_channel < len(CHANNEL_NAMES) else f"通道{earliest_channel}"
        print(f"最早出现发作的通道：{ch_name}（首次发作时间：{earliest_channel_events[0]['start_time']:.2f}s）")
    
    # 输出检测汇总信息
    print(f"\n===== 检测结果汇总 =====")
    print(f"全局确证全脑大发作：{len(valid_global_events)} 次（持续时间 > {seizure_min_duration}s）")
    if len(earliest_channel_events) > 0:
        print(f"最早发作通道检测到局部发作：{len(earliest_channel_events)} 次")
    total_channel_events = sum([len(ch_events) for ch_events in channel_valid_events])
    print(f"所有通道累计检测到局部早期发作：{total_channel_events} 次")
    print("===== 发作检测完成 =====\n")
    
    # 封装为3个字典
    global_info = {
        'valid_global_events': valid_global_events,
        'global_threshold': global_threshold,
        'global_energy': global_energy
    }
    
    earliest_channel_info = {
        'earliest_channel': earliest_channel,
        'earliest_channel_events': earliest_channel_events
    }
    
    channel_stats_info = {
        'channel_valid_events': channel_valid_events,
        'channel_thresholds': channel_thresholds,
        'channel_baseline_means': channel_baseline_means,
        'channel_baseline_stds': channel_baseline_stds
    }
    
    return global_info, earliest_channel_info, channel_stats_info

def extend_seizure_to_1min(seizure_start_sec, seizure_end_sec, fs, total_duration_sec, prev_seizure_end_sec=None):
    """
    将原始发作期扩展为1分钟
    :param seizure_start_sec: 发作开始时间
    :param seizure_end_sec: 发作结束时间
    :param fs: 采样率
    :param total_duration_sec: 总时长
    :param prev_seizure_end_sec: 前一个发作结束时间
    :return: 扩展后的发作信息
    """
    original_duration = seizure_end_sec - seizure_start_sec
    from .core import PHASE_DURATION
    target_duration = PHASE_DURATION
    missing_duration = max(0, target_duration - original_duration)
    
    if prev_seizure_end_sec is not None:
        interval_with_prev = seizure_start_sec - prev_seizure_end_sec
        if interval_with_prev < 30.0:
            pre_extend = 0.0
            post_extend = missing_duration
        else:
            pre_extend = missing_duration / 2.0
            post_extend = missing_duration / 2.0
    else:
        pre_extend = missing_duration / 2.0
        post_extend = missing_duration / 2.0
    
    extended_start = seizure_start_sec - pre_extend
    extended_end = seizure_end_sec + post_extend
    
    if original_duration >= target_duration:
        mid_point = (seizure_start_sec + seizure_end_sec) / 2.0
        extended_start = mid_point - target_duration / 2.0
        extended_end = mid_point + target_duration / 2.0
    else:
        extended_end = extended_start + target_duration
    
    extended_start = max(0.0, extended_start)
    extended_end = min(total_duration_sec, extended_end)
    
    if (extended_end - extended_start) < target_duration:
        remaining_missing = target_duration - (extended_end - extended_start)
        extended_end = min(total_duration_sec, extended_end + remaining_missing)
        if (extended_end - extended_start) < target_duration:
            extended_start = max(0.0, extended_start - (target_duration - (extended_end - extended_start)))
    
    extended_start = round(extended_start, 2)
    extended_end = round(min(extended_start + target_duration, extended_end), 2)
    
    return {
        'start_sec': extended_start,
        'end_sec': extended_end,
        'duration_sec': extended_end - extended_start,
        'is_valid': (extended_end - extended_start) >= 30.0
    }

def divide_seizure_phases(channel_stats_info, fs, total_samples, ch_idx):
    """
    划分单个通道的所有发作阶段
    :param channel_stats_info: 通道统计信息
    :param fs: 采样率
    :param total_samples: 总采样点数
    :param ch_idx: 通道索引
    :return: 发作阶段列表
    """
    channel_valid_events = channel_stats_info['channel_valid_events']
    
    if ch_idx < 0 or ch_idx >= len(channel_valid_events):
        raise ValueError(f"通道索引 {ch_idx} 超出有效范围（0 ~ {len(channel_valid_events) - 1}）")
    
    ch_local_events = channel_valid_events[ch_idx]
    seizure_phases = []
    total_duration_sec = total_samples / fs
    prev_seizure_end_sec = None
    
    for seizure_idx, seizure_event in enumerate(ch_local_events, 1):
        orig_seizure_start = seizure_event['start_time']
        orig_seizure_end = seizure_event['end_time']
        
        extended_seizure = extend_seizure_to_1min(
            seizure_start_sec=orig_seizure_start,
            seizure_end_sec=orig_seizure_end,
            fs=fs,
            total_duration_sec=total_duration_sec,
            prev_seizure_end_sec=prev_seizure_end_sec
        )
        
        if not extended_seizure['is_valid']:
            continue
        
        prev_seizure_end_sec = extended_seizure['end_sec']
        
        from .core import INTERVAL_DURATION, PHASE_DURATION
        pre_phase_start = extended_seizure['start_sec'] - 2 * INTERVAL_DURATION
        pre_phase_end = extended_seizure['start_sec'] - INTERVAL_DURATION
        pre_phase_valid = (pre_phase_start >= 0.0) and (pre_phase_end <= total_duration_sec) and (pre_phase_end > pre_phase_start)
        
        post_phase_start = extended_seizure['end_sec'] + INTERVAL_DURATION
        post_phase_end = extended_seizure['end_sec'] + 2 * INTERVAL_DURATION
        post_phase_valid = (post_phase_start >= 0.0) and (post_phase_end <= total_duration_sec) and (post_phase_end > post_phase_start)
        
        current_seizure_phases = {
            'seizure_id': seizure_idx,
            'attack_phase': {
                'phase_name': f"发作期{seizure_idx}",
                'start_sec': extended_seizure['start_sec'],
                'end_sec': extended_seizure['end_sec'],
                'duration_sec': extended_seizure['duration_sec'],
                'is_valid': extended_seizure['is_valid'],
                'original_start': orig_seizure_start,
                'original_end': orig_seizure_end
            },
            'pre_phase': {
                'phase_name': f"发作前期{seizure_idx}",
                'start_sec': pre_phase_start,
                'end_sec': pre_phase_end,
                'duration_sec': PHASE_DURATION,
                'is_valid': pre_phase_valid
            } if pre_phase_valid else None,
            'post_phase': {
                'phase_name': f"发作后期{seizure_idx}",
                'start_sec': post_phase_start,
                'end_sec': post_phase_end,
                'duration_sec': PHASE_DURATION,
                'is_valid': post_phase_valid
            } if post_phase_valid else None
        }
        
        seizure_phases.append(current_seizure_phases)
    
    return seizure_phases

def detect_interictal_spikes_global(lfp_data, fs, global_info, channel_stats_info, threshold_sd=3):
    """
    检测间期棘波
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param global_info: 全局信息
    :param channel_stats_info: 通道统计信息
    :param threshold_sd: 阈值倍数
    :return: 总棘波数和各通道棘波数
    """
    print("===== 开始检测间期棘波 (排除发作期，复用通道专属基线阈值) =====")
    n_samples, n_channels = lfp_data.shape
    total_spikes = 0
    channel_spikes = []
    
    seizure_events = global_info['valid_global_events']
    channel_baseline_means = channel_stats_info['channel_baseline_means']
    channel_baseline_stds = channel_stats_info['channel_baseline_stds']
    
    if len(channel_baseline_means) != n_channels or len(channel_baseline_stds) != n_channels:
        raise ValueError("通道基线均值/标准差数组长度与通道数不匹配！")
    
    # 生成发作期的掩码
    is_in_seizure = np.zeros(n_samples, dtype=bool)
    for ev in seizure_events:
        s, e = ev['start_idx'], ev['end_idx']
        s = max(0, s)
        e = min(n_samples, e)
        is_in_seizure[s:e] = True
    
    for ch_idx in range(n_channels):
        lfp = lfp_data[:, ch_idx]
        ch_mean = channel_baseline_means[ch_idx]
        ch_std = channel_baseline_stds[ch_idx]
        
        if ch_std == 0:
            from .core import CHANNEL_NAMES
            ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
            print(f"警告：通道{ch_name}基线标准差为0，跳过该通道棘波检测")
            channel_spikes.append(0)
            continue
        
        peak_amp = ch_mean + threshold_sd * ch_std
        peaks, _ = signal.find_peaks(lfp, height=peak_amp, distance=int(0.2 * fs))
        valid_spikes = peaks[~is_in_seizure[peaks]]
        current_channel_spike_count = len(valid_spikes)
        total_spikes += current_channel_spike_count
        
        channel_spikes.append(current_channel_spike_count)
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        
        if ch_idx < 12:
            print(f"通道{ch_name}: 基线均值={ch_mean:.2f}, 基线标准差={ch_std:.2f}, 总峰数={len(peaks)}, 间期棘波数={current_channel_spike_count}")
    
    print(f"\n所有通道合计间期棘波: {total_spikes} 个")
    print("===== 间期棘波检测完成 =====\n")
    
    return total_spikes, channel_spikes

def detect_first_hfo_time_all_channels(lfp_data, fs, channel_stats_info):
    """
    检测第一次HFO出现时间
    :param lfp_data: 脑电数据
    :param fs: 采样率
    :param channel_stats_info: 通道统计信息
    :return: 各通道第一次HFO出现时间
    """
    print("\n===== 开始按单通道独立窗口检测第一次HFO出现时间 =====")
    n_samples, n_channels = lfp_data.shape
    hfo_first_time = {}
    
    nyq = fs / 2
    try:
        sos_ripple = signal.butter(4, [80 / nyq, 250 / nyq], btype='bandpass', output='sos')
        sos_fripple = signal.butter(4, [250 / nyq, 450 / nyq], btype='bandpass', output='sos')
    except ValueError as e:
        print(f"滤波器设计失败：{e}，请检查采样率是否满足奈奎斯特要求")
        return hfo_first_time
    
    channel_valid_events = channel_stats_info['channel_valid_events']
    
    if len(channel_valid_events) != n_channels:
        raise ValueError("通道有效发作事件列表长度与通道数不匹配！")
    
    channel_seizure_data_list = []
    for ch_valid_events in channel_valid_events:
        if len(ch_valid_events) > 0:
            earliest_event = ch_valid_events[0]
            ch_seizure_start_sample = earliest_event['start_idx']
            ch_seizure_duration_sample = earliest_event['end_idx'] - earliest_event['start_idx']
            ch_seizure_duration_sec = earliest_event['duration']
            channel_seizure_data_list.append((ch_seizure_start_sample, ch_seizure_duration_sample, ch_seizure_duration_sec))
        else:
            channel_seizure_data_list.append((None, None, None))
    
    for ch_idx in range(n_channels):
        from .core import CHANNEL_NAMES
        ch_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"通道{ch_idx}"
        ch_seizure_data = channel_seizure_data_list[ch_idx]
        ch_seizure_start_sample, _, _ = ch_seizure_data
        
        if ch_seizure_start_sample is None:
            continue
        
        window_pre = int(1.0 * fs)
        window_post = int(2.0 * fs)
        start_analysis = max(0, ch_seizure_start_sample - window_pre)
        end_analysis = min(n_samples, ch_seizure_start_sample + window_post)
        
        if end_analysis <= start_analysis:
            continue
        
        segment = lfp_data[start_analysis:end_analysis, ch_idx]
        sig_ripple = signal.sosfiltfilt(sos_ripple, segment)
        sig_fripple = signal.sosfiltfilt(sos_fripple, segment)
        
        env_ripple = np.abs(signal.hilbert(sig_ripple))
        env_fripple = np.abs(signal.hilbert(sig_fripple))
        
        smooth_window = int(0.05 * fs)
        if smooth_window < 1:
            smooth_window = 1
        env_ripple_smooth = np.convolve(env_ripple, np.ones(smooth_window) / smooth_window, mode='same')
        env_fripple_smooth = np.convolve(env_fripple, np.ones(smooth_window) / smooth_window, mode='same')
        
        th_r = np.mean(env_ripple_smooth) + 3 * np.std(env_ripple_smooth)
        th_f = np.mean(env_fripple_smooth) + 3 * np.std(env_fripple_smooth)
        
        is_hfo = (env_ripple_smooth > th_r) | (env_fripple_smooth > th_f)
        hfo_indices = np.where(is_hfo)[0]
        
        if len(hfo_indices) > 0:
            first_hfo_idx_local = hfo_indices[0]
            first_hfo_idx_global = start_analysis + first_hfo_idx_local
            first_hfo_time_abs = round(first_hfo_idx_global / fs, 3)
            hfo_first_time[ch_name] = first_hfo_time_abs
            print(f"{ch_name}：第一次HFO出现时间 = {first_hfo_time_abs} 秒（绝对时间，数据开始为0秒）")
        else:
            print(f"{ch_name}：有局部发作，但未检测到有效HFO")
    
    print("\n===== 所有有效通道HFO独立窗口检测完成 =====\n")
    return hfo_first_time
