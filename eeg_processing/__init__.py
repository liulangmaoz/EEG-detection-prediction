# EEG Processing Package
# 版本: 1.0.0
# 功能: 脑电数据处理、癫痫检测、特征提取

from .core import *
from .preprocessing import *
from .detection import *
from .features import *
from .visualization import *
from .utils import *

__all__ = [
    # Core
    'load_nwb_data',
    'fs',
    'CHANNEL_NAMES',
    'PHASE_DURATION',
    'INTERVAL_DURATION',
    'ALPHA_BAND',
    'BETA_BAND',
    'ANALYSIS_BAND',
    'SAMPLING_FREQUENCY',
    'EEG_BANDS',
    
    # Preprocessing
    'preprocess_signal',
    
    # Detection
    'detect_seizures_multichannel',
    'detect_interictal_spikes_global',
    'detect_first_hfo_time_all_channels',
    'extend_seizure_to_1min',
    'divide_seizure_phases',
    
    # Features
    'batch_extract_seizure_eeg_data',
    'batch_extract_full_duration_eeg_data',
    'batch_extract_non_seizure_eeg_data',
    'batch_calculate_channel_phase_power',
    'batch_calculate_non_seizure_phase_power',
    'batch_calculate_phase_spectral_entropy',
    'batch_calculate_non_seizure_spectral_entropy',
    'batch_calculate_phase_band_energy',
    'batch_calculate_non_seizure_band_energy',
    'batch_calculate_seizure_features',
    'calculate_main_frequency_power',
    'calculate_alpha_beta_main_power',
    'calculate_power_spectral_entropy',
    'calculate_eeg_band_energy',
    
    # Visualization
    'generate_chart1_html',
    'plot_psd_comparison_html',
    'generate_spike_detection_report',
    'generate_seizure_detection_report',
    'generate_multi_channel_timefreq_report',
    'save_raw_data_view',
    'save_seizure_detection',
    'save_band_energy_results',
    'save_non_seizure_band_energy_results',
    'save_phase_power_results',
    'save_non_seizure_phase_power_results',
    'save_spectral_entropy_results',
    'save_non_seizure_spectral_entropy_results',
    'InteractiveChartGenerator',
    
    # Utils
    'save_dir',
    'font_cn',
    'font_en'
]
