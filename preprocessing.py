    # utils/preprocessing.py

import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from scipy.io import loadmat
import pywt

    # -----------------------------
    # 1. 滤波器设计
    # -----------------------------

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    low = lowcut / (0.5 * fs)
    high = highcut / (0.5 * fs)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, freq=50, fs=200):
# 50Hz notch filter to remove powerline noise
    b, a = butter(2, [(freq - 1)/(0.5*fs), (freq + 1)/(0.5*fs)], btype='bandstop')
    return filtfilt(b, a, data, axis=-1)

# -----------------------------
# 2. 小波去噪
# -----------------------------

def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

# -----------------------------
# 3. 多频段提取
# -----------------------------

def extract_band_power(data, fs, band_dict):
# 输入：shape=(channels, time)
    power_features = {}
    for band_name, (low, high) in band_dict.items():
        filtered = bandpass_filter(data, low, high, fs)
    power = np.log1p(np.square(filtered)) # log(power)
    power_features[band_name] = power
    return power_features # 返回一个字典

# -----------------------------
# 4. 归一化
# -----------------------------

def zscore_normalize(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)

# -----------------------------
# 5. 一键处理函数
# -----------------------------

def preprocess_eeg(raw_data, fs=1000, target_fs=200):
# raw_data: shape (channels, time)

# 1. 降采样
    factor = fs // target_fs
    data = raw_data[:, ::factor]

# 2. 去工频噪声
    data = notch_filter(data, freq=50, fs=target_fs)

# 3. 滤波（带通）
    data = bandpass_filter(data, 1, 50, fs=target_fs)

# 4. 小波去噪（每通道）
    data = np.array([wavelet_denoise(ch)[:data.shape[1]] for ch in data]) # 保证尺寸一致

# 5. 标准化
    data = zscore_normalize(data)

    return data # shape: (channels, time)