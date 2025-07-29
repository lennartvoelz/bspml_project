import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, medfilt, lfilter
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Custom modules
from .rls_filter import rls_filter

def run_all_steps(ppg, acc, hr_gt, ppg_fs=64.0, hr_fs=1.0, wv='bior3.9', wv_level=6, bp_lowcut=0.6, bp_highcut=4.0, rls_lambda=0.99, rls_order=4, peak_min_distance=0.7, peak_prominence=0.7, peak_window_size_sec=7, peak_stride_sec=1):

    ppg_detrended = wavelet_detrend(ppg, wavelet=wv, level=wv_level)
    ppg_filtered = bandpass_filter(ppg_detrended, ppg_fs, bp_lowcut, bp_highcut)
    ppg_cleaned = rls_filter(ppg_filtered, acc, forgetting_factor=rls_lambda, order=rls_order)
    # ppg_cleaned = ppg_filtered

    peaks, _ = detect_peaks(
        ppg_cleaned,
        fs=ppg_fs,
        min_dist=ppg_fs*peak_min_distance,
        prominence=peak_prominence,
        window_sec=peak_window_size_sec,
        stride_sec=peak_stride_sec
    )

    peaks2, _2 = detect_peaks(
        ppg_filtered,
        fs=ppg_fs,
        min_dist=ppg_fs*peak_min_distance,
        prominence=peak_prominence,
        window_sec=peak_window_size_sec,
        stride_sec=peak_stride_sec
    )

    est_times, est_hr = estimate_hr_from_peaks(
        peaks,
        window_size=peak_window_size_sec,
        stride=peak_stride_sec,
        signal_len=len(ppg_cleaned)/ppg_fs,
        smoothing="butterworth+exp"
    )

    # mae, rmse, corr = evaluate_hr(est_times, est_hr, hr_gt, gt_fs=hr_fs)
    
    return {
        "peaks": peaks,
        "peaks2": peaks2,
        "ppg_detrended": ppg_detrended,
        "ppg_filtered": ppg_filtered,
        "ppg_cleaned": ppg_cleaned,
        "hr_estimation": {
            "est_times": est_times,
            "est_hr": est_hr
        },
        # "metrics": {
        #     "mae": mae,
        #     "rmse": rmse,
        #     "corr": corr
        # }
    }

def run_preprocessing(ppg, acc, ppg_fs=64.0, wv='bior3.9', wv_level=6, bp_lowcut=0.6, bp_highcut=4.0, rls_lambda=0.99, rls_order=4, peak_min_distance=0.7, peak_prominence=0.7, peak_window_size_sec=7, peak_stride_sec=1):
    ppg_detrended = wavelet_detrend(ppg, wavelet=wv, level=wv_level)
    ppg_filtered = bandpass_filter(ppg_detrended, ppg_fs, bp_lowcut, bp_highcut)
    ppg_cleaned = rls_filter(ppg_filtered, acc, forgetting_factor=rls_lambda, order=rls_order)
    peaks, _ = detect_peaks(
        ppg_cleaned,
        fs=ppg_fs,
        min_dist=ppg_fs*peak_min_distance,
        prominence=peak_prominence,
        window_sec=peak_window_size_sec,
        stride_sec=peak_stride_sec
    )
    return {
        "peaks": peaks,
        "ppg_detrended": ppg_detrended,
        "ppg_filtered": ppg_filtered,
        "ppg_cleaned": ppg_cleaned
    }

def wavelet_detrend(signal, wavelet='db6', level=6):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # Remove approximation
    return pywt.waverec(coeffs, wavelet)

def bandpass_filter(signal, fs, low=0.5, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def detect_peaks(ppg, fs, min_dist, prominence=0.5, window_sec=5, stride_sec=1):
    peaks = []
    times = []
    window_size = int(window_sec * fs)
    stride_size = int(stride_sec * fs)
    # min_dist = fs * 60 / max_bpm
    # min_dist = fs * 0.6
    for start in range(0, len(ppg) - window_size, stride_size):
        window = ppg[start:start + window_size]
        local_peaks, _ = find_peaks(window, distance=min_dist, prominence=prominence)
        peak_times = (local_peaks + start) / fs
        peaks.extend(peak_times)
        times.append(start + window_sec * fs // 2)
    return np.array(sorted(list(set(peaks)))), np.array(times) / fs

def estimate_hr_from_peaks(peaks, window_size=5, stride=1, signal_len=None, 
                           smoothing=None, kernel_size=5,
                           fs_hr=1.0, fc_lp=0.33, alpha_exp=0.2):
    """
    Estimate heart rate over time from peak locations.

    Parameters:
        peaks (np.ndarray): Detected peak times in seconds.
        window_size (float): Size of the sliding window in seconds.
        stride (float): Step size for sliding window in seconds.
        signal_len (float, optional): Total signal duration in seconds.
        smoothing (str, optional): 'moving_average', 'median', or 'butterworth+exp'.
        kernel_size (int): Smoothing kernel size.
        fs_hr (float): Sampling frequency of HR estimates (Hz). Only for butterworth+exp.
        fc_lp (float): Cutoff frequency of low-pass filter for HR (Hz).
        alpha_exp (float): Alpha parameter for exponential smoothing.

    Returns:
        est_times (np.ndarray): Time points for each HR estimate (center of window).
        estimated_hr (np.ndarray): Estimated heart rate in BPM.
    """
    estimated_hr = []
    est_times = []

    end_time = signal_len if signal_len is not None else peaks[-1]

    for start in np.arange(0, end_time - window_size, stride):
        window_peaks = peaks[(peaks >= start) & (peaks < start + window_size)]
        if len(window_peaks) > 1:
            ibi = np.diff(window_peaks)
            hr = 60.0 / np.mean(ibi)
            estimated_hr.append(hr)
            est_times.append(start + window_size / 2)

    estimated_hr = np.array(estimated_hr)
    est_times = np.array(est_times)

    # Smoothing options
    if smoothing == "moving_average":
        kernel = np.ones(kernel_size) / kernel_size
        estimated_hr = np.convolve(estimated_hr, kernel, mode='same')
    elif smoothing == "median":
        if kernel_size % 2 == 0:
            kernel_size += 1
        estimated_hr = medfilt(estimated_hr, kernel_size=kernel_size)
    elif smoothing == "butterworth+exp" and len(estimated_hr) > 3:
        # Butterworth low-pass filter
        b, a = butter(2, fc_lp / (fs_hr / 2))
        hr_lp = filtfilt(b, a, estimated_hr)

        # Exponential smoothing (IIR)
        estimated_hr = lfilter([alpha_exp], [1, alpha_exp - 1], hr_lp)

    return est_times, estimated_hr


def evaluate_hr(estimated_times, estimated_hr, ground_truth, gt_fs=1):
    gt_times = np.arange(0, len(ground_truth)) / gt_fs
    interp_gt = interp1d(gt_times, ground_truth, kind='linear', fill_value='extrapolate')
    matched_gt = interp_gt(estimated_times)
    
    mae = mean_absolute_error(matched_gt, estimated_hr)
    rmse = np.sqrt(mean_squared_error(matched_gt, estimated_hr))
    corr, _ = pearsonr(matched_gt, estimated_hr)
    return mae, rmse, corr