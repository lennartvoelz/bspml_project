import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, find_peaks, resample, medfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Custom modules
from plot_utils import plot_signals, plot_signal
from ppg_pipeline.rls_filter import rls_filter
from data_loader import load_ppg_dalia

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
                           smoothing=None, kernel_size=5):
    """
    Estimate heart rate over time from peak locations.

    Parameters:
        peaks (np.ndarray): Detected peak times in seconds.
        window_size (float): Size of the sliding window in seconds.
        stride (float): Step size for sliding window in seconds.
        signal_len (float, optional): Total signal duration in seconds.
        smoothing (str, optional): Apply 'moving_average' or 'median' filter to smooth HR.
        kernel_size (int): Smoothing kernel size (must be odd).

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

    # Smoothing
    if smoothing == "moving_average":
        kernel = np.ones(kernel_size) / kernel_size
        estimated_hr = np.convolve(estimated_hr, kernel, mode='same')
    elif smoothing == "median":
        if kernel_size % 2 == 0:
            kernel_size += 1  # ensure odd kernel size
        estimated_hr = medfilt(estimated_hr, kernel_size=kernel_size)

    return est_times, estimated_hr


def evaluate_hr(estimated_times, estimated_hr, ground_truth, gt_fs=1):
    gt_times = np.arange(0, len(ground_truth)) / gt_fs
    interp_gt = interp1d(gt_times, ground_truth, kind='linear', fill_value='extrapolate')
    matched_gt = interp_gt(estimated_times)
    
    mae = mean_absolute_error(matched_gt, estimated_hr)
    rmse = np.sqrt(mean_squared_error(matched_gt, estimated_hr))
    corr, _ = pearsonr(matched_gt, estimated_hr)
    return mae, rmse, corr

df = pd.read_csv("../results/grid_search_results_20250711_071232.csv")

# Convert metrics to numeric (if they aren't already)
df['mae'] = pd.to_numeric(df['mae'], errors='coerce')
df['rmse'] = pd.to_numeric(df['rmse'], errors='coerce')
df['corr'] = pd.to_numeric(df['corr'], errors='coerce')

# Get the row with the lowest MAE
best_mae = df.loc[df['mae'].idxmin()]

# Get the row with the lowest RMSE
# best_rmse = df.loc[df['rmse'].idxmin()]

# Get the row with the highest correlation
# best_corr = df.loc[df['corr'].idxmax()]

print("Lowest MAE:\n", best_mae)
# print("\nLowest RMSE:\n", best_rmse)
# print("\nHighest Correlation:\n", best_corr)

# Load signals
base_path = "C:\\Users\\Florian\\Downloads\\datasets_cache\\ppg_dalia\\PPG_FieldStudy\\S1\\S1_E4"
# print(os.listdir(base_path))
signals = load_ppg_dalia(base_path)

fs_ppg = signals['ppg_fs']
ppg_raw = signals['ppg']
acc_xyz_raw = signals['acc_xyz']
# Resample ACC to match PPG length
acc_resampled = resample(acc_xyz_raw, len(ppg_raw), axis=0)

hr_gt = signals['hr']
hr_fs = signals['hr_fs']

start_s = best_mae['start_s']
end_s = best_mae['end_s']

ppg_segment = ppg_raw[int(start_s*fs_ppg):int(end_s*fs_ppg)]
acc_segment = acc_resampled[int(start_s*fs_ppg):int(end_s*fs_ppg)]
acc_segment = np.linalg.norm(acc_segment, axis=1)
gt_segment = hr_gt[int(start_s*hr_fs):int(end_s*hr_fs)]

params = best_mae

ppg_detrended = wavelet_detrend(ppg_segment, wavelet=params['wavelet'], level=params['wavelet_level'])
ppg_filtered = bandpass_filter(ppg_detrended, fs_ppg, params['bandpass_low'], params['bandpass_high'])
ppg_cleaned = rls_filter(ppg_filtered, acc_segment, forgetting_factor=params['rls_lambda'], order=params['rls_order'])

plot_signals(
    [ppg_segment, ppg_detrended, ppg_filtered, ppg_cleaned],
    fs=fs_ppg,
    labels=["Raw", "Detrended (Wavelet)", "Bandpass filtered", "RLS filtered"],
    title="Signal Cleanup",
    filename="signals",
    # ylim=(-0.5,0.5),
    xlim=(200, 210)
)

peaks, _ = detect_peaks(
    ppg_cleaned,
    fs=fs_ppg,
    min_dist=fs_ppg*params['min_distance'],
    prominence=params['prominence'],
    window_sec=params['window_size_sec'],
    stride_sec=params['stride_sec']
)
plot_signal(ppg_cleaned, fs=fs_ppg, label="Cleaned PPG", peaks=peaks, title="Detected Peaks", filename="peaks")

est_times, est_hr = estimate_hr_from_peaks(
    peaks,
    window_size=params['window_size_sec'],
    stride=params['stride_sec'],
    signal_len=len(ppg_cleaned)/fs_ppg
)
gt_times = np.arange(len(gt_segment)) / hr_fs
plot_signals(
    [],
    fs=fs_ppg,
    title="HR Estimation vs Ground Truth",
    hr_overlay=(est_times, est_hr, gt_times, gt_segment),
    filename="hr_comparison",
    ylim=(20, 200)
)

mae, rmse, corr = evaluate_hr(est_times, est_hr, gt_segment, gt_fs=hr_fs)

# print(f"[{act_name}] Params: {params} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, Corr: {corr:.2f}")
print(f"→ MAE: {mae:.2f}, RMSE: {rmse:.2f}, Corr: {corr:.2f}")