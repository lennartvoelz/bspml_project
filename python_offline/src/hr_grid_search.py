import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, find_peaks, resample, medfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import csv
from itertools import product
from datetime import datetime

# Custom modules
# from plot_utils import plot_signals, plot_signal
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

def load_activity_intervals(csv_path):
    """
    Load activity intervals from S1_activity.csv, using NO_ACTIVITY lines as end markers for previous activities.
    Returns a list of (activity_label, start_time, end_time) excluding NO_ACTIVITY intervals.
    """
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["activity", "time"])
    
    intervals = []
    current_activity = None
    current_start_time = None

    for _, row in df.iterrows():
        activity = row["activity"][1:].strip()
        time = row["time"]

        if activity == "NO_ACTIVITY":
            # Use NO_ACTIVITY as the end time if an activity is open
            if current_activity is not None:
                intervals.append((current_activity, current_start_time, time))
                current_activity = None
                current_start_time = None
        else:
            # Start a new activity
            current_activity = activity
            current_start_time = time

    return intervals

# os.makedirs('plots', exist_ok=True)

base_path = r"C:\Users\Florian\Downloads\datasets_cache\ppg_dalia\PPG_FieldStudy\S1"
activity_csv = os.path.join(base_path, "S1_activity.csv")
activity_intervals = load_activity_intervals(activity_csv)

# for label, start, end in activity_intervals:
#     print(f"{label} from {start}s to {end}s")

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

# Search space
grid_params = {
    "lowcut": [0.5, 0.6, 0.7],
    "highcut": [3.5, 4.0, 4.5],
    "rls_lambda": [0.90, 0.95, 0.99],
    "rls_order": [4],
    "window_size_sec": [3, 5, 7],
    "stride_sec": [1],
    "min_distance": [0.5, 0.6, 0.7],
    "prominence": [0.3, 0.5, 0.7],
    "wavelet": ["db6", "bior3.9"],
    "wavelet_level": [3, 4, 5, 6]
}

param_keys = list(grid_params.keys())
param_combos = list(product(*[grid_params[k] for k in param_keys]))

results = []

csv_file = f"grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# Define CSV columns (flatten params and metrics for convenience)
fieldnames = [
    "interval", "start_s", "end_s",
    # Flatten params keys here, example keys you might have:
    "wavelet", "wavelet_level", "bandpass_low", "bandpass_high",
    "rls_lambda", "rls_order",
    "window_size_sec", "stride_sec", "min_distance", "prominence",
    # Metrics:
    "mae", "rmse", "corr"
]
# Write header once before loop if file does not exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

for act_name, start_s, end_s in activity_intervals:
    print(f"Processing interval: {act_name} ({start_s}–{end_s}s)")
    
    ppg_segment = ppg_raw[int(start_s*fs_ppg):int(end_s*fs_ppg)]
    acc_segment = acc_resampled[int(start_s*fs_ppg):int(end_s*fs_ppg)]
    acc_segment = np.linalg.norm(acc_segment, axis=1)
    gt_segment = hr_gt[int(start_s*hr_fs):int(end_s*hr_fs)]

    for combo in param_combos:
        params = dict(zip(param_keys, combo))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # os.makedirs(f"plots/{act_name}/{timestamp}", exist_ok=True)

        if params['lowcut'] >= params['highcut']:
            continue

        # Process pipeline
        try:
            ppg_detrended = wavelet_detrend(ppg_segment, wavelet=params['wavelet'], level=params['wavelet_level'])
            ppg_filtered = bandpass_filter(ppg_detrended, fs_ppg, params['lowcut'], params['highcut'])
            ppg_cleaned = rls_filter(ppg_filtered, acc_segment, forgetting_factor=params['rls_lambda'], order=params['rls_order'])

            # plot_signals(
            #     [ppg_segment, ppg_detrended, ppg_filtered, ppg_cleaned],
            #     fs=fs_ppg,
            #     labels=["Raw", "Detrended (Wavelet)", "Bandpass filtered", "RLS filtered"],
            #     title="Signal Cleanup",
            #     filename=f"{act_name}/{timestamp}/signals"
            # )

            peaks, _ = detect_peaks(
                ppg_cleaned,
                fs=fs_ppg,
                min_dist=fs_ppg*params['min_distance'],
                prominence=params['prominence'],
                window_sec=params['window_size_sec'],
                stride_sec=params['stride_sec']
            )
            # plot_signal(ppg_cleaned, fs=fs_ppg, label="Cleaned PPG", peaks=peaks, title="Detected Peaks", filename=f"{act_name}/{timestamp}/peaks")

            est_times, est_hr = estimate_hr_from_peaks(
                peaks,
                window_size=params['window_size_sec'],
                stride=params['stride_sec'],
                signal_len=len(ppg_cleaned)/fs_ppg
            )

            if len(est_hr) < 3:
                continue

            # gt_times = np.arange(len(gt_segment)) / hr_fs
            # plot_signals(
            #     [],
            #     fs=fs_ppg,
            #     title="HR Estimation vs Ground Truth",
            #     hr_overlay=(est_times, est_hr, gt_times, gt_segment),
            #     filename=f"{act_name}/{timestamp}-hr_comparison"
            # )

            mae, rmse, corr = evaluate_hr(est_times, est_hr, gt_segment, gt_fs=hr_fs)

            # print(f"[{act_name}] Params: {params} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, Corr: {corr:.2f}")
            print(f"[{act_name}] → MAE: {mae:.2f}, RMSE: {rmse:.2f}, Corr: {corr:.2f}")

            result_row = {
                "interval": act_name,
                "start_s": start_s,
                "end_s": end_s,
                # params (flatten dictionary)
                "wavelet": params["wavelet"],
                "wavelet_level": params["wavelet_level"],
                "bandpass_low": params["lowcut"],
                "bandpass_high": params["highcut"],
                "rls_lambda": params["rls_lambda"],
                "rls_order": params["rls_order"],
                "window_size_sec": params["window_size_sec"],
                "stride_sec": params["stride_sec"],
                "min_distance": params["min_distance"],
                "prominence": params["prominence"],
                # metrics
                "mae": mae,
                "rmse": rmse,
                "corr": corr
            }

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result_row)

        except Exception as e:
            print(f"Error with parameters {params}: {e}")