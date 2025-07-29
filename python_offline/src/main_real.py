import os
import numpy as np
import pandas as pd
from scipy.signal import resample

from ppg_pipeline.pipeline2 import run_preprocessing, run_all_steps
from plot_utils import plot_signals, plot_signal

# === Load your new format file ===
file_path = os.path.join(
    'C:\\Users\\Florian\\Nextcloud\\Uni_Heidelberg\\BMSP_ML\\bpml_final_project\\data',
    "2025-07-14(14_11_26)_sitting_....txt"
)

# Load and parse data
data = pd.read_csv(file_path, sep="\t", header=0)
ppg_signal = data['RED'].values.astype(float)
acc_signal = data['ACC'].values.astype(float)

# === Define sampling frequency manually (e.g., 50Hz) ===
ppg_fs = 50.0  # adjust this if needed

# Normalize ACC since it's 1D
acc_resampled = resample(acc_signal, len(ppg_signal))

# Optional: Segment range in seconds
start_s, end_s = 0, len(ppg_signal) / ppg_fs

# Slice segments
ppg_segment = ppg_signal[int(start_s*ppg_fs):int(end_s*ppg_fs)]
acc_segment = acc_resampled[int(start_s*ppg_fs):int(end_s*ppg_fs)]

# Run preprocessing pipeline
# results = run_preprocessing(
#     ppg=ppg_segment,
#     ppg_fs=ppg_fs,
#     acc=acc_segment,
#     peak_min_distance=0.8
# )
results = run_all_steps(
    ppg=ppg_segment,
    ppg_fs=ppg_fs,
    hr_gt=[],
    acc=acc_segment,
    peak_min_distance=0.5,
    peak_prominence=None,
    peak_window_size_sec=8,
    peak_stride_sec=4
)

# Unpack results
peaks = results['peaks']
ppg_detrended = results['ppg_detrended']
ppg_filtered = results['ppg_filtered']
ppg_cleaned = results['ppg_cleaned']

# Plotting
plot_signals(
    [ppg_segment, ppg_detrended, ppg_filtered, ppg_cleaned],
    fs=ppg_fs,
    labels=['Raw', 'Detrended (Wavelet)', 'Bandpass filtered', 'RLS filtered'],
    title='Signal Cleanup',
    # filename='signals',
    # xlim=(200, 210)
)
# quit()



plot_signal(
    ppg_cleaned,
    fs=ppg_fs,
    label='Cleaned PPG',
    peaks=peaks,
    title='Detected Peaks',
    filename='peaks'
)

def plot_estimated_hr(hr_estimation, title="Estimated Heart Rate", filename=None):
    """
    Plot estimated heart rate over time.
    
    Parameters:
    - hr_estimation: dict with keys "est_times" and "est_hr"
    - title: Plot title
    - filename: If provided, saves the figure as PNG
    """
    est_times = hr_estimation['est_times']
    est_hr = hr_estimation['est_hr']

    plt.figure(figsize=(10, 4))
    plt.plot(est_times, est_hr, marker='o', linestyle='-', color='tab:red')
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title(title)
    plt.grid(True)
    
    if filename:
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
    plt.show()

hr_estimation = results['hr_estimation']
est_times = hr_estimation['est_times']
est_hr = hr_estimation['est_hr']

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(est_times, est_hr, marker='o', linestyle='-', color='tab:red')
plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (BPM)")
plt.title('HR estimation')
plt.grid(True)
plt.show()

def plot_acc_magnitude(acc_magnitude, acc_fs, title="ACC Magnitude Over Time", filename=None):
    """
    Plot ACC magnitude over time.

    Parameters:
    - acc_magnitude: 1D numpy array of ACC magnitude values
    - acc_fs: sampling frequency of the ACC signal (Hz)
    - title: plot title
    - filename: if provided, saves the plot as PNG
    """
    duration = len(acc_magnitude) / acc_fs
    time_axis = np.linspace(0, duration, len(acc_magnitude))

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, acc_magnitude, color='tab:blue')
    plt.xlabel("Time (s)")
    plt.ylabel("ACC Magnitude (a.u.)")
    plt.title(title)
    plt.grid(True)

    if filename:
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
    plt.show()

acc_magnitude = np.abs(acc_segment)  # If 1D ACC
# Or use: acc_magnitude = np.linalg.norm(acc_xyz_segment, axis=1)  # if 3D ACC

plot_acc_magnitude(acc_magnitude, acc_fs=50.0, title="Accelerometer Magnitude", filename="acc_plot")

def plot_ppg_stages(results, ppg_fs, title_prefix="PPG Processing Stages", filename=None, xlim=None):
    """
    Plot ppg_detrended, ppg_filtered, and ppg_cleaned side by side.

    Parameters:
    - results: dictionary returned by `run_all_steps`, containing the PPG stages
    - ppg_fs: sampling frequency (Hz)
    - title_prefix: string prefix for subplot titles
    - filename: if provided, saves figure as PNG
    - xlim: tuple (start_sec, end_sec) to limit time axis
    """
    ppg_detrended = results['ppg_detrended']
    ppg_filtered = results['ppg_filtered']
    ppg_cleaned = results['ppg_cleaned']

    signal_names = ['Detrended', 'Filtered', 'RLS Cleaned']
    signals = [ppg_detrended, ppg_filtered, ppg_cleaned]

    time_axis = np.arange(len(ppg_detrended)) / ppg_fs
    if xlim:
        idx_start = int(xlim[0] * ppg_fs)
        idx_end = int(xlim[1] * ppg_fs)
        time_axis = time_axis[idx_start:idx_end]
        signals = [s[idx_start:idx_end] for s in signals]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time_axis, signals[i], color='tab:blue')
        ax.set_title(f"{title_prefix}: {signal_names[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
    plt.show()

plot_ppg_stages(results, ppg_fs)

def plot_ppg_and_acc(ppg_signal, acc_signal, ppg_fs, acc_fs, title="PPG and ACC Signal", filename=None):
    """
    Plot PPG and ACC (magnitude) signals in a single figure with two y-axes.

    Parameters:
    - ppg_signal: 1D NumPy array (e.g., raw or cleaned PPG)
    - acc_signal: 1D NumPy array (ACC magnitude)
    - ppg_fs: sampling frequency of the PPG signal
    - acc_fs: sampling frequency of the ACC signal
    - title: plot title
    - filename: optional filename to save the plot
    """
    # Resample ACC to match PPG length (if needed)
    if len(acc_signal) != len(ppg_signal):
        acc_signal = np.interp(
            np.linspace(0, len(ppg_signal) - 1, len(ppg_signal)),
            np.linspace(0, len(acc_signal) - 1, len(acc_signal)),
            acc_signal
        )

    time_axis = np.arange(len(ppg_signal)) / ppg_fs

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # PPG plot
    ax1.plot(time_axis, ppg_signal, color='tab:red', label='PPG')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("PPG Signal", color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # ACC plot (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.plot(time_axis, acc_signal, color='tab:blue', label='ACC', alpha=0.6)
    ax2.set_ylabel("ACC Magnitude", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Title and grid
    plt.title(title)
    fig.tight_layout()
    ax1.grid(True)

    # Save
    if filename:
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')

    plt.show()

ppg_cleaned = results['ppg_cleaned']  # or any PPG stage
acc_magnitude = np.abs(acc_segment)  # or np.linalg.norm(acc_xyz, axis=1)

plot_ppg_and_acc(ppg_cleaned, acc_magnitude, ppg_fs=50, acc_fs=50, title="PPG and ACC over Time", filename="ppg_acc_plot")