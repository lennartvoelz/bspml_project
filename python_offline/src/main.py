import os
from scipy.signal import resample
import numpy as np

from ppg_pipeline.pipeline import run_all_steps, run_preprocessing
from data_loader import load_ppg_dalia, load_activity_intervals
from plot_utils import plot_signals, plot_signal, plot_estimated_hr

base_path = r'C:\Users\Florian\Downloads\datasets_cache\ppg_dalia\PPG_FieldStudy\S1'
activity_csv = os.path.join(base_path, "S1_activity.csv")
activity_intervals = load_activity_intervals(activity_csv)

for label, start, end in activity_intervals:
    print(f"{label} from {start}s to {end}s")
act_name, start_s, end_s = activity_intervals[0]

# Load signals
base_path = 'C:\\Users\\Florian\\Downloads\\datasets_cache\\ppg_dalia\\PPG_FieldStudy\\S1\\S1_E4'
# print(os.listdir(base_path))
signals = load_ppg_dalia(base_path)

ppg_fs = signals['ppg_fs']
# print(f"PPG FS: {ppg_fs}")
ppg_raw = signals['ppg']
acc_xyz_raw = signals['acc_xyz']
# Resample ACC to match PPG length
acc_resampled = resample(acc_xyz_raw, len(ppg_raw), axis=0)

hr_gt = signals['hr']
hr_fs = signals['hr_fs']
# print(f"HR FS: {hr_fs}")

ppg_segment = ppg_raw[int(start_s*ppg_fs):int(end_s*ppg_fs)]
acc_segment = acc_resampled[int(start_s*ppg_fs):int(end_s*ppg_fs)]
acc_segment = np.linalg.norm(acc_segment, axis=1)
gt_segment = hr_gt[int(start_s*hr_fs):int(end_s*hr_fs)]

# results = run_preprocessing(ppg=ppg_segment, ppg_fs=ppg_fs, acc=acc_segment, peak_min_distance=0.8)
results = run_all_steps(ppg=ppg_segment, acc=acc_segment, hr_gt=gt_segment, ppg_fs=ppg_fs, hr_fs=1)

peaks, ppg_detrended, ppg_filtered, ppg_cleaned = results['peaks'], results['ppg_detrended'], results['ppg_filtered'], results['ppg_cleaned']

plot_signals(
    [ppg_segment, ppg_detrended, ppg_filtered, ppg_cleaned],
    fs=ppg_fs,
    labels=['Raw', 'Detrended (Wavelet)', 'Bandpass filtered', 'RLS filtered'],
    title='Signal Cleanup',
    filename='signals',
    xlim=(200, 210)
)

plot_signal(ppg_cleaned, fs=ppg_fs, label='Cleaned PPG', peaks=peaks, title='Detected Peaks', filename='peaks')

plot_estimated_hr(results['hr_estimation'])