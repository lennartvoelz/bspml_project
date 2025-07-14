import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if installed
import matplotlib.pyplot as plt

def plot_signal(signal, fs=1.0, label="Signal", title="Signal Plot",
                filename=None, peaks=None, xlim=None, ylim=None, plot_size=(12, 4)):
    # Determine the plotting range
    total_len = len(signal)
    if xlim is not None:
        start_idx = max(0, int(xlim[0] * fs))
        end_idx = min(total_len, int(xlim[1] * fs))
    else:
        start_idx, end_idx = 0, total_len

    times = np.arange(start_idx, end_idx) / fs
    y_vals = signal[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=plot_size)
    ax.plot(times, y_vals, label=label)

    # Plot peaks if provided and in range
    if peaks is not None:
        # Convert time-based peaks to indices for accessing signal
        peak_indices = np.asarray(peaks * fs, dtype=int)
        in_range = (peak_indices >= start_idx) & (peak_indices < end_idx)
        visible_indices = peak_indices[in_range]
        peak_times = visible_indices / fs
        ax.plot(peak_times, signal[visible_indices], "rx", label="Peaks")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # Auto set y-limits unless provided
    if ylim is not None:
        ax.set_ylim(ylim)
    elif len(y_vals) > 0:
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

    # Set x-limits
    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()

    if filename:
        plt.savefig(f"../plots/{filename}.svg", dpi=150)
    plt.show()
    plt.close(fig)


def plot_signals(signals, fs=1.0, labels=None, title="Overlay Plot",
                 filename=None, xlim=None, ylim=None, hr_overlay=None, plot_size=(12, 4)):
    if not isinstance(signals, list):
        signals = [signals]
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(len(signals))]

    if signals:
        total_len = max(len(sig) for sig in signals)
        if xlim is not None:
            start_idx = max(0, int(xlim[0] * fs))
            end_idx = min(total_len, int(xlim[1] * fs))
        else:
            start_idx, end_idx = 0, total_len

    fig, ax = plt.subplots(figsize=plot_size)

    y_min = float('inf')
    y_max = float('-inf')

    if signals:
        for sig, label in zip(signals, labels):
            clipped = sig[start_idx:end_idx]
            times = np.arange(start_idx, start_idx + len(clipped)) / fs
            ax.plot(times, clipped, label=label)

            if len(clipped) > 0:
                y_min = min(y_min, np.min(clipped))
                y_max = max(y_max, np.max(clipped))

    # HR overlay
    if hr_overlay:
        est_times, est_hr, gt_times, gt_hr = hr_overlay
        ax.plot(est_times, est_hr, label="Estimated HR (PPG)", color='red', linestyle='--', marker='o')
        ax.plot(gt_times, gt_hr, label="Ground Truth HR (ECG)", color='black', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / HR")
    ax.legend()
    ax.grid(True)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif y_min < y_max:
        margin = 0.1 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)

    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()

    if filename:
        plt.savefig(f"../plots/{filename}.svg", dpi=150)
    fig.show()
    plt.close(fig)
