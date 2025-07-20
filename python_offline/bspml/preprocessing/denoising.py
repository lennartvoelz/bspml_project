import numpy as np
from typing import Union, Tuple, Optional
from scipy import signal


def bandpass_filter(
    ppg_signal: np.ndarray,
    sampling_rate: float,
    low_cutoff: float = 0.5,
    high_cutoff: float = 4.0,
    filter_order: int = 4,
    filter_type: str = 'butterworth'
) -> np.ndarray:
    """
    Apply band-pass filter to PPG signal (0.5-4 Hz for 30-240 BPM).

    Args:
        ppg_signal: Input PPG signal
        sampling_rate: Sampling rate in Hz
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        filter_order: Filter order
        filter_type: Filter type ('butterworth', 'chebyshev1', 'elliptic')

    Returns:
        Filtered PPG signal
    """
    if len(ppg_signal) == 0:
        raise ValueError("Input signal cannot be empty")

    if not np.isfinite(ppg_signal).all():
        raise ValueError("Input signal contains non-finite values")

    if low_cutoff >= high_cutoff:
        raise ValueError("Low cutoff must be less than high cutoff")

    if high_cutoff >= sampling_rate / 2:
        raise ValueError("High cutoff must be less than Nyquist frequency")

    # Normalize cutoff frequencies
    nyquist = sampling_rate / 2
    low_norm = low_cutoff / nyquist
    high_norm = high_cutoff / nyquist

    # Design filter based on type
    if filter_type.lower() == 'butterworth':
        b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
    elif filter_type.lower() == 'chebyshev1':
        b, a = signal.cheby1(
            filter_order, 1, [low_norm, high_norm], btype='band')
    elif filter_type.lower() == 'elliptic':
        b, a = signal.ellip(filter_order, 1, 40, [
                            low_norm, high_norm], btype='band')
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    # Apply zero-phase filtering to avoid phase distortion
    filtered_signal = signal.filtfilt(b, a, ppg_signal)

    return filtered_signal
