import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def find_peaks_sliding_window(
    ppg_signal: np.ndarray,
    sampling_rate: float,
    window_size: float = 10.0,
    overlap: float = 0.3,
    min_peak_distance: float = 0.32,
    prominence: Optional[float] = 0.55,
    min_peak_height: Optional[float] = 0,
    adaptive_threshold: bool = True,
    use_envelope_method: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find PPG peaks using sliding window with local maxima detection.

    Args:
        ppg_signal: Input PPG signal
        sampling_rate: Sampling rate in Hz
        window_size: Window size in seconds
        overlap: Window overlap ratio (0-1)
        min_peak_distance: Minimum distance between peaks in seconds
        min_peak_height: Minimum peak height (default: adaptive)
        adaptive_threshold: Whether to use adaptive thresholding
        use_envelope_method: If True, use envelope method instead of detect_peaks_in_window

    Returns:
        Tuple of (peak_indices, peak_values)
    """
    if len(ppg_signal) == 0:
        raise ValueError("PPG signal cannot be empty")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")

    # Convert time parameters to samples
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    min_distance_samples = int(min_peak_distance * sampling_rate)

    # Initialize peak storage
    all_peaks = []
    all_peak_values = []

    # Process signal in sliding windows
    for start_idx in range(0, len(ppg_signal) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        window_signal = ppg_signal[start_idx:end_idx]

        # Find peaks in current window using selected method
        if use_envelope_method:
            window_peaks = detect_peaks_envelope_in_window(
                window_signal,
                sampling_rate,
                min_peak_distance
            )
        else:
            window_peaks = detect_peaks_in_window(
                window_signal,
                min_distance_samples,
                prominence,
                adaptive_threshold,
                min_peak_height
            )

        # Convert local indices to global indices
        global_peaks = window_peaks + start_idx

        # Store peaks and their values
        for peak_idx in global_peaks:
            if peak_idx < len(ppg_signal):
                all_peaks.append(peak_idx)
                all_peak_values.append(ppg_signal[peak_idx])

    # Remove duplicate peaks from overlapping windows
    if all_peaks:
        peaks_array = np.array(all_peaks)
        values_array = np.array(all_peak_values)

        # Sort by peak index
        sort_indices = np.argsort(peaks_array)
        peaks_array = peaks_array[sort_indices]
        values_array = values_array[sort_indices]

        # Remove duplicates within minimum distance
        unique_peaks, unique_values = remove_duplicate_peaks(
            peaks_array, values_array, min_distance_samples
        )

        return unique_peaks, unique_values
    else:
        return np.array([]), np.array([])


def detect_peaks_in_window(
    window_signal: np.ndarray,
    min_distance: int,
    prominence: float,
    adaptive_threshold: bool = True,
    min_height: Optional[float] = None
) -> np.ndarray:
    """
    Detect peaks within a single window using local maxima.

    Args:
        window_signal: Signal window
        min_distance: Minimum distance between peaks in samples
        min_height: Minimum peak height
        adaptive_threshold: Whether to use adaptive thresholding

    Returns:
        Array of peak indices within the window
    """
    if len(window_signal) < 3:
        return np.array([])

    # Calculate adaptive threshold if requested
    if adaptive_threshold:
        if prominence is None:
            prominence = 0.5 * np.std(window_signal)

    # Find peaks using scipy
    peaks, properties = signal.find_peaks(
        window_signal,
        prominence=prominence,
        distance=min_distance,
        height=min_height
    )

    return peaks


def detect_peaks_envelope_in_window(
    window_signal: np.ndarray,
    sampling_rate: float,
    min_peak_distance: float = 0.5,
    max_peak_distance: float = 2.0,
    prominence_factor: float = 0.5
) -> np.ndarray:
    """
    Detect peaks within a single window using envelope method.
    Uses the same logic as find_peaks_envelope but applied to a window.

    Args:
        window_signal: Signal window
        sampling_rate: Sampling rate in Hz
        min_peak_distance: Minimum distance between peaks in seconds
        max_peak_distance: Maximum distance between peaks in seconds
        prominence_factor: Factor for prominence calculation

    Returns:
        Array of peak indices within the window
    """
    if len(window_signal) < 3:
        return np.array([])

    analytic_signal = signal.hilbert(window_signal)
    envelope = np.abs(analytic_signal)

    win = int(0.1 * sampling_rate)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    env_smooth = np.convolve(envelope, kernel, mode="same")

    # Mask negative values and calculate prominence
    env_masked = env_smooth.copy()
    env_masked[window_signal <= 0] = 0
    prom = np.std(env_masked) * prominence_factor

    # Find peaks
    min_dist = int(min_peak_distance * sampling_rate)
    peaks, _ = signal.find_peaks(
        env_masked, distance=min_dist, prominence=prom)

    peak_times = peaks/sampling_rate
    ibi = np.diff(peak_times)
    valid = np.logical_and(ibi >= min_peak_distance, ibi <= max_peak_distance)
    filtered_peaks = [peaks[0]]
    for idx, ok in enumerate(valid, start=1):
        if ok:
            filtered_peaks.append(peaks[idx])

    return np.array(filtered_peaks)


def detect_ppg_peaks(
    ppg_signal: np.ndarray,
    sampling_rate: float,
    method: str = 'sliding_window',
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect PPG peaks using specified method.

    Args:
        ppg_signal: Input PPG signal
        sampling_rate: Sampling rate in Hz
        method: Peak detection method ('sliding_window', 'envelope')
        **kwargs: Additional parameters for the chosen method

    Returns:
        Tuple of (peak_indices, detection_info)
    """
    if method == 'sliding_window':
        peaks, values = find_peaks_sliding_window(
            ppg_signal, sampling_rate, **kwargs)
        detection_info = {
            'method': method,
            'num_peaks': len(peaks),
            'peak_values': values,
            'parameters': kwargs
        }
    else:
        raise ValueError(f"Unknown peak detection method: {method}")

    return peaks, detection_info


def remove_duplicate_peaks(
    peaks: np.ndarray,
    values: np.ndarray,
    min_distance: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate peaks that are too close together.

    When peaks are within minimum distance, keep the one with higher value.

    Args:
        peaks: Array of peak indices
        values: Array of peak values
        min_distance: Minimum distance between peaks

    Returns:
        Tuple of (unique_peaks, unique_values)
    """
    if len(peaks) <= 1:
        return peaks, values

    unique_peaks = []
    unique_values = []

    i = 0
    while i < len(peaks):
        current_peak = peaks[i]
        current_value = values[i]

        # Find all peaks within minimum distance
        j = i + 1
        while j < len(peaks) and peaks[j] - current_peak < min_distance:
            # Keep the peak with higher value
            if values[j] > current_value:
                current_peak = peaks[j]
                current_value = values[j]
            j += 1

        unique_peaks.append(current_peak)
        unique_values.append(current_value)

        i = j

    return np.array(unique_peaks), np.array(unique_values)


def find_peaks_envelope(
        ppg_signal: np.ndarray,
        sampling_rate: float,
        min_peak_distance: float = 0.4,
        max_peak_distance: float = 1.5,
        prominence_factor: float = 0.4):

    analytic_signal = signal.hilbert(ppg_signal)
    envelope = np.abs(analytic_signal)

    win = int(0.1 * sampling_rate)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    env_smooth = np.convolve(envelope, kernel, mode="same")

    env_masked = env_smooth.copy()
    env_masked[ppg_signal <= 0] = 0
    prom = np.std(env_masked) * prominence_factor

    min_dist = int(min_peak_distance * sampling_rate)
    peaks, props = signal.find_peaks(
        env_smooth, distance=min_dist, prominence=prom)

    peak_times = peaks/sampling_rate

    ibi = np.diff(peak_times)
    valid = np.logical_and(ibi >= min_peak_distance, ibi <= max_peak_distance)
    filtered_peaks = [peaks[0]]
    for idx, ok in enumerate(valid, start=1):
        if ok:
            filtered_peaks.append(peaks[idx])

    return np.array(filtered_peaks), {"method": "envelope"}
