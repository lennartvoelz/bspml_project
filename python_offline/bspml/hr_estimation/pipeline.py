import numpy as np
from typing import Optional, Dict, Any

from .peak_detection import detect_ppg_peaks
from .hr_calculation import calculate_instantaneous_hr


def estimate_heart_rate(
    ppg_signal: np.ndarray,
    sampling_rate: float,
    peak_detection_method: str = 'sliding_window',
    interpolation_rate: Optional[float] = 1.0,
    return_peaks: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete heart rate estimation pipeline from preprocessed PPG signal.

    Args:
        ppg_signal: Preprocessed PPG signal
        sampling_rate: Sampling rate in Hz
        peak_detection_method: Peak detection method
        interpolation_rate: Target interpolation rate in Hz
        return_peaks: Whether to include peak information
        return_statistics: Whether to calculate HR statistics
        **kwargs: Additional parameters for peak detection

    Returns:
        Dictionary containing heart rate estimation results
    """
    if len(ppg_signal) == 0:
        raise ValueError("PPG signal cannot be empty")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")

    try:
        peaks, peak_info = detect_ppg_peaks(
            ppg_signal,
            sampling_rate,
            method=peak_detection_method,
            **kwargs
        )

        if len(peaks) < 2:
            return {
                'success': False,
                'error': 'insufficient_peaks',
                'num_peaks': len(peaks),
                'signal_length_seconds': len(ppg_signal) / sampling_rate
            }

    except Exception as e:
        return {
            'success': False,
            'error': 'peak_detection_failed',
            'error_message': str(e)
        }

    try:
        time_points, hr_values = calculate_instantaneous_hr(
            peaks,
            sampling_rate,
            interpolation_rate=interpolation_rate,
            method='linear'
        )

    except Exception as e:
        return {
            'success': False,
            'error': 'hr_calculation_failed',
            'error_message': str(e),
            'num_peaks': len(peaks)
        }

    from scipy import signal


    fs = 2.0
    fc = 0.33
    b, a = signal.butter(2, fc/(fs/2))
    hr_lp = signal.filtfilt(b, a, hr_values)

    alpha = 0.2
    hr_values = signal.lfilter([alpha], [1, alpha-1], hr_lp)

    hr_values_smoothed = hr_values.copy()

    # Calculate statistics
    statistics = {
        'mean': float(np.mean(hr_values)),
        'std': float(np.std(hr_values)),
        'min': float(np.min(hr_values)),
        'max': float(np.max(hr_values)),
        'median': float(np.median(hr_values))
    }

    # Prepare results
    results = {
        'success': True,
        'num_peaks': len(peaks),
        'time_points': time_points,
        'hr_values': hr_values,
        'hr_values_smoothed': hr_values_smoothed,
        'statistics': statistics,
        'sampling_rate': sampling_rate,
        'signal_length_seconds': len(ppg_signal) / sampling_rate,
        'estimation_parameters': {
            'peak_detection_method': peak_detection_method,
            'interpolation_rate': interpolation_rate,
            **kwargs
        }
    }

    if return_peaks:
        results['peaks'] = {
            'indices': peaks,
            'values': ppg_signal[peaks],
            'times': peaks / sampling_rate,
            'detection_info': peak_info
        }

    return results
