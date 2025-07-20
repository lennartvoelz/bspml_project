"""
Heart Rate Estimation Module

This module contains algorithms for extracting instantaneous heart rate
from cleaned PPG signals using local maxima detection in sliding windows.
"""

from .peak_detection import find_peaks_sliding_window, detect_ppg_peaks
from .hr_calculation import calculate_instantaneous_hr
from .pipeline import estimate_heart_rate

__all__ = [
    "find_peaks_sliding_window",
    "detect_ppg_peaks",
    "calculate_instantaneous_hr",
    "estimate_heart_rate"
]
