"""
Signal Preprocessing Module

This module contains algorithms for PPG signal preprocessing including:
- Detrending using wavelet-based reconstruction
- Denoising with band-pass filtering (0.5-4 Hz)
- Motion artifact removal using RLS filter with accelerometer reference
"""

from .detrending import wavelet_detrend
from .denoising import bandpass_filter
from .motion_artifacts import rls_filter, combine_accelerometer_channels
from .pipeline import preprocess_ppg, resample_accelerometer_data

__all__ = [
    "wavelet_detrend",
    "bandpass_filter",
    "rls_filter",
    "combine_accelerometer_channels",
    "preprocess_ppg",
    "resample_accelerometer_data"
]
