"""
Preprocessing Pipeline Module

Combines all preprocessing steps into a unified pipeline for PPG signal processing.
"""

import numpy as np
from typing import Optional, Dict, Any
from scipy import signal

from .detrending import wavelet_detrend
from .denoising import bandpass_filter
from .motion_artifacts import rls_filter


def resample_accelerometer_data(
    acc_signals: np.ndarray,
    original_rate: float,
    target_rate: float,
    target_length: int,
) -> np.ndarray:
    """
    Resample accelerometer data to match PPG sampling rate and length.

    Args:
        acc_signals: Accelerometer signals [n_samples, n_axes]
        original_rate: Original sampling rate
        target_rate: Target sampling rate
        target_length: Target number of samples

    Returns:
        Resampled accelerometer signals
    """
    if original_rate == target_rate and len(acc_signals) == target_length:
        return acc_signals

    # Calculate resampling ratio
    ratio = target_rate / original_rate

    # Resample each axis
    resampled_signals = np.zeros((target_length, acc_signals.shape[1]))

    for axis in range(acc_signals.shape[1]):
        # Use scipy.signal.resample for high-quality resampling
        resampled_axis = signal.resample(
            acc_signals[:, axis], int(len(acc_signals) * ratio)
        )

        # Trim or pad to exact target length
        if len(resampled_axis) >= target_length:
            resampled_signals[:, axis] = resampled_axis[:target_length]
        else:
            resampled_signals[: len(resampled_axis), axis] = resampled_axis
            # Pad with last value if needed
            if len(resampled_axis) < target_length:
                resampled_signals[len(resampled_axis) :, axis] = resampled_axis[-1]

    return resampled_signals


def preprocess_ppg(
    ppg_signal: np.ndarray,
    acc_signals: Optional[np.ndarray] = None,
    sampling_rate: float = 50.0,
    acc_sampling_rate: Optional[float] = None,
    enable_detrending: bool = True,
    enable_denoising: bool = True,
    enable_motion_removal: bool = True,
    adaptive_motion_removal: bool = True,
    motion_threshold: float = 80.0,
    detrending_method: str = "wavelet",
    denoising_method: str = "bandpass",
    motion_removal_method: str = "rls",
    preprocessing_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Complete PPG signal preprocessing pipeline.

    This function applies the complete preprocessing pipeline including:
    1. Detrending (baseline drift removal)
    2. Denoising (high-frequency noise removal)
    3. Motion artifact removal (if accelerometer data available and motion detected)

    Args:
        ppg_signal: Input PPG signal array
        acc_signals: Accelerometer signals [n_samples, n_axes] (optional)
        sampling_rate: PPG sampling rate in Hz
        acc_sampling_rate: Accelerometer sampling rate in Hz (auto-detected if None)
        enable_detrending: Whether to apply detrending
        enable_denoising: Whether to apply denoising
        enable_motion_removal: Whether to apply motion artifact removal
        adaptive_motion_removal: Whether to use adaptive motion removal based on motion level
        motion_threshold: Motion magnitude threshold for adaptive RLS (higher = less sensitive)
        detrending_method: Detrending method ('wavelet')
        denoising_method: Denoising method ('bandpass')
        motion_removal_method: Motion removal method ('rls')
        preprocessing_params: Additional parameters for preprocessing methods

    Returns:
        Preprocessed PPG signal
    """
    if len(ppg_signal) == 0:
        raise ValueError("PPG signal cannot be empty")

    if not np.isfinite(ppg_signal).all():
        raise ValueError("PPG signal contains non-finite values")

    # Initialize parameters
    if preprocessing_params is None:
        preprocessing_params = {}

    # Start with original signal
    processed_signal = ppg_signal.copy()

    if enable_motion_removal and acc_signals is not None:
        # Determine accelerometer sampling rate
        if acc_sampling_rate is None:
            acc_sampling_rate = 50.0  # Default for PPG Dalia dataset

        # Resample accelerometer data to match PPG signal
        acc_resampled = resample_accelerometer_data(
            acc_signals=acc_signals,
            original_rate=acc_sampling_rate,
            target_rate=sampling_rate,
            target_length=len(processed_signal),
        )

        # Adaptive motion removal: check if motion level warrants RLS filtering
        should_use_rls = True
        if adaptive_motion_removal:
            from .motion_artifacts import combine_accelerometer_channels

            acc_combined = combine_accelerometer_channels(acc_resampled)
            motion_magnitude = acc_combined.flatten()

            # Calculate motion statistics
            high_motion_percentage = np.mean(motion_magnitude > motion_threshold) * 100

            # Use RLS only if significant high motion is detected
            should_use_rls = high_motion_percentage > 10.0  # 10% threshold

            print(
                f"Motion analysis: {high_motion_percentage:.1f}% above threshold {motion_threshold:.1f}"
            )
            print(
                f"RLS filter: {'ENABLED' if should_use_rls else 'DISABLED'} (adaptive)"
            )

        if should_use_rls:
            if motion_removal_method == "rls":
                rls_params = preprocessing_params.get("rls", {})
                processed_signal = rls_filter(
                    processed_signal, acc_resampled, **rls_params
                )
            else:
                raise ValueError(
                    f"Unknown motion removal method: {motion_removal_method}"
                )
        else:
            print("RLS filter skipped due to low motion level")

    if enable_denoising:
        if denoising_method == "bandpass":
            bandpass_params = preprocessing_params.get("bandpass", {})
            processed_signal = bandpass_filter(
                processed_signal, sampling_rate, **bandpass_params
            )
        else:
            raise ValueError(f"Unknown denoising method: {denoising_method}")

    if enable_detrending:
        if detrending_method == "wavelet":
            wavelet_params = preprocessing_params.get("wavelet", {})
            processed_signal = wavelet_detrend(processed_signal, **wavelet_params)
        else:
            raise ValueError(f"Unknown detrending method: {detrending_method}")

    return processed_signal


def get_default_preprocessing_params() -> Dict[str, Any]:
    """
    Get default preprocessing parameters.

    Returns:
        Dictionary with default parameters for all preprocessing methods
    """
    return {
        "wavelet": {"wavelet": "db4", "levels": None, "mode": "symmetric"},
        "bandpass": {
            "low_cutoff": 0.5,
            "high_cutoff": 4.0,
            "filter_order": 4,
            "filter_type": "butterworth",
        },
        "rls": {"filter_order": 8, "forgetting_factor": 0.99},
    }
