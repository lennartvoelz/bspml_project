import numpy as np
from typing import Tuple, Optional, Dict
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)


def calculate_instantaneous_hr(
    peak_indices: np.ndarray,
    sampling_rate: float,
    interpolation_rate: Optional[float] = None,
    method: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate instantaneous heart rate from detected peaks.

    Args:
        peak_indices: Array of peak indices
        sampling_rate: Signal sampling rate in Hz
        interpolation_rate: Target interpolation rate in Hz (optional)
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        Tuple of (time_points, hr_values) in seconds and BPM
    """
    if len(peak_indices) < 2:
        raise ValueError("Need at least 2 peaks to calculate heart rate")

    # Calculate inter-beat intervals (IBI) in seconds
    ibi_samples = np.diff(peak_indices)
    ibi_seconds = ibi_samples / sampling_rate

    # Convert IBI to heart rate in BPM
    hr_bpm = 60.0 / ibi_seconds

    # Time points for heart rate values (at peak locations)
    peak_times = peak_indices[1:] / sampling_rate  # Skip first peak

    # Filter physiologically plausible heart rates
    valid_mask = (hr_bpm >= 30) & (hr_bpm <= 200)
    valid_times = peak_times[valid_mask]
    valid_hr = hr_bpm[valid_mask]

    if len(valid_hr) == 0:
        raise ValueError("No valid heart rate values found")

    # Interpolate to regular time grid if requested
    if interpolation_rate is not None:
        time_start = valid_times[0]
        time_end = valid_times[-1]

        # Create regular time grid
        regular_times = np.arange(time_start, time_end, 1.0 / interpolation_rate)

        if len(regular_times) > 1:
            # Interpolate heart rate values
            # interp_func = interpolate.interp1d(
            #     valid_times,
            #     valid_hr,
            #     kind="linear",
            #     bounds_error=False,
            #     fill_value="extrapolate",
            # )

            interp_func = interpolate.PchipInterpolator(
                x=valid_times,
                y=valid_hr,
                extrapolate=False
            )

            interpolated_hr = interp_func(regular_times)

            # Ensure interpolated values are within reasonable range
            interpolated_hr = np.clip(interpolated_hr, 30, 200)

            return regular_times, interpolated_hr

    return valid_times, valid_hr
