import numpy as np
from typing import Dict, Any, Optional
from scipy import interpolate, signal


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.

    Args:
        signal: Clean signal
        noise: Noise signal

    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return float("inf")

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_delta_snr(
    ppg_raw: np.ndarray,
    ppg_processed: np.ndarray,
    ground_truth_signal: Optional[np.ndarray] = None,
    sampling_rate: float = 50.0,
) -> float:
    """
    Calculate ΔSNR (improvement in SNR after preprocessing).

    Args:
        ppg_raw: Raw PPG signal
        ppg_processed: Processed PPG signal
        ground_truth_signal: Ground truth clean signal (optional)

    Returns:
        ΔSNR in dB (positive values indicate improvement)
    """
    if ground_truth_signal is not None:
        # If ground truth is available, use it as reference
        noise_raw = ppg_raw - ground_truth_signal
        noise_processed = ppg_processed - ground_truth_signal

        snr_raw = calculate_snr(ground_truth_signal, noise_raw)
        snr_processed = calculate_snr(ground_truth_signal, noise_processed)
    else:
        # Estimate noise as high-frequency components
        # Use high-pass filter to extract noise
        fs = sampling_rate
        nyquist = fs / 2
        high_cutoff = 8.0

        if high_cutoff < nyquist:
            b, a = signal.butter(4, high_cutoff / nyquist, btype="high")

            noise_raw = signal.filtfilt(b, a, ppg_raw)
            noise_processed = signal.filtfilt(b, a, ppg_processed)

            # Signal is low-frequency component
            b_low, a_low = signal.butter(4, high_cutoff / nyquist, btype="low")
            signal_raw = signal.filtfilt(b_low, a_low, ppg_raw)
            signal_processed = signal.filtfilt(b_low, a_low, ppg_processed)

            snr_raw = calculate_snr(signal_raw, noise_raw)
            snr_processed = calculate_snr(signal_processed, noise_processed)
        else:
            # Fallback: use signal variance as proxy
            snr_raw = 10 * np.log10(np.var(ppg_raw) / (np.var(ppg_raw) * 0.1))
            snr_processed = 10 * np.log10(
                np.var(ppg_processed) / (np.var(ppg_processed) * 0.1)
            )

    delta_snr = snr_processed - snr_raw
    return delta_snr


def calculate_hr_error(
    estimated_hr: np.ndarray,
    ground_truth_hr: Optional[np.ndarray],
    estimated_time: np.ndarray,
    ground_truth_time: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    Calculate heart rate estimation error metrics.

    Args:
        estimated_hr: Estimated heart rate values (BPM)
        ground_truth_hr: Ground truth heart rate values (BPM) - can be None
        estimated_time: Time points for estimated HR
        ground_truth_time: Time points for ground truth HR - can be None

    Returns:
        Dictionary containing error metrics (or error message if no ground truth)
    """
    if ground_truth_hr is None or ground_truth_time is None:
        return {
            "error": "No ground truth available",
            "mean_absolute_error": float("nan"),
            "correlation": float("nan"),
        }
    # Interpolate to common time grid
    min_time = max(estimated_time.min(), ground_truth_time.min())
    max_time = min(estimated_time.max(), ground_truth_time.max())

    if min_time >= max_time:
        return {
            "error": "No overlapping time range",
            "mean_absolute_error": float("nan"),
            "correlation": float("nan"),
        }

    # Create common time grid
    common_time = np.linspace(
        min_time, max_time, int((max_time - min_time) * 2)
    )  # 2 Hz

    # Interpolate both signals to common grid
    try:
        f_est = interpolate.interp1d(
            estimated_time,
            estimated_hr,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        f_gt = interpolate.interp1d(
            ground_truth_time,
            ground_truth_hr,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        hr_est_interp = f_est(common_time)
        hr_gt_interp = f_gt(common_time)

        # Remove NaN values
        valid_mask = ~(np.isnan(hr_est_interp) | np.isnan(hr_gt_interp))
        hr_est_clean = hr_est_interp[valid_mask]
        hr_gt_clean = hr_gt_interp[valid_mask]

        if len(hr_est_clean) == 0:
            return {
                "error": "No valid data points after interpolation",
                "mean_absolute_error": float("nan"),
                "correlation": float("nan"),
            }

        absolute_errors = np.abs(hr_est_clean - hr_gt_clean)
        mae = np.mean(absolute_errors)

        # Calculate correlation
        if len(hr_est_clean) > 1:
            correlation = np.corrcoef(hr_est_clean, hr_gt_clean)[0, 1]
        else:
            correlation = float("nan")

        # Calculate root mean square error (RMSE)
        rmse = np.sqrt(np.mean(absolute_errors**2))

        return {
            "mean_absolute_error": mae,
            "rmse": rmse,
            "correlation": correlation,
            "num_points": len(hr_est_clean),
            "time_range": (min_time, max_time),
        }

    except Exception as e:
        return {
            "error": f"Interpolation failed: {str(e)}",
            "mean_absolute_error": float("nan"),
            "correlation": float("nan"),
        }


def evaluate_pipeline_performance(
    ppg_raw: np.ndarray,
    ppg_processed: np.ndarray,
    hr_results: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]],
    sampling_rate: float = 50.0,
) -> Dict[str, Any]:
    """
    Evaluation of pipeline performance.

    Args:
        ppg_raw: Raw PPG signal
        ppg_processed: Processed PPG signal
        hr_results: Heart rate estimation results
        ground_truth: Ground truth data (can be None for real-world data)
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary containing all evaluation metrics
    """
    evaluation = {
        "timestamp": str(np.datetime64("now")),
        "sampling_rate": sampling_rate,
    }

    # Calculate ΔSNR
    try:
        delta_snr = calculate_delta_snr(ppg_raw, ppg_processed)
        evaluation["delta_snr_db"] = delta_snr
        evaluation["snr_improvement"] = "Improved" if delta_snr > 0 else "Degraded"
    except Exception as e:
        evaluation["delta_snr_db"] = float("nan")
        evaluation["delta_snr_error"] = str(e)

    # Calculate HR error metrics
    if (
        hr_results.get("success", False)
        and ground_truth is not None
        and ground_truth.get("heart_rate") is not None
        and "time_points" in hr_results
        and "hr_values" in hr_results
    ):
        try:
            gt_hr = ground_truth["heart_rate"]
            hr_error_metrics = calculate_hr_error(
                estimated_hr=hr_results["hr_values"],
                ground_truth_hr=gt_hr["values"],
                estimated_time=hr_results["time_points"],
                ground_truth_time=gt_hr["time_axis"],
            )
            evaluation["hr_error_metrics"] = hr_error_metrics

        except Exception as e:
            evaluation["hr_error_metrics"] = {"error": str(e)}
    else:
        evaluation["hr_error_metrics"] = {"error": "No ground truth available"}

    try:
        # Signal statistics
        evaluation["signal_quality"] = {
            "raw_signal_std": float(np.std(ppg_raw)),
            "processed_signal_std": float(np.std(ppg_processed)),
            "raw_signal_range": float(np.ptp(ppg_raw)),
            "processed_signal_range": float(np.ptp(ppg_processed)),
            "noise_reduction_ratio": float(np.std(ppg_raw) / np.std(ppg_processed))
            if np.std(ppg_processed) > 0
            else float("inf"),
        }
    except Exception as e:
        evaluation["signal_quality"] = {"error": str(e)}

    # Peak detection quality
    if hr_results.get("success", False):
        evaluation["peak_detection"] = {
            "num_peaks": hr_results.get("num_peaks", 0),
            "peak_detection_success": True,
            "average_peak_interval": hr_results.get("statistics", {}).get(
                "mean_ibi", float("nan")
            ),
        }
    else:
        evaluation["peak_detection"] = {
            "num_peaks": 0,
            "peak_detection_success": False,
            "error": hr_results.get("error", "Unknown error"),
        }

    return evaluation


def print_evaluation_summary(evaluation: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # ΔSNR
    if "delta_snr_db" in evaluation and not np.isnan(evaluation["delta_snr_db"]):
        delta_snr = evaluation["delta_snr_db"]
        print(
            f"ΔSNR: {delta_snr:.2f} dB ({evaluation.get('snr_improvement', 'Unknown')})"
        )
    else:
        print("ΔSNR: Could not calculate")

    # HR Error
    hr_metrics = evaluation.get("hr_error_metrics", {})
    if "error" in hr_metrics:
        print(f"\nHeart Rate Error: {hr_metrics['error']}")

    # Signal Quality
    sq = evaluation.get("signal_quality", {})
    if "error" not in sq:
        print(f"\nSignal Quality:")
        print(f"  Noise Reduction Ratio: {sq.get('noise_reduction_ratio', 'N/A'):.2f}")
        print(f"  Raw Signal Std: {sq.get('raw_signal_std', 'N/A'):.4f}")
        print(f"  Processed Signal Std: {sq.get('processed_signal_std', 'N/A'):.4f}")

    # Peak Detection
    pd = evaluation.get("peak_detection", {})
    print(f"\nPeak Detection:")
    print(f"  Success: {pd.get('peak_detection_success', False)}")
    print(f"  Number of Peaks: {pd.get('num_peaks', 0)}")

    print("=" * 60)
