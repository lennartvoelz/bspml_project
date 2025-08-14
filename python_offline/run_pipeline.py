import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Import BSPML modules
from bspml import (
    load_ppg_dalia_data,
    load_accelerometer_data,
    load_ground_truth_data,
    load_data_auto,
    get_available_real_world_files,
    preprocess_ppg,
    estimate_heart_rate,
    get_available_subjects,
    evaluate_pipeline_performance,
    print_evaluation_summary,
    load_config,
    get_dataset_config,
)


def create_output_directory(base_path: str = "results") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"pipeline_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results_to_json(results: Dict[str, Any], output_path: str):
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    json_results[key][sub_key] = sub_value.tolist()
                else:
                    json_results[key][sub_key] = sub_value
        else:
            json_results[key] = value

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)


def plot_pipeline_results(
    ppg_raw: np.ndarray,
    ppg_processed: np.ndarray,
    acc_signals: np.ndarray,
    hr_results: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]],
    sampling_rate: float,
    acc_sampling_rate: float,
    output_dir: str,
    subject_id: str,
):
    """Create visualization of pipeline results."""

    # Create time axis for PPG signals
    time_axis = np.arange(len(ppg_raw)) / sampling_rate

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(
        f"PPG Processing Pipeline Results - Subject {subject_id}", fontsize=16)

    # Plot 1: Raw vs Processed PPG
    axes[0].plot(time_axis, ppg_raw, "b-", alpha=0.7,
                 label="Raw PPG", linewidth=0.8)
    axes[0].plot(time_axis, ppg_processed, "r-",
                 label="Processed PPG", linewidth=1.0)
    axes[0].set_ylabel("PPG Amplitude")
    axes[0].set_title("Raw vs Processed PPG Signal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accelerometer signals
    from bspml.preprocessing import (
        combine_accelerometer_channels,
        resample_accelerometer_data,
    )

    # Resample and combine accelerometer for visualization
    acc_resampled_viz = resample_accelerometer_data(
        acc_signals, acc_sampling_rate, sampling_rate, len(ppg_raw)
    )
    acc_combined_viz = combine_accelerometer_channels(acc_resampled_viz)
    acc_magnitude = acc_combined_viz.flatten()

    acc_time = np.arange(len(acc_magnitude)) / sampling_rate
    axes[1].plot(
        acc_time,
        acc_magnitude,
        "purple",
        linewidth=2,
        label="ACC Magnitude (Combined)",
        alpha=0.8,
    )
    axes[1].set_ylabel("Acceleration Magnitude")
    axes[1].set_title("Combined Accelerometer Signal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Detected peaks on processed signal
    if hr_results.get("success", False):
        # Show a portion of the signal with detected peaks
        window_start = int(40 * sampling_rate)
        window_end = int(80 * sampling_rate)

        window_time = time_axis[window_start:window_end]
        window_signal = ppg_processed[window_start:window_end]

        axes[2].plot(window_time, window_signal, "b-", linewidth=1.0)

        # Mark detected peaks if available
        if "peaks" in hr_results and isinstance(hr_results["peaks"], dict):
            peak_indices = hr_results["peaks"].get("indices", np.array([]))
            if len(peak_indices) > 0:
                peak_mask = (peak_indices >= window_start) & (
                    peak_indices < window_end)
                window_peaks = peak_indices[peak_mask] - window_start
                if len(window_peaks) > 0:
                    axes[2].plot(
                        window_time[window_peaks],
                        window_signal[window_peaks],
                        "ro",
                        markersize=6,
                        label="Detected Peaks",
                    )

        axes[2].set_ylabel("PPG Amplitude")
        axes[2].set_title("Peak Detection (30s window)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    # Plot 4: Heart rate comparison
    if hr_results.get("success", False) and "time_points" in hr_results:
        hr_time = hr_results["time_points"]
        hr_values = hr_results["hr_values"]
        axes[3].plot(hr_time, hr_values, "r-",
                     linewidth=2, label="Estimated HR")

        # Plot ground truth if available
        if ground_truth is not None and ground_truth.get("heart_rate") is not None:
            gt_hr = ground_truth["heart_rate"]
            gt_time = gt_hr["time_axis"]
            gt_values = gt_hr["values"]
            axes[3].plot(gt_time, gt_values, "g-",
                         linewidth=2, label="Ground Truth HR")

        axes[3].set_ylabel("Heart Rate (BPM)")
        axes[3].set_xlabel("Time (seconds)")
        axes[3].set_title("Heart Rate Estimation vs Ground Truth")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(40, 175)  # Reasonable HR range

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"pipeline_results_{subject_id}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    plt.show()

    print(f"Plot saved to: {plot_path}")


def run_pipeline(
    data_path: str,
    data_identifier: str,
    duration: Optional[float] = 600.0,  # Process 600 seconds by default
    start_time: float = 100,
    output_dir: Optional[str] = None,
    auto_detect: bool = True,
    dataset_type: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete PPG processing pipeline.

    Args:
        data_identifier: Subject ID for PPG Dalia or file path for real-world data
        duration: Duration in seconds to process
        start_time: Start time offset in seconds
        data_path: Path to dataset directory
        output_dir: Output directory (auto-created if None)
        auto_detect: Whether to automatically detect data type
        dataset_type: Dataset type ('ppgDalia' or 'real') - overrides auto-detection
        config_path: Path to configuration file (uses default if None)

    Returns:
        Dictionary containing all results
    """
    print(f"Starting PPG processing pipeline for data: {data_identifier}")
    print(f"Duration: {duration}s, Start time: {start_time}s")
    print("-" * 50)

    # Load configuration
    config = load_config(config_path)

    if output_dir is None:
        output_dir = create_output_directory()

    results = {
        "data_identifier": data_identifier,
        "duration": duration,
        "start_time": start_time,
        "timestamp": datetime.now().isoformat(),
        "auto_detect": auto_detect,
        "success": True,
    }

    try:
        if auto_detect:
            print("1. Auto-detecting data type and loading data...")
            ppg_signal, acc_signals, ppg_sampling_rate, acc_sampling_rate, metadata, ground_truth = (
                load_data_auto(
                    data_identifier=data_identifier,
                    duration=duration,
                    start_time=start_time,
                    data_path=data_path,
                )
            )

            data_type = metadata.get("data_source", "Unknown")
            print(f"   Detected data type: {data_type}")

            print(
                f"   Loaded {len(ppg_signal)} PPG samples at {ppg_sampling_rate} Hz")
            if acc_signals is not None:
                print(
                    f"   Loaded {len(acc_signals)} ACC samples at {acc_sampling_rate} Hz"
                )

            results["data_type"] = data_type
            results["metadata"] = metadata

        else:
            print("1. Loading PPG data (PPG Dalia format)...")
            ppg_signal, ppg_sampling_rate, ppg_metadata = load_ppg_dalia_data(
                subject_id=data_identifier,
                duration=duration,
                start_time=start_time,
                data_path=data_path,
            )
            print(
                f"   Loaded {len(ppg_signal)} PPG samples at {ppg_sampling_rate} Hz")

            print("2. Loading accelerometer data...")
            acc_signals, acc_sampling_rate, acc_metadata = load_accelerometer_data(
                subject_id=data_identifier,
                duration=duration,
                start_time=start_time,
                data_path=data_path,
            )
            print(
                f"   Loaded {len(acc_signals)} ACC samples at {acc_sampling_rate} Hz")

            print("3. Loading ground truth data...")
            ground_truth = load_ground_truth_data(
                subject_id=data_identifier,
                start_time=start_time,
                duration=duration,
                data_path=data_path,
            )

            results["ppg_metadata"] = ppg_metadata
            results["acc_metadata"] = acc_metadata
            results["data_type"] = "PPG_Dalia"

        # Display ground truth info
        if ground_truth is not None and ground_truth.get("heart_rate") is not None:
            gt_hr = ground_truth["heart_rate"]
            print(
                f"   Ground truth HR: {gt_hr['mean']:.1f} ± {gt_hr['std']:.1f} BPM")
        else:
            print("   No ground truth HR data available")
        results["ground_truth"] = ground_truth

        # Determine dataset type for configuration
        if dataset_type is not None:
            # Use explicitly provided dataset type
            config_dataset_type = dataset_type
        elif auto_detect:
            # Map detected data type to config key
            detected_type = results.get("data_type", "Unknown")
            if "PPG_Dalia" in detected_type or "PPG Dalia" in detected_type:
                config_dataset_type = "ppgDalia"
            elif "Real" in detected_type or "real" in detected_type:
                config_dataset_type = "real"
            else:
                print(
                    f"Warning: Unknown data type '{detected_type}', using ppgDalia config")
                config_dataset_type = "ppgDalia"
        else:
            # Default to ppgDalia when not auto-detecting
            config_dataset_type = "ppgDalia"

        # Get dataset-specific configuration
        dataset_config = get_dataset_config(config, config_dataset_type)
        print(f"   Using configuration for: {config_dataset_type}")
        results["config_dataset_type"] = config_dataset_type
        results["config"] = dataset_config

        print("2. Preprocessing PPG signal...")
        print("   - Wavelet detrending")
        print("   - Bandpass filtering (0.5-4 Hz)")
        print("   - RLS motion artifact removal")

        ppg_processed = preprocess_ppg(
            ppg_signal=ppg_signal,
            acc_signals=acc_signals,
            sampling_rate=ppg_sampling_rate,
            acc_sampling_rate=acc_sampling_rate,
            enable_detrending=dataset_config["enable_detrending"],
            enable_denoising=dataset_config["enable_denoising"],
            enable_motion_removal=dataset_config["enable_motion_removal"],
            adaptive_motion_removal=dataset_config["adaptive_motion_removal"],
            motion_threshold=dataset_config["motion_threshold"],
            preprocessing_params={
                "rls": {
                    "filter_order": dataset_config["filter_order"],
                    "forgetting_factor": dataset_config["forgetting_factor"],
                    "delay_compensation": dataset_config["delay_compensation"],
                    "auto_delay_detection": dataset_config["auto_delay_detection"],
                },
                "bandpass": {
                    "low_cutoff": dataset_config["low_cutoff"],
                    "high_cutoff": dataset_config["high_cutoff"],
                    "filter_order": dataset_config["bandpass_filter_order"],
                    "filter_type": dataset_config["filter_type"],
                },
                "wavelet": {
                    "wavelet": dataset_config["wavelet"],
                    "levels": dataset_config["levels"],
                    "mode": dataset_config["mode"],
                },
            },
        )
        print(f"   Preprocessing completed")

        print("3. Estimating heart rate...")

        # Prepare HR estimation parameters based on detection method
        hr_params = {
            "window_size": dataset_config["window_size"],
            "overlap": dataset_config["overlap"],
            "min_peak_distance": dataset_config["min_peak_distance"],
            "prominence": dataset_config["prominence"],
            "min_peak_height": dataset_config["min_peak_height"],
            "adaptive_threshold": dataset_config["adaptive_threshold"],
            "use_envelope_method": dataset_config["use_envelope_method"],
        }

        # Add envelope-specific parameters only if using envelope method
        if dataset_config["use_envelope_method"]:
            hr_params.update({
                "max_peak_distance": dataset_config["max_peak_distance"],
                "prominence_factor": dataset_config["prominence_factor"],
            })

        hr_results = estimate_heart_rate(
            ppg_signal=ppg_processed,
            sampling_rate=ppg_sampling_rate,
            peak_detection_method=dataset_config["peak_detection_method"],
            interpolation_rate=dataset_config["interpolation_rate"],
            return_peaks=dataset_config["return_peaks"],
            **hr_params
        )

        # Adjust time axis to account for start_time offset
        if hr_results.get("success", False) and "time_points" in hr_results:
            hr_results["time_points"] = hr_results["time_points"] + start_time
            if "peaks" in hr_results and isinstance(hr_results["peaks"], dict):
                hr_results["peaks"]["times"] = hr_results["peaks"]["times"] + start_time

        if hr_results.get("success", False):
            hr_stats = hr_results.get("statistics", {})
            mean_hr = hr_stats.get("mean", None)
            std_hr = hr_stats.get("std", None)

            if mean_hr is not None and std_hr is not None:
                print(f"   Estimated HR: {mean_hr:.1f} ± {std_hr:.1f} BPM")
            else:
                print("   Estimated HR: Statistics not available")
            print(f"   Detected {hr_results.get('num_peaks', 0)} peaks")
        else:
            print(
                f"   Heart rate estimation failed: {hr_results.get('error', 'Unknown error')}"
            )

        results["hr_estimation"] = hr_results

        print("4. Creating visualizations...")
        # Extract filename for output
        if os.path.isfile(data_identifier):
            output_name = os.path.splitext(
                os.path.basename(data_identifier))[0]
        else:
            output_name = data_identifier

        plot_pipeline_results(
            ppg_raw=ppg_signal,
            ppg_processed=ppg_processed,
            acc_signals=acc_signals,
            hr_results=hr_results,
            ground_truth=ground_truth,
            sampling_rate=ppg_sampling_rate,
            acc_sampling_rate=acc_sampling_rate,
            output_dir=output_dir,
            subject_id=output_name,
        )

        print("5. Evaluating performance...")
        evaluation = evaluate_pipeline_performance(
            ppg_raw=ppg_signal,
            ppg_processed=ppg_processed,
            hr_results=hr_results,
            ground_truth=ground_truth,
            sampling_rate=ppg_sampling_rate,
        )
        results["evaluation"] = evaluation

        print_evaluation_summary(evaluation)

        print("6. Saving results...")
        results_path = os.path.join(output_dir, f"results_{output_name}.json")
        save_results_to_json(results, results_path)
        print(f"   Results saved to: {results_path}")

        eval_path = os.path.join(output_dir, f"evaluation_{output_name}.json")
        save_results_to_json(evaluation, eval_path)
        print(f"   Evaluation saved to: {eval_path}")

        print("-" * 50)
        print("Pipeline completed successfully!")
        print(f"Output directory: {output_dir}")

        return results

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        results["error"] = str(e)
        results["success"] = False
        return results


if __name__ == "__main__":
    USE_REAL_WORLD_DATA = True
    real_world_files = get_available_real_world_files("data/real_world")
    ppg_dalia_subjects = get_available_subjects("data/ppg_dalia")

    print(f"Available real-world files: {len(real_world_files)}")
    for file in real_world_files:
        print(f"  - {os.path.basename(file)}")

    print(f"Available PPG Dalia subjects: {ppg_dalia_subjects}")

    if real_world_files and USE_REAL_WORLD_DATA:
        print("\nRunning pipeline with real-world data...")
        results = run_pipeline(
            data_identifier=real_world_files[5],
            duration=4000.0,
            start_time=100.0,
            data_path="../data",
            auto_detect=True,
            dataset_type="real",
        )
    elif ppg_dalia_subjects:
        print("\nRunning pipeline with PPG Dalia data...")
        results = run_pipeline(
            data_identifier=ppg_dalia_subjects[0],
            duration=50.0,
            start_time=1210.0,
            data_path="data/ppg_dalia",
            auto_detect=False,
            dataset_type="ppgDalia",
        )
    else:
        print("No data found! Please check your data directory.")
