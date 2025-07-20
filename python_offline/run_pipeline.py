"""
PPG Signal Processing Pipeline

This script executes the complete PPG signal processing pipeline including:
1. Data loading from PPG Dalia dataset
2. Preprocessing (wavelet detrending, bandpass filtering, RLS motion artifact removal)
3. Heart rate estimation
4. Results visualization and saving
"""

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
    preprocess_ppg,
    estimate_heart_rate,
    get_available_subjects,
    evaluate_pipeline_performance,
    print_evaluation_summary
)


def create_output_directory(base_path: str = "results") -> str:
    """Create timestamped output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"pipeline_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results_to_json(results: Dict[str, Any], output_path: str):
    """Save results dictionary to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
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

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)


def plot_pipeline_results(
    ppg_raw: np.ndarray,
    ppg_processed: np.ndarray,
    acc_signals: np.ndarray,
    hr_results: Dict[str, Any],
    ground_truth: Dict[str, Any],
    sampling_rate: float,
    acc_sampling_rate: float,
    output_dir: str,
    subject_id: str
):
    """Create comprehensive visualization of pipeline results."""

    # Create time axis for PPG signals
    time_axis = np.arange(len(ppg_raw)) / sampling_rate

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(
        f'PPG Processing Pipeline Results - Subject {subject_id}', fontsize=16)

    # Plot 1: Raw vs Processed PPG
    axes[0].plot(time_axis, ppg_raw, 'b-', alpha=0.7,
                 label='Raw PPG', linewidth=0.8)
    axes[0].plot(time_axis, ppg_processed, 'r-',
                 label='Processed PPG', linewidth=1.0)
    axes[0].set_ylabel('PPG Amplitude')
    axes[0].set_title('Raw vs Processed PPG Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accelerometer signals (combined magnitude)
    from bspml.preprocessing import combine_accelerometer_channels, resample_accelerometer_data

    # Resample and combine accelerometer for visualization
    acc_resampled_viz = resample_accelerometer_data(
        acc_signals, acc_sampling_rate, sampling_rate, len(ppg_raw))
    acc_combined_viz = combine_accelerometer_channels(acc_resampled_viz)
    acc_magnitude = acc_combined_viz.flatten()

    acc_time = np.arange(len(acc_magnitude)) / sampling_rate
    axes[1].plot(acc_time, acc_magnitude, 'purple', linewidth=2,
                 label='ACC Magnitude (Combined)', alpha=0.8)
    axes[1].set_ylabel('Acceleration Magnitude')
    axes[1].set_title('Combined Accelerometer Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Detected peaks on processed signal
    if hr_results.get('success', False):
        # Show a portion of the signal with detected peaks
        window_start = len(ppg_processed) // 4
        window_end = min(window_start + int(30 * sampling_rate),
                         len(ppg_processed))  # 30 seconds

        window_time = time_axis[window_start:window_end]
        window_signal = ppg_processed[window_start:window_end]

        axes[2].plot(window_time, window_signal, 'b-', linewidth=1.0)

        # Mark detected peaks if available
        if 'peaks' in hr_results and isinstance(hr_results['peaks'], dict):
            peak_indices = hr_results['peaks'].get('indices', np.array([]))
            if len(peak_indices) > 0:
                peak_mask = (peak_indices >= window_start) & (
                    peak_indices < window_end)
                window_peaks = peak_indices[peak_mask] - window_start
                if len(window_peaks) > 0:
                    axes[2].plot(window_time[window_peaks], window_signal[window_peaks],
                                 'ro', markersize=6, label='Detected Peaks')

        axes[2].set_ylabel('PPG Amplitude')
        axes[2].set_title('Peak Detection (30s window)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Peak detection failed',
                     transform=axes[2].transAxes, ha='center', va='center')
        axes[2].set_title('Peak Detection - Failed')

    # Plot 4: Heart rate comparison
    if hr_results.get('success', False) and 'time_points' in hr_results:
        hr_time = hr_results['time_points']
        hr_values = hr_results['hr_values']
        axes[3].plot(hr_time, hr_values, 'r-',
                     linewidth=2, label='Estimated HR')

        # Plot ground truth if available
        if ground_truth.get('heart_rate') is not None:
            gt_hr = ground_truth['heart_rate']
            gt_time = gt_hr['time_axis']
            gt_values = gt_hr['values']
            axes[3].plot(gt_time, gt_values, 'g-',
                         linewidth=2, label='Ground Truth HR')

        axes[3].set_ylabel('Heart Rate (BPM)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_title('Heart Rate Estimation vs Ground Truth')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(40, 160)  # Reasonable HR range
    else:
        axes[3].text(0.5, 0.5, 'Heart rate estimation failed',
                     transform=axes[3].transAxes, ha='center', va='center')
        axes[3].set_title('Heart Rate Estimation - Failed')
        axes[3].set_xlabel('Time (seconds)')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'pipeline_results_{subject_id}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_path}")


def run_pipeline(
    subject_id: str = "S1",
    duration: Optional[float] = 60.0,  # Process 60 seconds by default
    start_time: float = 0,
    data_path: str = "../data/ppg_dalia",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete PPG processing pipeline.

    Args:
        subject_id: Subject to process
        duration: Duration in seconds to process
        start_time: Start time offset in seconds
        data_path: Path to dataset
        output_dir: Output directory (auto-created if None)

    Returns:
        Dictionary containing all results
    """
    print(f"Starting PPG processing pipeline for subject {subject_id}")
    print(f"Duration: {duration}s, Start time: {start_time}s")
    print("-" * 50)

    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory()

    results = {
        'subject_id': subject_id,
        'duration': duration,
        'start_time': start_time,
        'timestamp': datetime.now().isoformat()
    }

    try:
        # Step 1: Load PPG data
        print("1. Loading PPG data...")
        ppg_signal, ppg_sampling_rate, ppg_metadata = load_ppg_dalia_data(
            subject_id=subject_id,
            duration=duration,
            start_time=start_time,
            data_path=data_path
        )
        print(
            f"   Loaded {len(ppg_signal)} PPG samples at {ppg_sampling_rate} Hz")
        results['ppg_metadata'] = ppg_metadata

        # Step 2: Load accelerometer data
        print("2. Loading accelerometer data...")
        acc_signals, acc_sampling_rate, acc_metadata = load_accelerometer_data(
            subject_id=subject_id,
            duration=duration,
            start_time=start_time,
            data_path=data_path
        )
        print(
            f"   Loaded {len(acc_signals)} ACC samples at {acc_sampling_rate} Hz")
        results['acc_metadata'] = acc_metadata

        # Step 3: Load ground truth
        print("3. Loading ground truth data...")
        ground_truth = load_ground_truth_data(
            subject_id=subject_id,
            start_time=start_time,
            duration=duration,
            data_path=data_path
        )
        if ground_truth.get('heart_rate') is not None:
            gt_hr = ground_truth['heart_rate']
            print(
                f"   Ground truth HR: {gt_hr['mean']:.1f} ± {gt_hr['std']:.1f} BPM")
        else:
            print("   No ground truth HR data available")
        results['ground_truth'] = ground_truth

        # Step 4: Preprocess PPG signal
        print("4. Preprocessing PPG signal...")
        print("   - Wavelet detrending")
        print("   - Bandpass filtering (0.5-4 Hz)")
        print("   - RLS motion artifact removal")

        ppg_processed = preprocess_ppg(
            ppg_signal=ppg_signal,
            acc_signals=acc_signals,
            sampling_rate=ppg_sampling_rate,
            acc_sampling_rate=acc_sampling_rate,
            enable_detrending=True,
            enable_denoising=True,
            enable_motion_removal=True,
            adaptive_motion_removal=True,  # Use adaptive RLS
            motion_threshold=60.0  # High threshold - effectively disables RLS for this dataset
        )
        print(f"   Preprocessing completed")

        # Step 5: Estimate heart rate
        print("5. Estimating heart rate...")
        hr_results = estimate_heart_rate(
            ppg_signal=ppg_processed,
            sampling_rate=ppg_sampling_rate,
            return_peaks=True
        )

        # Adjust time axis to account for start_time offset
        if hr_results.get('success', False) and 'time_points' in hr_results:
            hr_results['time_points'] = hr_results['time_points'] + start_time
            if 'peaks' in hr_results and isinstance(hr_results['peaks'], dict):
                hr_results['peaks']['times'] = hr_results['peaks']['times'] + start_time

        if hr_results.get('success', False):
            hr_stats = hr_results.get('statistics', {})
            mean_hr = hr_stats.get('mean', None)
            std_hr = hr_stats.get('std', None)

            if mean_hr is not None and std_hr is not None:
                print(f"   Estimated HR: {mean_hr:.1f} ± {std_hr:.1f} BPM")
            else:
                print("   Estimated HR: Statistics not available")
            print(f"   Detected {hr_results.get('num_peaks', 0)} peaks")
        else:
            print(
                f"   Heart rate estimation failed: {hr_results.get('error', 'Unknown error')}")

        results['hr_estimation'] = hr_results

        # Step 6: Create visualizations
        print("6. Creating visualizations...")
        plot_pipeline_results(
            ppg_raw=ppg_signal,
            ppg_processed=ppg_processed,
            acc_signals=acc_signals,
            hr_results=hr_results,
            ground_truth=ground_truth,
            sampling_rate=ppg_sampling_rate,
            acc_sampling_rate=acc_sampling_rate,
            output_dir=output_dir,
            subject_id=subject_id
        )

        # Step 7: Evaluate performance
        print("7. Evaluating performance...")
        evaluation = evaluate_pipeline_performance(
            ppg_raw=ppg_signal,
            ppg_processed=ppg_processed,
            hr_results=hr_results,
            ground_truth=ground_truth,
            sampling_rate=ppg_sampling_rate
        )
        results['evaluation'] = evaluation

        # Print evaluation summary
        print_evaluation_summary(evaluation)

        # Step 8: Save results
        print("8. Saving results...")
        results_path = os.path.join(output_dir, f'results_{subject_id}.json')
        save_results_to_json(results, results_path)
        print(f"   Results saved to: {results_path}")

        # Save evaluation separately for easy access
        eval_path = os.path.join(output_dir, f'evaluation_{subject_id}.json')
        save_results_to_json(evaluation, eval_path)
        print(f"   Evaluation saved to: {eval_path}")

        print("-" * 50)
        print("Pipeline completed successfully!")
        print(f"Output directory: {output_dir}")

        return results

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        results['error'] = str(e)
        results['success'] = False
        return results


if __name__ == "__main__":
    # Get available subjects
    data_path = "/home/lvoelz/bspml_project/data/ppg_dalia"
    subjects = get_available_subjects(data_path)
    print(f"Available subjects: {subjects}")

    # Run pipeline for first available subject
    if subjects:
        results = run_pipeline(
            subject_id=subjects[0],
            duration=1000.0,  # Process 5 minutes
            start_time=2000.0,
            data_path=data_path
        )
    else:
        print("No subjects found in dataset!")
