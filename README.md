# Biosignal Processing and Heart Rate Estimation

Project for PPG signal preprocessing and heart rate estimation, designed for offline analysis.

## Project Overview

This project implements signal processing techniques for photoplethysmography (PPG) signals to extract accurate heart rate measurements. The system addresses three major signal quality issues:

1. **Detrending**: Baseline drift removal using wavelet-based reconstruction
2. **Denoising**: High-frequency noise reduction with band-pass filtering (0.5–4 Hz)
3. **Motion Artifact Removal**: Motion-induced distortion correction using RLS filter with accelerometer reference

## Project Structure

```
bspml_project/
├── python_offline/          # Python-based offline processing and prototyping
│   ├── bspml/
│   │   ├── preprocessing/   # Signal preprocessing modules
│   │   ├── hr_estimation/   # Heart rate estimation algorithms
│   │   ├── data_loading.py  # Data loading utilities
│   │   ├── config.py        # Configuration management
│   │   └── evaluation.py    # Performance evaluation
│   ├── config/              # Configuration files
│   │   └── default_config.json
│   └── run_pipeline.py      # Main pipeline execution script
├── data/                    # Dataset storage
│   ├── ppg_dalia/           # PPG-DaLiA dataset
│   └── real_world/          # Self-collected data
```

## Features

### Signal Preprocessing
- **Wavelet-based detrending**: Removes baseline drift caused by respiration
- **Band-pass filtering**: Eliminates high-frequency noise and signal jitter
- **Motion artifact removal**: Uses RLS adaptive filtering with accelerometer reference

### Heart Rate Estimation
- **Local maxima detection**: Extracts peak candidates using windowed local maxima filtering
- **Performance validation**: Evaluation on public and real-world datasets

## Datasets

- **PPG-DaLiA**: Public dataset for algorithm validation
- **Real-world data**: Self-collected data for real-world performance testing

## Quick Start

### Installation

1. Clone the repository

2. Install required dependencies:
```bash
pip install -r python_offline/requirements.txt
```

### Running the Pipeline

The pipeline can be executed in several ways depending on your data type and requirements.

#### Method 1: Using the Main Script

The simplest way to run the pipeline is using the main script:

```bash
python python_offline/run_pipeline.py
```

This will automatically detect available data and run the pipeline with default settings.

#### Method 2: Programmatic Usage

For more control, you can use the pipeline programmatically. In the `python_offline/run_pipeline.py` script, you can specify a
flag for switching between real-world data and PPG-DaLiA dataset processing: 'USE_REAL_WORLD_DATA'. With this flag set to `True`, the pipeline will process real-world data files. If set to `False`, it will process the PPG-DaLiA dataset. Running the script will list the available data files, including self collected as well as PPG-DaLiA data (you have to copy the files into the data folder before using the pipeline!). You can then choose via the index which file to process and can set start time and duration for the processing. The script will then run the pipeline on the selected file.

### Configuration File Structure

The default configuration file (`config/default_config.json`) contains settings and can be adjusted:
```json
{
    "ppgDalia": {
        "forgetting_factor": 0.99,
        "filter_order": 5,
        "delay_compensation": null,
        "auto_delay_detection": false,
        "low_cutoff": 0.5,
        "high_cutoff": 4.0,
        "bandpass_filter_order": 4,
        "filter_type": "butterworth",
        "wavelet": "db4",
        "levels": 6,
        "mode": "symmetric",
        "motion_threshold": 50.0,
        "adaptive_motion_removal": true,
        "enable_detrending": true,
        "enable_denoising": true,
        "enable_motion_removal": true,
        "peak_detection_method": "sliding_window",
        "window_size": 10.0,
        "overlap": 0.3,
        "min_peak_distance": 0.4,
        "prominence": 1.0,
        "min_peak_height": 10,
        "adaptive_threshold": true,
        "use_envelope_method": false,
        "interpolation_rate": 0.6,
        "return_peaks": true,
        "max_peak_distance": 2.0,
        "prominence_factor": 0.5
    },
    "real": {
        "forgetting_factor": 0.9,
        "filter_order": 5,
        "delay_compensation": null,
        "auto_delay_detection": false,
        "low_cutoff": 0.5,
        "high_cutoff": 4.0,
        "bandpass_filter_order": 4,
        "filter_type": "butterworth",
        "wavelet": "db4",
        "levels": 6,
        "mode": "symmetric",
        "motion_threshold": 50.0,
        "adaptive_motion_removal": true,
        "enable_detrending": true,
        "enable_denoising": true,
        "enable_motion_removal": true,
        "peak_detection_method": "sliding_window",
        "window_size": 10.0,
        "overlap": 0.2,
        "min_peak_distance": 0.45,
        "prominence": 100.0,
        "min_peak_height": 10,
        "adaptive_threshold": true,
        "use_envelope_method": false,
        "interpolation_rate": 0.5,
        "return_peaks": true,
        "max_peak_distance": 1.5,
        "prominence_factor": 0.8
    }
}
```

### Configuration Parameters

#### RLS Filter Parameters
- **`forgetting_factor`**: Controls adaptation rate (0.9-0.99). Lower values = faster adaptation
- **`filter_order`**: Order of the RLS filter (typically 5-10)
- **`delay_compensation`**: Manual delay compensation in samples (null for auto-detection)
- **`auto_delay_detection`**: Enable automatic delay detection between PPG and accelerometer

#### Signal Processing Parameters
- **`low_cutoff`**: Low-pass filter cutoff frequency in Hz (typically 0.5)
- **`high_cutoff`**: High-pass filter cutoff frequency in Hz (typically 4.0)
- **`bandpass_filter_order`**: Order of the bandpass filter
- **`filter_type`**: Filter type ("butterworth", "chebyshev", etc.)
- **`wavelet`**: Wavelet type for detrending ("db4", "db6", "haar", etc.)
- **`levels`**: Number of wavelet decomposition levels
- **`mode`**: Wavelet boundary condition ("symmetric", "periodization", etc.)

#### Motion Detection Parameters
- **`motion_threshold`**: Threshold for motion detection (higher = less sensitive)
- **`adaptive_motion_removal`**: Enable adaptive motion artifact removal
- **`enable_detrending`**: Enable wavelet-based detrending
- **`enable_denoising`**: Enable bandpass filtering
- **`enable_motion_removal`**: Enable RLS-based motion artifact removal

#### Heart Rate Estimation Parameters
- **`peak_detection_method`**: Peak detection algorithm ("sliding_window")
- **`window_size`**: Window size for peak detection in seconds
- **`overlap`**: Window overlap ratio (0.0-1.0)
- **`min_peak_distance`**: Minimum distance between peaks in seconds
- **`prominence`**: Minimum peak prominence
- **`min_peak_height`**: Minimum peak height
- **`adaptive_threshold`**: Enable adaptive peak detection threshold
- **`interpolation_rate`**: Heart rate interpolation rate in Hz
- **`return_peaks`**: Include peak information in results
- **`max_peak_distance`**: Maximum allowed distance between peaks in seconds
- **`prominence_factor`**: Factor for adaptive prominence calculation

## Working with Real-World Data

### Data Format Requirements

Real-world data files should be tab-separated `.txt` files with the following columns:
- **Time Difference**: Time difference in milliseconds
- **IR**: Infrared PPG signal
- **RED**: Red PPG signal (used as primary PPG)
- **ACC**: Accelerometer signal (single-axis)