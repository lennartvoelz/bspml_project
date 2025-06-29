# Biosignal Processing and Heart Rate Estimation

Project for PPG signal preprocessing and real-time heart rate estimation, designed for both offline analysis and mobile deployment.

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
│       ├── preprocessing/   # Signal preprocessing modules
│       └── hr_estimation/   # Heart rate estimation algorithms
├── cpp_realtime/            # C++ real-time implementation
│   ├── src/                 # Source code
│   ├── include/             # Header files
│   └── tests/               # C++ unit tests
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
- **Real-time processing**: Optimized algorithms for mobile deployment
- **Performance validation**: Evaluation on public and real-world datasets

## Datasets

- **PPG-DaLiA**: Public dataset for algorithm validation
- **Real-world data**: Self-collected data for real-world performance testing