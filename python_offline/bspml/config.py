"""
Configuration Module

Provides configuration management for PPG signal processing parameters
for different datasets (ppgDalia and real-world data).
"""

import json
import os
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file (JSON format)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file is not found
        json.JSONDecodeError: If config file is not valid JSON
    """
    if config_path is None:
        # Use default config file path
        config_path = os.path.join(os.path.dirname(
            __file__), "..", "config", "default_config.json")
        config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in configuration file {config_path}: {e}")
    except IOError as e:
        raise IOError(f"Could not read configuration file {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except IOError as e:
        print(f"Error saving configuration to {config_path}: {e}")


def get_dataset_config(config: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset type.

    Args:
        config: Full configuration dictionary
        dataset_type: Dataset type ('ppgDalia' or 'real')

    Returns:
        Configuration dictionary for the specified dataset

    Raises:
        ValueError: If dataset_type is not supported
    """
    if dataset_type not in config:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                         f"Supported types: {list(config.keys())}")

    return config[dataset_type]


def print_config(config: Dict[str, Any], dataset_type: Optional[str] = None) -> None:
    """
    Print configuration parameters in a readable format.

    Args:
        config: Configuration dictionary
        dataset_type: Specific dataset type to print (None for all)
    """
    if dataset_type is not None:
        if dataset_type not in config:
            print(f"Dataset type '{dataset_type}' not found in configuration")
            return
        datasets = [dataset_type]
    else:
        datasets = list(config.keys())

    for dataset in datasets:
        print(f"\n=== {dataset.upper()} Configuration ===")
        dataset_config = config[dataset]

        print("RLS Filter:")
        print(f"  - Forgetting factor: {dataset_config['forgetting_factor']}")
        print(f"  - Filter order: {dataset_config['filter_order']}")
        print(
            f"  - Auto delay detection: {dataset_config['auto_delay_detection']}")

        print("Bandpass Filter:")
        print(f"  - Low cutoff: {dataset_config['low_cutoff']} Hz")
        print(f"  - High cutoff: {dataset_config['high_cutoff']} Hz")
        print(f"  - Filter order: {dataset_config['bandpass_filter_order']}")
        print(f"  - Filter type: {dataset_config['filter_type']}")

        print("Wavelet Detrending:")
        print(f"  - Wavelet: {dataset_config['wavelet']}")
        print(f"  - Levels: {dataset_config['levels']}")
        print(f"  - Mode: {dataset_config['mode']}")

        print("Motion Detection:")
        print(f"  - Motion threshold: {dataset_config['motion_threshold']}")
        print(
            f"  - Adaptive motion removal: {dataset_config['adaptive_motion_removal']}")

        print("Processing Flags:")
        print(f"  - Enable detrending: {dataset_config['enable_detrending']}")
        print(f"  - Enable denoising: {dataset_config['enable_denoising']}")
        print(
            f"  - Enable motion removal: {dataset_config['enable_motion_removal']}")

        print("Heart Rate Estimation:")
        print(
            f"  - Peak detection method: {dataset_config['peak_detection_method']}")
        print(f"  - Window size: {dataset_config['window_size']} s")
        print(f"  - Overlap: {dataset_config['overlap']}")
        print(
            f"  - Min peak distance: {dataset_config['min_peak_distance']} s")
        print(f"  - Prominence: {dataset_config['prominence']}")
        print(
            f"  - Adaptive threshold: {dataset_config['adaptive_threshold']}")
        print(
            f"  - Use envelope method: {dataset_config['use_envelope_method']}")
        print(
            f"  - Interpolation rate: {dataset_config['interpolation_rate']} Hz")
