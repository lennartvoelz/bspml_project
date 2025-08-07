"""
Module for PPG signal preprocessing and heart rate estimation.

- Signal preprocessing (detrending, denoising, motion artifact removal)
- Heart rate estimation algorithms
- Data loading utilities for PPG Dalia and real-world data
- Evaluation metrics with optional ground truth support

"""

# Import main modules
from . import preprocessing
from . import hr_estimation
from . import data_loading
from . import evaluation
from . import config

# Import key functions
from .preprocessing import preprocess_ppg
from .hr_estimation import estimate_heart_rate
from .data_loading import (
    load_ppg_dalia_data,
    load_accelerometer_data,
    load_ground_truth_data,
    load_real_world_data,
    load_data_auto,
    detect_data_type,
    get_available_subjects,
    get_available_real_world_files,
)
from .evaluation import evaluate_pipeline_performance, print_evaluation_summary
from .config import load_config, get_dataset_config, print_config

__all__ = [
    "preprocessing",
    "hr_estimation",
    "data_loading",
    "evaluation",
    "config",
    "preprocess_ppg",
    "estimate_heart_rate",
    "load_ppg_dalia_data",
    "load_accelerometer_data",
    "load_ground_truth_data",
    "get_available_subjects",
    "evaluate_pipeline_performance",
    "print_evaluation_summary",
    "load_config",
    "get_dataset_config",
    "print_config",
]
