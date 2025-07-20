"""
Module for PPG signal preprocessing and heart rate estimation.

- Signal preprocessing (detrending, denoising, motion artifact removal)
- Heart rate estimation algorithms
- Data loading utilities

"""

# Import main modules
from . import preprocessing
from . import hr_estimation
from . import data_loading
from . import evaluation

# Import key functions
from .preprocessing import preprocess_ppg
from .hr_estimation import estimate_heart_rate
from .data_loading import load_ppg_dalia_data, load_accelerometer_data, load_ground_truth_data, get_available_subjects
from .evaluation import evaluate_pipeline_performance, print_evaluation_summary

__all__ = [
    "preprocessing",
    "hr_estimation",
    "data_loading",
    "evaluation",
    "preprocess_ppg",
    "estimate_heart_rate",
    "load_ppg_dalia_data",
    "load_accelerometer_data",
    "load_ground_truth_data",
    "get_available_subjects",
    "evaluate_pipeline_performance",
    "print_evaluation_summary"
]
