import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple, Dict


def load_ppg_dalia_data(
    subject_id: str = "S1",
    duration: Optional[float] = None,
    start_time: float = 0,
    data_path: str = "data/ppg_dalia",
) -> Tuple[np.ndarray, float, Dict]:
    """
    Load PPG data from the PPG Dalia dataset.

    Args:
        subject_id: Subject identifier (S1, S2, S3, S4, S5)
        duration: Duration in seconds to load (None for full signal)
        start_time: Start time offset in seconds
        data_path: Path to the PPG Dalia dataset

    Returns:
        Tuple of (ppg_signal, sampling_rate, metadata)
    """
    subject_path = os.path.join(data_path, subject_id)

    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Subject path not found: {subject_path}")

    # Load BVP (PPG) data
    bvp_file = os.path.join(subject_path, "BVP.csv")
    if not os.path.exists(bvp_file):
        raise FileNotFoundError(f"BVP file not found: {bvp_file}")

    with open(bvp_file, "r") as f:
        lines = f.read().strip().split("\n")

    # First line is start timestamp, second line is sampling frequency
    start_timestamp = float(lines[0])
    sampling_freq = float(lines[1])

    # Rest are PPG values
    ppg_values = np.array([float(line) for line in lines[2:]])

    # Initialize metadata
    metadata = {
        "start_timestamp": start_timestamp,
        "sampling_freq": sampling_freq,
        "data_source": "PPG_Dalia_BVP",
        "subject_id": subject_id,
    }

    # Load additional metadata if available
    try:
        # Load activity data
        activity_file = os.path.join(
            subject_path, f"{subject_id}_activity.csv")
        if os.path.exists(activity_file):
            activity_df = pd.read_csv(activity_file)
            activities = []
            for _, row in activity_df.iterrows():
                if len(row) >= 2:
                    activity_name = str(row.iloc[0]).strip().replace("# ", "")
                    timestamp = row.iloc[1]
                    if activity_name != "SUBJECT_ID":
                        activities.append(
                            {"activity": activity_name, "timestamp": timestamp}
                        )
            metadata["activities"] = activities
    except Exception as e:
        print(f"Warning: Could not load activity data: {e}")

    try:
        # Load subject info
        quest_file = os.path.join(subject_path, f"{subject_id}_quest.csv")
        if os.path.exists(quest_file):
            quest_df = pd.read_csv(quest_file)
            subject_info = {}
            for _, row in quest_df.iterrows():
                if len(row) >= 2:
                    key = str(row.iloc[0]).strip().replace("# ", "")
                    value = row.iloc[1]
                    if key != "SUBJECT_ID":
                        subject_info[key.lower()] = value
            metadata["subject_info"] = subject_info
    except Exception as e:
        print(f"Warning: Could not load subject info: {e}")

    # Apply time windowing
    if start_time > 0 or duration is not None:
        start_sample = int(start_time * sampling_freq)

        if duration is not None:
            end_sample = start_sample + int(duration * sampling_freq)
            end_sample = min(end_sample, len(ppg_values))
        else:
            end_sample = len(ppg_values)

        ppg_values = ppg_values[start_sample:end_sample]

    ppg_signal = ppg_values.astype(np.float64)

    # Update metadata with final signal properties
    metadata["duration"] = len(ppg_signal) / sampling_freq
    metadata["num_samples"] = len(ppg_signal)
    metadata["start_time"] = start_time
    metadata["requested_duration"] = duration

    return ppg_signal, sampling_freq, metadata


def load_accelerometer_data(
    subject_id: str = "S1",
    duration: Optional[float] = None,
    start_time: float = 0,
    data_path: str = "data/ppg_dalia",
) -> Tuple[np.ndarray, float, Dict]:
    """
    Load accelerometer data from the PPG Dalia dataset.

    Args:
        subject_id: Subject identifier (S1, S2, S3, S4, S5)
        duration: Duration in seconds to load (None for full signal)
        start_time: Start time offset in seconds
        data_path: Path to the PPG Dalia dataset

    Returns:
        Tuple of (acc_signals, sampling_rate, metadata)
    """
    subject_path = os.path.join(data_path, subject_id)

    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Subject path not found: {subject_path}")

    # Load ACC data
    acc_file = os.path.join(subject_path, "ACC.csv")
    if not os.path.exists(acc_file):
        raise FileNotFoundError(f"ACC file not found: {acc_file}")

    with open(acc_file, "r") as f:
        lines = f.read().strip().split("\n")

    # First line contains start timestamps for each axis (should be same)
    start_timestamps = [float(x.strip()) for x in lines[0].split(",")]
    start_timestamp = start_timestamps[0]  # Use first timestamp

    # Second line contains sampling frequencies for each axis (should be same)
    sampling_freqs = [float(x.strip()) for x in lines[1].split(",")]
    sampling_freq = sampling_freqs[0]  # Use first sampling frequency

    # Rest are accelerometer values (x, y, z)
    acc_data = []
    for line in lines[2:]:
        values = [float(x.strip()) for x in line.split(",")]
        if len(values) == 3:  # Ensure we have x, y, z values
            acc_data.append(values)

    acc_signals = np.array(acc_data)  # Shape: [n_samples, 3]

    # Initialize metadata
    metadata = {
        "start_timestamp": start_timestamp,
        "sampling_freq": sampling_freq,
        "data_source": "PPG_Dalia_ACC",
        "subject_id": subject_id,
        "axes": ["x", "y", "z"],
    }

    # Apply time windowing
    if start_time > 0 or duration is not None:
        start_sample = int(start_time * sampling_freq)

        if duration is not None:
            end_sample = start_sample + int(duration * sampling_freq)
            end_sample = min(end_sample, len(acc_signals))
        else:
            end_sample = len(acc_signals)

        acc_signals = acc_signals[start_sample:end_sample]

    # Update metadata with final signal properties
    metadata["duration"] = len(acc_signals) / sampling_freq
    metadata["num_samples"] = len(acc_signals)
    metadata["start_time"] = start_time
    metadata["requested_duration"] = duration

    return acc_signals, sampling_freq, metadata


def load_real_world_data(
    file_path: str,
    duration: Optional[float] = None,
    start_time: float = 0,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Load real-world PPG and accelerometer data from .txt files.

    Args:
        file_path: Path to the .txt file containing real-world data
        duration: Duration in seconds to load (None for full signal)
        start_time: Start time offset in seconds

    Returns:
        Tuple of (ppg_signal, acc_signals, sampling_rate, metadata)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real-world data file not found: {file_path}")

    # Read the data file
    try:
        data = pd.read_csv(file_path, sep="\t")
    except Exception:
        try:
            data = pd.read_csv(file_path, sep=" ")
        except Exception:
            data = pd.read_csv(file_path)

    expected_columns = ["Time Difference", "IR", "RED", "ACC"]
    if not all(col in data.columns for col in expected_columns):
        raise ValueError(
            f"Expected columns {expected_columns}, but found {list(data.columns)}"
        )

    # Extract data
    time_diff = data["Time Difference"].values  # in milliseconds
    ir_signal = data["IR"].values
    red_signal = data["RED"].values
    acc_signal = data["ACC"].values

    # Calculate sampling rate from time differences
    # Convert milliseconds to seconds and calculate frequency
    avg_time_diff_sec = np.mean(time_diff) / 1000.0
    sampling_rate = 1.0 / avg_time_diff_sec

    # Use red signal as PPG
    ppg_signal = red_signal.astype(np.float64)

    # Create 3-axis accelerometer data (using single ACC value for all axes)
    # This is a simplification - real data might have separate x,y,z values
    acc_signals = np.column_stack([acc_signal, acc_signal, acc_signal]).astype(
        np.float64
    )

    # Create metadata
    filename = os.path.basename(file_path)
    metadata = {
        "data_source": "Real_World_Data",
        "filename": filename,
        "sampling_freq": sampling_rate,
        "avg_time_diff_ms": np.mean(time_diff),
        "time_diff_std_ms": np.std(time_diff),
        "original_length": len(ppg_signal),
        "has_ground_truth": False,
    }

    # Apply time windowing if requested
    if start_time > 0 or duration is not None:
        start_sample = int(start_time * sampling_rate)

        if duration is not None:
            end_sample = start_sample + int(duration * sampling_rate)
            end_sample = min(end_sample, len(ppg_signal))
        else:
            end_sample = len(ppg_signal)

        ppg_signal = ppg_signal[start_sample:end_sample]
        acc_signals = acc_signals[start_sample:end_sample]

    # Update metadata with final signal properties
    metadata["duration"] = len(ppg_signal) / sampling_rate
    metadata["num_samples"] = len(ppg_signal)
    metadata["start_time"] = start_time
    metadata["requested_duration"] = duration

    return ppg_signal, acc_signals, sampling_rate, metadata


def load_ground_truth_data(
    subject_id: str = "S1",
    start_time: float = 0,
    duration: Optional[float] = None,
    data_path: str = "data/ppg_dalia",
    optional: bool = False,
) -> Dict:
    """
    Load ground truth heart rate data from PPG Dalia dataset.

    Args:
        subject_id: Subject identifier (S1, S2, S3, S4, S5)
        start_time: Start time offset in seconds
        duration: Duration in seconds to load (None for full signal)
        data_path: Path to the PPG Dalia dataset
        optional: If True, don't print warnings when ground truth is not available

    Returns:
        Dictionary containing ground truth data (or None values if not available)
    """
    # Return empty ground truth if data_path doesn't exist (e.g., for real-world data)
    if not os.path.exists(data_path):
        return {"heart_rate": None}

    subject_path = os.path.join(data_path, subject_id)

    # Return empty ground truth if subject path doesn't exist
    if not os.path.exists(subject_path):
        if not optional:
            print(f"Warning: Subject path not found: {subject_path}")
        return {"heart_rate": None}

    ground_truth = {}

    # Load ground truth heart rate from HR.csv
    hr_file = os.path.join(subject_path, "HR.csv")

    try:
        if os.path.exists(hr_file):
            with open(hr_file, "r") as f:
                lines = f.read().strip().split("\n")

            # First line is start timestamp, second line is sampling frequency
            hr_start_timestamp = float(lines[0])
            hr_sampling_freq = float(lines[1])

            # Rest are heart rate values in bpm
            hr_values = np.array([float(line)
                                 for line in lines[2:] if line.strip()])

            # Create time axis for HR data
            hr_time_axis = np.arange(len(hr_values)) / hr_sampling_freq

            # Apply time windowing
            if start_time > 0 or duration is not None:
                start_idx = int(start_time * hr_sampling_freq)

                if duration is not None:
                    end_idx = start_idx + int(duration * hr_sampling_freq)
                    end_idx = min(end_idx, len(hr_values))
                else:
                    end_idx = len(hr_values)

                hr_values = hr_values[start_idx:end_idx]
                hr_time_axis = hr_time_axis[start_idx:end_idx]

            ground_truth["heart_rate"] = {
                "values": hr_values,
                "time_axis": hr_time_axis,
                "sampling_freq": hr_sampling_freq,
                "mean": np.mean(hr_values),
                "std": np.std(hr_values),
                "min": np.min(hr_values),
                "max": np.max(hr_values),
                "start_timestamp": hr_start_timestamp,
            }
        else:
            if not optional:
                print(f"Warning: HR file not found: {hr_file}")
            ground_truth["heart_rate"] = None

    except Exception as e:
        if not optional:
            print(f"Could not load ground truth heart rate: {e}")
        ground_truth["heart_rate"] = None

    return ground_truth


def detect_data_type(data_path: str) -> str:
    """
    Detect the type of data based on directory structure and file extensions.

    Args:
        data_path: Path to the data directory or file

    Returns:
        String indicating data type: 'ppg_dalia', 'real_world', or 'unknown'
    """
    if os.path.isfile(data_path):
        # Single file - check extension and content
        if data_path.endswith(".txt"):
            return "real_world"
        elif data_path.endswith(".csv"):
            return "ppg_dalia"
    elif os.path.isdir(data_path):
        # Directory - check structure
        if "real_world" in data_path.lower() or any(
            f.endswith(".txt") for f in os.listdir(data_path)
        ):
            return "real_world"
        elif any(
            os.path.isdir(os.path.join(data_path, item))
            for item in os.listdir(data_path)
        ):
            # Has subdirectories, likely PPG Dalia format
            return "ppg_dalia"

    return "unknown"


def get_available_real_world_files(data_path: str = "data/real_world") -> list:
    """
    Get list of available real-world data files.

    Args:
        data_path: Path to real-world data directory

    Returns:
        List of available .txt files
    """
    # Handle relative paths
    if not os.path.isabs(data_path):
        # Try to resolve relative to current working directory
        abs_path = os.path.abspath(data_path)
        if os.path.exists(abs_path):
            data_path = abs_path
        else:
            # Try relative to this module's directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(module_dir))
            alt_path = os.path.join(project_root, data_path)
            if os.path.exists(alt_path):
                data_path = alt_path

    if not os.path.exists(data_path):
        return []

    txt_files = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            txt_files.append(os.path.join(data_path, file))

    return sorted(txt_files)


def load_data_auto(
    data_identifier: str,
    duration: Optional[float] = None,
    start_time: float = 0,
    data_path: str = "data/",
) -> Tuple[np.ndarray, Optional[np.ndarray], float, float, Dict, Dict]:
    """
    Automatically detect data type and load appropriate data.

    Args:
        data_identifier: Subject ID for PPG Dalia or file path for real-world data
        duration: Duration in seconds to load (None for full signal)
        start_time: Start time offset in seconds
        data_path: Base path to data directory

    Returns:
        Tuple of (ppg_signal, acc_signals, ppg_sampling_rate, acc_sampling_rate, metadata, ground_truth)
    """
    # Try to determine data type
    if os.path.isfile(data_identifier):
        # Direct file path provided
        data_type = detect_data_type(data_identifier)
        if data_type == "real_world":
            ppg_signal, acc_signals, sampling_rate, metadata = load_real_world_data(
                data_identifier, duration, start_time
            )
            # No ground truth for real-world data
            ground_truth = {"heart_rate": None}
            return ppg_signal, acc_signals, sampling_rate, sampling_rate, metadata, ground_truth

    # Check if it's a real-world file in the data directory
    real_world_path = os.path.join(data_path, "real_world")
    if os.path.exists(real_world_path):
        potential_file = os.path.join(real_world_path, data_identifier)
        if os.path.exists(potential_file):
            ppg_signal, acc_signals, sampling_rate, metadata = load_real_world_data(
                potential_file, duration, start_time
            )
            ground_truth = {"heart_rate": None}
            return ppg_signal, acc_signals, sampling_rate, sampling_rate, metadata, ground_truth

    # Default to PPG Dalia format
    ppg_dalia_path = os.path.join(data_path, "ppg_dalia")

    ppg_signal, ppg_sampling_rate, ppg_metadata = load_ppg_dalia_data(
        subject_id=data_identifier,
        duration=duration,
        start_time=start_time,
        data_path=ppg_dalia_path,
    )

    acc_signals, acc_sampling_rate, acc_metadata = load_accelerometer_data(
        subject_id=data_identifier,
        duration=duration,
        start_time=start_time,
        data_path=ppg_dalia_path,
    )

    ground_truth = load_ground_truth_data(
        subject_id=data_identifier,
        start_time=start_time,
        duration=duration,
        data_path=ppg_dalia_path,
    )

    # Combine metadata
    combined_metadata = {**ppg_metadata, **acc_metadata}

    return ppg_signal, acc_signals, ppg_sampling_rate, acc_sampling_rate, combined_metadata, ground_truth


def get_available_subjects(data_path: str = "data/ppg_dalia") -> list:
    """
    Get list of available subjects in the dataset.

    Args:
        data_path: Path to the PPG Dalia dataset

    Returns:
        List of available subject IDs
    """
    if not os.path.exists(data_path):
        return []

    subjects = []
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path) and item.startswith("S"):
            subjects.append(item)

    return sorted(subjects)
