import pandas as pd
import numpy as np
import os

def load_ppg_dalia(base_folder):

    # Load BVP (PPG signal) - skip first 2 rows (timestamp and sample rate)
    with open(os.path.join(base_folder, "BVP.csv"), "r") as f:
        lines = f.readlines()
        start_time_unix = float(lines[0].strip())
        ppg_fs = float(lines[1].strip())
        ppg_values = np.array([float(x.strip()) for x in lines[2:]])

    # Load ACC (3D) - skipping first 2 rows as header with pandas
    acc = pd.read_csv(os.path.join(base_folder, "ACC.csv"), header=None, skiprows=2).values
    # Make sure acc is float type
    acc = acc.astype(float)

    # Load HR metadata and values
    with open(os.path.join(base_folder, "HR.csv"), "r") as f:
        lines = f.readlines()
        start_time_unix = float(lines[0].strip())
        hr_fs = float(lines[1].strip())
        hr_values = np.array([float(x.strip()) for x in lines[2:]])

    return {
        "ppg": ppg_values,
        "ppg_fs": ppg_fs,
        "acc_xyz": acc,
        "hr": hr_values,
        "hr_fs": hr_fs,
        "hr_start_time": start_time_unix
    }

def load_activity_intervals(csv_path):
    """
    Load activity intervals from S1_activity.csv, using NO_ACTIVITY lines as end markers for previous activities.
    Returns a list of (activity_label, start_time, end_time) excluding NO_ACTIVITY intervals.
    """
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["activity", "time"])
    
    intervals = []
    current_activity = None
    current_start_time = None

    for _, row in df.iterrows():
        activity = row["activity"][1:].strip()
        time = row["time"]

        if activity == "NO_ACTIVITY":
            # Use NO_ACTIVITY as the end time if an activity is open
            if current_activity is not None:
                intervals.append((current_activity, current_start_time, time))
                current_activity = None
                current_start_time = None
        else:
            # Start a new activity
            current_activity = activity
            current_start_time = time

    return intervals