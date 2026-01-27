"""
load_icicle_trial_with_gait_events.py

Utility function to load ICICLE raw HDF5 data and associated
GaitRite Initial/Final Contacts for supervised gait-event detection.

Author: ---
Date: 26 January 2026
"""

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_icicle_trial_with_gait_events(
    h5_file: Path,
    gaitrite_excel: Path,
    sensor: str = "WT",
    signal: str = "acc",
    axes=("x", "y", "z"),
    gait_id: int | None = None,
    plot: bool = False,
):
    """
    Load ICICLE raw IMU data and corresponding GaitRite IC/FC annotations.

    Parameters
    ----------
    h5_file : Path
        Path to ICICLE .h5 file.
    gaitrite_excel : Path
        Path to GaitRite Excel file.
    sensor : str, optional
        Sensor placement: {'HD','C7','WT','LA','RA'}.
    signal : str, optional
        Signal type: {'acc','gyr'}.
    axes : tuple, optional
        Axes to extract: any combination of ('x','y','z').
    gait_id : int or None, optional
        If provided, restrict data to a single Gait_Id.
    plot : bool, optional
        If True, plot signal(s) with IC/FC overlaid.

    Returns
    -------
    time_sec : np.ndarray, shape (N,)
        Time vector in seconds.
    data : np.ndarray, shape (N, n_channels)
        Extracted raw signal(s).
    events : np.ndarray, shape (M, 2)
        Gait events [IC, FC] in seconds.
    """

    # --------------------------------------------------
    # 1. Parse subject / timepoint / test from filename
    # --------------------------------------------------
    fname = Path(h5_file).name
    id_part = fname[16:]
    subject_tp = id_part.split("_")[0]
    subject = subject_tp[:7]
    timepoint = subject_tp[7:]
    test_opal = id_part.split("_")[1][:2]

    # --------------------------------------------------
    # 2. Read IMU data from HDF5
    # --------------------------------------------------
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = [axis_map[a] for a in axes]

    with h5py.File(h5_file, "r") as f:

        monitor_labels = f.attrs["MonitorLabelList"]
        case_ids = f.attrs["CaseIdList"]

        sensor_idx = None
        for i, lbl in enumerate(monitor_labels):
            if lbl.decode("utf-8") == sensor:
                sensor_idx = i
                break

        if sensor_idx is None:
            raise ValueError(f"Sensor {sensor} not found in file")

        sensor_group = case_ids[sensor_idx].decode("utf-8")
        base_path = f"/{sensor_group}"

        if signal == "acc":
            raw = f[f"{base_path}/Calibrated/Accelerometers"][:].T
        elif signal == "gyr":
            raw = f[f"{base_path}/Calibrated/Gyroscopes"][:].T
        else:
            raise ValueError("signal must be 'acc' or 'gyr'")

        time = f[f"{base_path}/Time"][:]
        time_sec = (time - time[0]) / 1e6

        data = raw[axis_idx, :].T  # (N, n_axes)

    # --------------------------------------------------
    # 3. Load GaitRite annotations
    # --------------------------------------------------
    T = pd.read_excel(gaitrite_excel)

    T["First_name"] = T["First_name"].astype(str)
    T["WalkTask"] = T["WalkTask"].astype(str)

    key = subject + timepoint
    subset = T[T["First_name"] == key]

    if test_opal == "SC":
        subset = subset[subset["WalkTask"] == "Cont"]
    elif test_opal == "SI":
        subset = subset[subset["WalkTask"] == "Interm"]

    if gait_id is not None:
        subset = subset[subset["Gait_Id"] == gait_id]

    if subset.empty:
        raise RuntimeError("No matching GaitRite data found")

    IC = subset["FirstContact"].values
    FC = subset["LastContact"].values
    events = np.column_stack((IC, FC))

    # --------------------------------------------------
    # 4. Optional plot
    # --------------------------------------------------
    if plot:
        plt.figure(figsize=(12, 5))
        for i, ax in enumerate(axes):
            plt.plot(
                time_sec,
                data[:, i],
                label=f"{signal.upper()} {sensor} {ax}"
            )

        for ic, fc in events:
            plt.axvline(ic, color="red", linestyle="--", alpha=0.7)
            plt.axvline(fc, color="blue", linestyle="--", alpha=0.7)

        plt.xlabel("Time [s]")
        plt.ylabel("Signal")
        plt.title(f"{subject} {timepoint} {test_opal}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return time_sec, data, events
