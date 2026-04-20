import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d
from src.headwalk.utils import align_imu_to_gravity, align_imu_to_gravity_with_external_file
import re
from typing import List

def get_all_h5_files(root_dir: Path, exclude_list: list[str]) -> list[Path]:
    """Finds all .h5 files recursively, excluding specific filenames."""
    all_files = list(root_dir.rglob("*.h5"))
    return [f for f in all_files if f.name not in exclude_list]



def find_matching_h5_file(f: str | Path, candidates: List[str | Path]) -> Path | None:
    """
    Given:
        - f: absolute path to an .h5 file
        - candidates: list of paths to other .h5 files

    The function:
        1. Extracts the filename from f (without directory path)
        2. Splits the filename on both '-' and '_'
        3. Selects the third component
        4. Returns the first file in candidates whose path contains that component

    Returns:
        Path of the matching file, or None if no match is found.
    """

    f = Path(f)
    filename = f.name  # e.g. "20140613-143258-INGP056F2_SC.h5"

    # Remove extension
    stem = f.stem  # "20140613-143258-INGP056F2_SC"

    # Split on both '-' and '_'
    components = re.split(r"[-_]", stem)

    if len(components) < 3:
        raise ValueError(
            f"Filename '{filename}' does not contain at least three components."
        )

    target_component = components[2]  # third component

    # Search in candidate paths
    for candidate in candidates:
        candidate_path = Path(candidate)
        if target_component in str(candidate_path):
            return candidate_path

    return None

def get_all_h5_files_v2(root_dir: Path, exclude_strings: list[str]) -> list[Path]:
    """Finds all .h5 files recursively, excluding those whose path contains
    any of the specified substrings.
    """
    all_files = root_dir.rglob("*.h5")

    return [
        f for f in all_files
        if not any(ex_str in str(f) for ex_str in exclude_strings)
    ]

def load_icicle_trial_with_gait_events(
        h5_file: Path,
        gaitrite_df: pd.DataFrame,
        sensor: str = "HD"
):
    """Loads IMU data and GaitRite events from a pre-loaded DataFrame."""
    fname = h5_file.name
    id_part = fname.split("-")[-1]
    subject_tp = id_part.split("_")[0]
    subject = subject_tp[:7]
    timepoint = subject_tp[7:]
    test_opal = id_part.split("_")[1][:2]

    # 1. Read HDF5 Data
    with h5py.File(h5_file, "r") as f:
        monitor_labels = f.attrs["MonitorLabelList"]
        case_ids = f.attrs["CaseIdList"]
        sensor_idx = next(i for i, lbl in enumerate(monitor_labels) if lbl.decode("utf-8") == sensor)
        base_path = f"/{case_ids[sensor_idx].decode('utf-8')}"

        acc = f[f"{base_path}/Calibrated/Accelerometers"][:].T
        gyr = f[f"{base_path}/Calibrated/Gyroscopes"][:].T
        time = f[f"{base_path}/Time"][:]
        time_sec = (time - time[0]) / 1e6

    # 2. Extract Events from pre-loaded DF
    key = subject + timepoint
    subset = gaitrite_df[gaitrite_df["First_name"] == key].copy()
    task_map = {"SC": "Cont", "SI": "Interm"}
    subset = subset[subset["WalkTask"] == task_map.get(test_opal, "Cont")]

    events_df = pd.DataFrame({
        'IC': subset["FirstContact"].values,
        'FC': subset["LastContact"].values,
        'side': subset["Foot"].map({0: 'L', 1: 'R'}).values,
        'gait_id': subset["Gait_Id"].values
    })

    # Keep only ICs actually contained in the recording time axis
    t_min = time_sec[0]
    t_max = time_sec[-1]
    events_df = events_df[(events_df["IC"] >= t_min) & (events_df["FC"] <= t_max)].reset_index(drop=True)

    return time_sec, acc, gyr, events_df

def load_icicle_trial(
        h5_file: Path,
        sensor: str = "HD"
):
    """Loads IMU data and GaitRite events from a pre-loaded DataFrame."""
    fname = h5_file.name
    id_part = fname.split("-")[-1]
    subject_tp = id_part.split("_")[0]
    subject = subject_tp[:7]
    timepoint = subject_tp[7:]
    test_opal = id_part.split("_")[1][:2]

    # 1. Read HDF5 Data
    with h5py.File(h5_file, "r") as f:
        monitor_labels = f.attrs["MonitorLabelList"]
        case_ids = f.attrs["CaseIdList"]
        sensor_idx = next(i for i, lbl in enumerate(monitor_labels) if lbl.decode("utf-8") == sensor)
        base_path = f"/{case_ids[sensor_idx].decode('utf-8')}"

        acc = f[f"{base_path}/Calibrated/Accelerometers"][:].T
        gyr = f[f"{base_path}/Calibrated/Gyroscopes"][:].T
        time = f[f"{base_path}/Time"][:]
        time_sec = (time - time[0]) / 1e6

    return time_sec, acc, gyr

def preprocess_trial_data(h5_file: Path, gaitrite_df: pd.DataFrame, target_fs: float, h5_file_ss: Path = None):
    """Aligns, resamples, and prepares the trial for algorithms."""
    t, acc, gyr, events_df = load_icicle_trial_with_gait_events(h5_file, gaitrite_df)

    # Body frame convention
    # acc_bf = np.copy(acc.T)
    # gyr_bf = np.copy(gyr.T)
    # acc_bf[:, [0, 2]] = -acc.T[:, [0, 2]]
    # gyr_bf[:, [0, 2]] = -gyr.T[:, [0, 2]]
    # acc[:, [0, 2]] = -acc[:, [0, 2]]
    # gyr[:, [0, 2]] = -gyr[:, [0, 2]]
    acc[[0, 2], :] = -acc[[0, 2], :]
    gyr[[0, 2], :] = -gyr[[0, 2], :]

    # Resample
    time_rs = np.arange(t[0], t[-1], 1 / target_fs)
    imu_combined = np.concatenate((acc, gyr), axis=0)
    imu_rs = interp1d(t, imu_combined, axis=1, fill_value="extrapolate")(time_rs).T
    if h5_file_ss is None:
        # Align to Gravity (x-axis)
        acc_al, gyr_al, _ = align_imu_to_gravity(
            acc=imu_rs[:, :3], gyr=imu_rs[:, 3:],
            sampling_rate_hz=target_fs, static_duration_s=1.0,
            gravity_ideal=np.array([1.0, 0.0, 0.0])
        )
    else:
        t_ss, acc_ss, gyr_ss = load_icicle_trial(h5_file_ss)
        # Body frame convention
        acc_ss[[0, 2], :] = -acc_ss[[0, 2], :]
        # Align to Gravity with external file (x-axis)
        acc_al, gyr_al, _ = align_imu_to_gravity_with_external_file(
            acc_ss = acc_ss.T,
            acc=imu_rs[:, :3], gyr=imu_rs[:, 3:],
            sampling_rate_hz=target_fs, static_duration_s=1.0,
            gravity_ideal=np.array([1.0, 0.0, 0.0])
        )

    # # Debug plot to verify alignment
    # plt.figure(figsize=(10, 6))
    # colors = ['blue', 'orange', 'green']
    # labels = ['x', 'y', 'z']
    # for i in range(3):
    #     plt.plot(acc_al[:, i],
    #              color=colors[i],
    #              linestyle='-',
    #              label=f'acc_al {labels[i]}')
    #     plt.plot(imu_rs[:, i],
    #              color=colors[i],
    #              linestyle='--',
    #              label=f'imu_rs {labels[i]}')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()
    df_ready = pd.DataFrame(
        np.concatenate((acc_al, gyr_al), axis=1),
        columns=["acc_is", "acc_ml", "acc_ap", "gyr_is", "gyr_ml", "gyr_ap"]
    )
    return df_ready, events_df, time_rs


def match_and_validate(pred_times, pred_sides, ref_df, ev_type, tol=0.25):
    """
    Matches predicted events to reference events within valid gait intervals.
    Only predictions falling inside [First IC - tol, Last FC + tol] for each
    individual Gait_Id are considered.
    """
    # 1. Prepare reference data
    ref_times = ref_df[ev_type].values
    ref_sides = ref_df['side'].values

    # 2. Define valid sub-intervals for each gait_id
    # We use a boolean mask to filter only predictions inside valid walking zones
    mask = np.zeros(len(pred_times), dtype=bool)

    for gid in ref_df['gait_id'].unique():
        trial_subset = ref_df[ref_df['gait_id'] == gid]

        ic_values = np.sort(trial_subset['IC'].dropna().values)
        fc_values = np.sort(trial_subset['FC'].dropna().values)

        # Start of sub-interval: earliest IC in this gait segment - tol
        # End of sub-interval: latest FC in this gait segment + tol
        # start_lim = trial_subset['IC'].min() - tol
        # end_lim = trial_subset['FC'].max() + tol
        if len(ic_values) < 2 or len(fc_values) < 2:
            continue

        start_lim = ic_values[1] - tol
        end_lim = fc_values[-2] + tol
        mask |= (pred_times >= start_lim) & (pred_times <= end_lim)

    # Filter predictions
    p_t = pred_times[mask]
    p_s = pred_sides[mask]

    # 3. Initialize matching tables
    matches = []
    p_matched = np.zeros(len(p_t), dtype=bool)
    r_matched = np.zeros(len(ref_times), dtype=bool)

    # 4. Greedy temporal matching
    if len(p_t) > 0 and len(ref_times) > 0:
        # Distance matrix (Absolute time difference)
        dists = np.abs(p_t[:, None] - ref_times[None, :])

        # Find pairs within tolerance and sort by proximity (best matches first)
        pairs = np.argwhere(dists <= tol)
        distances = dists[pairs[:, 0], pairs[:, 1]]
        pairs = pairs[np.argsort(distances)]

        for p_idx, r_idx in pairs:
            if not p_matched[p_idx] and not r_matched[r_idx]:
                p_matched[p_idx] = r_matched[r_idx] = True
                matches.append({
                    'time_pred': p_t[p_idx],
                    'time_ref': ref_times[r_idx],
                    'side_pred': p_s[p_idx],
                    'side_ref': ref_sides[r_idx],
                    'diff': p_t[p_idx] - ref_times[r_idx],
                    'status': 'TP'
                })

    # 5. Handle False Positives (Extra predictions inside valid zones)
    for i in range(len(p_t)):
        if not p_matched[i]:
            matches.append({
                'time_pred': p_t[i], 'time_ref': -999,
                'side_pred': p_s[i], 'side_ref': 'N',
                'diff': -999, 'status': 'FP'
            })

    # 6. Handle False Negatives (Missed reference events)
    for i in range(len(ref_times)):
        if not r_matched[i]:
            matches.append({
                'time_pred': -999, 'time_ref': ref_times[i],
                'side_pred': 'N', 'side_ref': ref_sides[i],
                'diff': -999, 'status': 'FN'
            })

    m_df = pd.DataFrame(matches)
    m_df['type'] = ev_type
    return m_df


def plot_trial(time, acc_v, pred_ic, pred_fc, ref_df, path):
    """Generates the intermediate validation plot."""
    plt.figure(figsize=(12, 5))
    plt.plot(time, acc_v, 'k', alpha=0.5)
    for _, r in ref_df.iterrows():
        plt.axvline(r['IC'], color='g' if r['side'] == 'L' else 'r', ls='--', alpha=0.4)
        plt.axvline(r['FC'], color='g' if r['side'] == 'L' else 'r', ls=':', alpha=0.4)
    if not pred_ic.empty:
        plt.scatter(pred_ic['time'], np.interp(pred_ic['time'], time, acc_v), c='blue', marker='v', label='Pred IC')
    if not pred_fc.empty:
        plt.scatter(pred_fc['time'], np.interp(pred_fc['time'], time, acc_v), c='magenta', marker='^', label='Pred FC')
    plt.title(path.stem)
    plt.legend()
    # plt.savefig(path)
    plt.show()

def plot_global_errors(matches_df: pd.DataFrame, save_path: Path):
    """
    Figure 1: 2x2 Histograms of temporal errors (Pred - Ref)
    split by Event Type (IC/FC) and Side (L/R).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    tp_data = matches_df[matches_df["status"] == "TP"].copy()

    configs = [
        ('IC', 'L', 0, 0), ('IC', 'R', 0, 1),
        ('FC', 'L', 1, 0), ('FC', 'R', 1, 1)
    ]

    for ev_type, side, r, c in configs:
        ax = axes[r, c]
        subset = tp_data[(tp_data['type'] == ev_type) & (tp_data['side_ref'] == side)]

        if not subset.empty:
            sns.histplot(subset['diff'], ax=ax, kde=True, color='skyblue')
            ax.set_title(f"{side} {ev_type} Temporal Error")
            ax.set_xlabel("Error [s]")
            # Stats annotation
            mu, std = subset['diff'].mean(), subset['diff'].std()
            ax.text(0.05, 0.95, f"μ: {mu:.3f}s\nσ: {std:.3f}s", transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_global_boxplots(metrics_df: pd.DataFrame, save_path: Path):
    """
    Figure 2: 2x2 Boxplots showing Missed and Extra events %
    distribution across trials for IC and FC.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Missed %
    sns.boxplot(x='type', y='Missed%', data=metrics_df, ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Missed Events Percentage")

    # Extra %
    sns.boxplot(x='type', y='Extra%', data=metrics_df, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Extra Events Percentage")

    # F1 Score
    sns.boxplot(x='type', y='F1', data=metrics_df, ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("F1 Score Distribution")

    # MAE
    sns.boxplot(x='type', y='MAE', data=metrics_df, ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Mean Absolute Error [s]")

    for ax in axes.flatten():
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()