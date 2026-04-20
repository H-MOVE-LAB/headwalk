import matplotlib.pyplot as plt
from scipy.signal import freqz
import os
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from pathlib import Path


def load_head_data(filepath):
    """
    Loads Head IMU data (6 channels) and Walkway Contacts from CSV.
    Applies Mobilise-D convention.
    """
    # Define columns to load
    cols = [
        'L Foot Contact', 'R Foot Contact',
        'Forehead_Acc_X', 'Forehead_Acc_Y', 'Forehead_Acc_Z',
        'Forehead_Gyr_X', 'Forehead_Gyr_Y', 'Forehead_Gyr_Z'
    ]

    try:
        df = pd.read_csv(filepath, usecols=lambda c: c in cols)

        # Check if all columns exist
        if len(df.columns) < len(cols):
            # Try matching partial columns or return None
            return None, None, None, None

        # 1. Extract Input (Head IMU)
        acc_raw = df[['Forehead_Acc_X', 'Forehead_Acc_Y', 'Forehead_Acc_Z']].values
        gyr_raw = df[['Forehead_Gyr_X', 'Forehead_Gyr_Y', 'Forehead_Gyr_Z']].values

        # Mobilise-D convention transformation
        acc = np.copy(acc_raw)
        gyr = np.copy(gyr_raw)
        # Swap and invert based on convention
        # x (0): vertical (positive upwards)
        # y (1): mediolateral (positive rightwards)
        # z (2): anteroposterior (positive forwards)

        acc[:, 1] = acc_raw[:, 0]
        gyr[:, 1] = gyr_raw[:, 0]
        acc[:, 0] = -acc_raw[:, 1]
        gyr[:, 0] = -gyr_raw[:, 1]

        X = np.concatenate([acc, gyr], axis=-1)

        # 2. Extract Targets (Walkway -> IC/FC Indices)
        # We return the indices directly for matching logic
        # Contacts are binary or annotated? Assuming 0/1 for contact

        # Extract L and R contact arrays
        l_cont = df['L Foot Contact'].fillna(0).values
        r_cont = df['R Foot Contact'].fillna(0).values

        # Helper to find edges
        def get_edges(contact_arr):
            # IC: 0 -> 1
            ic_idx = np.where(np.diff(contact_arr, prepend=0) == 1)[0]
            # FC: 1 -> 0
            fc_idx = np.where(np.diff(contact_arr, prepend=0) == -1)[0] - 1
            return ic_idx, fc_idx

        l_ic, l_fc = get_edges(l_cont)
        r_ic, r_fc = get_edges(r_cont)

        # Pack GT into a dict for easier handling later
        Y = {
            'L_IC': l_ic, 'L_FC': l_fc,
            'R_IC': r_ic, 'R_FC': r_fc
        }

        return X, Y, l_cont, r_cont

    except Exception as e:
        print(f"[ERROR] Failed to load {os.path.basename(filepath)}: {e}")
        return None, None, None, None


def compute_quality_mask(X, l_cont, r_cont, gap_threshold=7, buffer=50):
    """
    Computes a boolean mask to identify valid regions for segmentation,
    excluding static periods and signal noise.

    Args:
        X (np.array): Head IMU data (N, 6).
        l_cont, r_cont (np.array): Binary walkway labels (1 if foot is on mat).
        gap_threshold (int): Max allowed width for NaNs in IMU data.
        buffer (int): Samples to remove at the edges of active regions.

    Returns:
        final_quality (np.array): Boolean mask of valid samples.
    """
    n_samples = X.shape[0]

    # --- 1. IMU Data Integrity ---
    # Identify rows containing at least one NaN
    is_nan = np.isnan(X).any(axis=1)
    # Label contiguous NaN regions to measure their duration
    labeled_nans, num_features = ndimage.label(is_nan)
    imu_quality = np.ones(n_samples, dtype=bool)

    if num_features > 0:
        slices = ndimage.find_objects(labeled_nans)
        for sl in slices:
            region = labeled_nans[sl]
            # Invalidate regions where the NaN gap is too long
            if len(region) >= gap_threshold:
                imu_quality[sl] = False

    # --- 2. Dynamic Walkway Activity ---
    # Detect transitions (0->1 or 1->0) for each foot to identify movement
    # This excludes periods where the subject is standing still on the mat
    l_trans = np.abs(np.diff(l_cont, prepend=l_cont[0]))
    r_trans = np.abs(np.diff(r_cont, prepend=r_cont[0]))

    # Combine all transitions from both feet
    all_transitions = np.logical_or(l_trans > 0, r_trans > 0)
    event_indices = np.where(all_transitions)[0]

    if len(event_indices) > 0:
        # Define the dynamic range from the very first to the very last event
        first_dynamic_event = event_indices[0]
        last_dynamic_event = event_indices[-1]

        movement_mask = np.zeros(n_samples, dtype=bool)
        movement_mask[first_dynamic_event: last_dynamic_event + 1] = True

        # Walkway is active only if there is movement AND the subject is on the mat
        walkway_on_mat = np.logical_or(l_cont == 1, r_cont == 1)
        walkway_active = np.logical_and(movement_mask, walkway_on_mat).astype(int)
    else:
        # No movement detected
        walkway_active = np.zeros(n_samples, dtype=int)

    walkway_quality = walkway_active.copy().astype(bool)

    # --- 3. Edge Buffering ---
    # Identify the start and end of each valid walking bout
    diff_transitions = np.diff(walkway_active, prepend=0)
    starts = np.where(diff_transitions == 1)[0]
    ends = np.where(diff_transitions == -1)[0]

    # Handle cases where the signal ends while the walkway is still active
    if len(walkway_active) > 0 and walkway_active[-1] == 1:
        ends = np.append(ends, n_samples)

    # Apply buffer to avoid edge effects (partial heel strikes)
    for s, e in zip(starts, ends):
        # Invalidate the first 'buffer' samples of the bout
        invalid_start_range = slice(s, min(s + buffer, n_samples))
        # Invalidate the last 'buffer' samples of the bout
        invalid_end_range = slice(max(0, e - buffer), e)

        walkway_quality[invalid_start_range] = False
        walkway_quality[invalid_end_range] = False

    # --- 4. Final Quality Fusion ---
    # Combine IMU integrity with movement-on-walkway quality
    final_quality = np.logical_and(imu_quality, walkway_quality)

    return final_quality

def rotate_to_gravity(accel_data, gyro_data):
    """
    Rotates the IMU data so that the gravity vector aligns with the
    Vertical axis (Y). Uses the average acceleration as the gravity reference.
    """
    # 1. Estimate gravity direction (mean of acceleration)
    gravity = np.mean(accel_data[1:100, :], axis=0)
    gravity_norm = gravity / np.linalg.norm(gravity)

    # 2. Target vertical axis (we'll use Y-axis: [0, -1, 0])
    vertical_target = np.array([1, 0, 0])

    # 3. Calculate rotation axis (cross product)
    rotation_axis = np.cross(gravity_norm, vertical_target)
    axis_norm = np.linalg.norm(rotation_axis)

    if axis_norm < 1e-6:
        # Already aligned
        return accel_data, gyro_data

    rotation_axis /= axis_norm

    # 4. Calculate rotation angle (dot product)
    angle = np.arccos(np.clip(np.dot(gravity_norm, vertical_target), -1.0, 1.0))

    # 5. Rodrigues' rotation formula components
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])

    # Rotation Matrix R
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # 6. Apply rotation to all samples
    accel_rotated = np.dot(accel_data, R.T)
    gyro_rotated = np.dot(gyro_data, R.T)

    return accel_rotated, gyro_rotated

def plot_filter_response(taps, fs, title="Filter Frequency Response"):
    # Calcola la risposta in frequenza (w è in radianti/campione)
    w, h = freqz(taps, worN=8000)

    # Converte w in Hz
    freq_hz = (w / np.pi) * (fs / 2)

    # Calcola la magnitudine in dB
    amplitude_db = 20 * np.log10(np.abs(h))

    # Calcola la fase in gradi
    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Magnitudine
    ax1.plot(freq_hz, amplitude_db, 'b', label='Magnitude (dB)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.grid(True)
    ax1.set_ylim([-100, 5])  # Limite tipico per vedere il ripple e la stopband

    # Plot Fase (su asse secondario)
    ax2 = ax1.twinx()
    ax2.plot(freq_hz, phase_deg, 'g--', label='Phase (degrees)')
    ax2.set_ylabel('Phase [deg]', color='g')

    plt.title(title)
    plt.show()

def plot_events_single_algo(target_fs, df, events, alg):
    time_rs = np.arange(0, df.shape[0], 1/target_fs)
    ax_ = "acc_is"
    plt.figure(figsize=(14, 5))
    plt.plot(time_rs, df[ax_], color="black", label="Vertical Acc")

    for i, t in enumerate(events[:, 0]):
        plt.axvline(x=t, color="red", linestyle="--", linewidth=2, label="GaitRite IC" if i == 0 else None)
    for i, t in enumerate(events[:, 1]):
        plt.axvline(x=t, color="red", linestyle=":", linewidth=2, label="GaitRite FC" if i == 0 else None)

    # ICD results
    ic_sec = alg.ic_list_["ic"].to_numpy() / target_fs
    plt.scatter(
        ic_sec,
        np.interp(ic_sec, time_rs, df[ax_]),
        marker="o",
        label="IC",
    )
    # FCD results
    fc_sec = alg.fc_list_["fc"].to_numpy() / target_fs
    plt.scatter(
        fc_sec,
        np.interp(fc_sec, time_rs, df[ax_]),
        marker="o",
        label="FC",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical Acceleration [m/s²]")
    plt.legend()
    plt.title("IC Detection Comparison (H-IMU)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_npz_trial(npz_path: str | Path):
    """
    Load a processed ICICLE trial saved as .npz.

    Returns
    -------
    time : np.ndarray
        Time vector in seconds
    data : np.ndarray
        IMU data (N x C)
    events : np.ndarray
        Gait events array [IC, FC]
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    return data["time"], data["data"], data["events"]