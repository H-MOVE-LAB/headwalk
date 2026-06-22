import sys
import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from pathlib import Path

# -----------------------------------------------------------------------------
# Function role: automatic project-root discovery
# -----------------------------------------------------------------------------
# This function makes the script portable across different Windows machines.
# It walks upward from the current file until it finds the project folder that
# contains both src/ and data/. This avoids hard-coded user-specific paths.

def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by moving upward from the current script.

    This avoids hard-coded user-specific paths such as:
    C:/Users/cesar/...
    C:/Utenti/mirko/...

    The only requirement is that the project contains both:
    - src/
    - data/
    """

    current = start_path.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "data").exists():
            return parent

    raise FileNotFoundError(
        "Project root not found. Expected folders: src/ and data/."
    )


PROJECT_ROOT = find_project_root(Path(__file__))

LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"

# Add the src folder to Python path before importing project-specific utilities.
# This makes imports robust even when the script is launched from a subfolder.
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.WearGaitPD_quality import (
    compute_weargait_quality_masks,
    save_weargait_quality_debug_plot
)

# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = PROJECT_ROOT / "data" / "WeargaitPD"

WINDOW_SIZE = 200
STEP_SIZE = 100
FS = 100

LOWPASS_CUTOFF_HZ = 15.0

# Minimum fraction required to assign a robust window-level path type.
# A window is considered straight or curved only if at least this fraction
# of the full window is assigned to that path type.
PATH_TYPE_MIN_FRACTION = 0.80

# Minimum fraction required to consider the final GSD window label robust.
# A window is considered robust only if the final static/walking label covers
# at least this fraction of the full window.
#
# This is safer than using only the walking/static ratio, because the ratio can
# become artificially high when a window contains many none samples and only a
# few walking samples.
WINDOW_LABEL_MIN_FRACTION = 0.80

# Minimum fraction required to mark a static subtype as relevant inside a window.
# This is not used to assign the final window label.
# It is used only to preserve contextual information in mixed windows.
#
# Example:
# a final walking window with 1 sitting sample should not be interpreted as a
# walking-to-sitting transition. However, a final walking window with 20% sitting
# samples should preserve this information for misclassification analysis.
STATIC_CONTEXT_MIN_FRACTION = 0.10

SAVE_FILTER_DEBUG_PLOTS = False
SAVE_QUALITY_DEBUG_PLOTS = False
MAX_DEBUG_PLOTS = 8

RESULTS_DIR = LOCAL_RESULTS_ROOT / "WearGaitPD_dataset"
FILTER_DEBUG_DIR = RESULTS_DIR / "filter_debug_plots"

debug_plot_count = 0

# GSD_LABEL_MAP is the only target used for binary Gait Sequence Detection:
# - none    = invalid / not usable for training
# - static  = non-walking
# - walking = any locomotor activity, including level walking and stairs
GSD_LABEL_MAP = {
    "none": -1,
    "static": 0,
    "walking": 1,
}

# ACTIVITY_DETAIL_MAP preserves the original activity type as metadata:
# - static, walking, ascending and descending remain distinguishable even if
#   the main GSD target is binary.
ACTIVITY_DETAIL_MAP = {
    "none": -1,
    "static": 0,
    "walking": 1,
    "ascending": 2,
    "descending": 3,
}

# STATIC_TYPE_MAP describes only the subtype of static samples:
# - none     = static subtype not applicable, mainly walking samples
# - static   = generic non-walking / pause / unknown static posture
# - standing = explicit standing condition
# - sitting  = explicit sitting condition, mainly relevant in TUG
STATIC_TYPE_MAP = {
    "none": -1,
    "static": 0,
    "standing": 1,
    "sitting": 2,
}

# GAIT_PHASE_MAP describes the temporal phase inside walking bouts.
GAIT_PHASE_MAP = {
    "none": -1,
    "gait_initiation": 0,
    "steady_state": 1,
    "gait_termination": 2,
}

# =====================================================
# TASK SELECTION
# =====================================================
# Keep only the tasks requested for the current WearGait-PD dataset version.
# TUG, selfpace_mat, hurriedpace_mat, selfpace_matturn, balance, freewalk.
# excluded: tandemgait, selfpace, hurriedpace, selfpace_doorpath
VALID_TASKS = {
    "tug",
    "selfpace_mat",
    "hurriedpace_mat",
    "selfpace_matturn",
    "balance",
    "freewalk"
}

# =====================================================
# WALKWAY QUALITY TASKS
# =====================================================
# WearGaitPD contains different task types. The walkway-quality mask should be
# applied only to tasks where the instrumented walkway contact information is
# meaningful for accepting/rejecting samples.
# In these tasks, L Foot Contact and R Foot Contact are used to check whether
# the subject is actually detected on the walkway during movement.
# Balance is excluded because it is mainly static.
# FreeWalk is excluded because it contains real-world/free-living portions where
# the walkway contact signal is not expected to describe the whole task.
WALKWAY_QUALITY_TASKS = {
    "tug",
    "selfpace_mat",
    "hurriedpace_mat",
    "selfpace_matturn"
}

# =====================================================
# STORAGE
# =====================================================
# Trial-level and window-level containers are initialized inside main().
# This prevents dataset construction from starting when the file is imported
# only to reuse helper functions.

# Extract subject ID from file name.
# -----------------------------------------------------------------------------
# Function role: subject identifier extraction
# -----------------------------------------------------------------------------
# This helper extracts the subject code directly from the CSV filename.
# It keeps the later metadata construction independent from folder names.

def get_subject_id(file_path):
    """
    Extract subject ID from WearGait-PD filename.
    Example: hc100_balance.csv -> hc100
    """
    return os.path.basename(file_path).split("_")[0].lower()

# -----------------------------------------------------------------------------
# Function role: task-name extraction from file name
# -----------------------------------------------------------------------------
# This helper centralizes the task-name extraction logic.
# The same logic is used during CSV deduplication and during the main processing
# loop, so duplicated files are detected using the same subject/task convention
# later stored in the metadata.

def get_task_name_from_file(file_path):
    """
    Extract the WearGait-PD task name from the CSV filename.

    Examples:
    - hc100_selfpace_mat.csv -> selfpace_mat
    - wpd015_freewalk.csv -> freewalk
    - nls145 (control)_freewalk.csv -> freewalk
    """

    subject_id = get_subject_id(file_path)

    task_name = (
        os.path.basename(file_path)
        .lower()
        .replace(".csv", "")
        .replace(subject_id + "_", "", 1)
    )

    return task_name

# Derive patient group (CONTROL or PD) from analyzed directory's name.
# -----------------------------------------------------------------------------
# Function role: diagnostic group extraction
# -----------------------------------------------------------------------------
# This helper infers whether the file belongs to PD or CONTROL participants
# by inspecting the folder path. The group is stored only as metadata.

def get_group_from_path(file_path):
    lower_path = file_path.lower() # Force lower characters in path name.
    if "pd participants" in lower_path:
        return "PD"
    elif "control participants" in lower_path:
        return "CONTROL"
    else:
        return "UNKNOWN"

# -----------------------------------------------------------------------------
# Function role: project-relative path construction
# -----------------------------------------------------------------------------
# This helper stores file provenance without making metadata dependent on a
# user-specific absolute Windows path.
#
# Absolute paths are useful during debugging, but metadata CSV files should remain
# portable across machines. Therefore, window-level metadata will store the path
# relative to PROJECT_ROOT.

def make_project_relative_path(file_path):
    """
    Convert an absolute file path into a project-relative path.

    If the file is not located inside PROJECT_ROOT, the function safely falls
    back to the file name only.
    """

    try:
        return str(
            Path(file_path).resolve().relative_to(PROJECT_ROOT.resolve())
        )

    except ValueError:
        return os.path.basename(file_path)

# -----------------------------------------------------------------------------
# Function role: detection of FreeWalk folder copies
# -----------------------------------------------------------------------------
# Some WearGait-PD CSV files (e.g. wpd015_freewalk.csv) may exist in more than one folder. For example, a
# FreeWalk file can be present both inside the specific FreeWalk directory and
# inside the generic PD/CONTROL participant directory. If both copies are scanned,
# the same trial is processed twice and all its windows become duplicated.

def path_contains_freewalk_folder(file_path):
    """
    Check whether a CSV file is located inside a folder named FreeWalk.
    """

    return any(
        str(part).lower() == "freewalk"
        for part in Path(file_path).parts
    )


# -----------------------------------------------------------------------------
# Function role: preferred-copy ranking for duplicated CSV files
# -----------------------------------------------------------------------------
# The ranking is used only when two or more CSV files represent the same logical
# trial. For FreeWalk tasks, the copy stored in the FreeWalk folder is preferred.
# For all other tasks, copies outside the FreeWalk folder are preferred.

def csv_deduplication_priority(file_path):
    """
    Return a deterministic priority tuple used to choose one CSV copy.

    Lower tuples are preferred by sorted().
    """

    task_name = get_task_name_from_file(file_path)
    is_freewalk_folder = path_contains_freewalk_folder(file_path)
    relative_path = make_project_relative_path(file_path).lower()

    if task_name == "freewalk":
        folder_priority = 0 if is_freewalk_folder else 1
    else:
        folder_priority = 0 if not is_freewalk_folder else 1

    return (
        folder_priority,
        len(relative_path),
        relative_path
    )


# -----------------------------------------------------------------------------
# Function role: input CSV deduplication before trial processing
# -----------------------------------------------------------------------------
# This function removes duplicated logical trials before any preprocessing or
# windowing step is performed. This is safer than removing duplicates only at the
# metadata level, because X, Y and metadata remain perfectly aligned.

def deduplicate_weargait_csv_files(csv_files):
    """
    Keep only one CSV file for each logical WearGait-PD trial.

    Logical key:
    - diagnostic group inferred from the path;
    - subject ID;
    - task name;
    - CSV file name.

    The returned duplicate_groups dictionary is used only for transparent debug
    printing, so the user can verify which physical copy was kept and which copy
    was discarded.
    """

    candidates_by_key = {}

    for file_path in csv_files:
        logical_key = (
            get_group_from_path(file_path),
            get_subject_id(file_path),
            get_task_name_from_file(file_path),
            os.path.basename(file_path).lower()
        )

        candidates_by_key.setdefault(logical_key, []).append(file_path)

    selected_files = []
    duplicate_groups = {}

    for logical_key, candidates in candidates_by_key.items():
        ranked_candidates = sorted(
            candidates,
            key=csv_deduplication_priority
        )

        selected_files.append(ranked_candidates[0])

        if len(ranked_candidates) > 1:
            duplicate_groups[logical_key] = ranked_candidates

    return sorted(selected_files), duplicate_groups

# -----------------------------------------------------------------------------
# Function role: FreeWalk stair direction assignment
# -----------------------------------------------------------------------------
# FreeWalk contains stair samples but the direction depends on the acquisition
# site. This function uses subject naming rules to split stair blocks into
# ascending and descending while excluding turn samples.

def assign_stair_ascent_descent(activity_detail, labels_raw, subject_id):
    """
    Assign ascending and descending stair activity details using the FreeWalk protocol.

    The stair order depends on the acquisition site:

    - NLS subjects:
      ascend first -> turn -> descend

    - W subjects:
      descend first -> turn -> ascend

    Turn samples remain marked as none in the activity-detail metadata.
    They are not used as an independent activity class.
    """

    # Identify stair and turn samples from the raw event annotations.
    stair_mask = labels_raw.str.contains("stair", na=False).to_numpy()
    turn_mask = labels_raw.str.contains("turn", na=False).to_numpy()

    if not np.any(stair_mask):
        return activity_detail

    # Extract indices of stair samples.
    stair_idx = np.where(stair_mask)[0]

    # Split stairs into contiguous blocks.
    split_points = np.where(np.diff(stair_idx) > 1)[0] + 1
    stair_blocks = np.split(stair_idx, split_points)

    # If only one stair block exists, assign according to subject site.
    if len(stair_blocks) == 1:

        if subject_id.lower().startswith("nls"):
            activity_detail[stair_blocks[0]] = ACTIVITY_DETAIL_MAP["ascending"]

        elif subject_id.lower().startswith("w"):
            activity_detail[stair_blocks[0]] = ACTIVITY_DETAIL_MAP["descending"]

        return activity_detail

    # NLS protocol:
    # ascent -> turn -> descent
    if subject_id.lower().startswith("nls"):

        activity_detail[stair_blocks[0]] = ACTIVITY_DETAIL_MAP["ascending"]

        for block in stair_blocks[1:]:
            activity_detail[block] = ACTIVITY_DETAIL_MAP["descending"]

    # W protocol:
    # descent -> turn -> ascent
    elif subject_id.lower().startswith("w"):

        activity_detail[stair_blocks[0]] = ACTIVITY_DETAIL_MAP["descending"]

        for block in stair_blocks[1:]:
            activity_detail[block] = ACTIVITY_DETAIL_MAP["ascending"]

    # Unknown protocol fallback.
    else:
        print(
            f"[WARNING] Unknown FreeWalk site for subject {subject_id}. "
            f"Using default ascent-first assumption."
        )

        activity_detail[stair_blocks[0]] = ACTIVITY_DETAIL_MAP["ascending"]

        for block in stair_blocks[1:]:
            activity_detail[block] = ACTIVITY_DETAIL_MAP["descending"]

    # Turn samples are not assigned to ascending or descending stairs here.
    # They remain none/unlabeled at HAR level unless another task-specific rule
    # explicitly treats them as walking.
    activity_detail[turn_mask] = ACTIVITY_DETAIL_MAP["none"]

    return activity_detail

# -----------------------------------------------------------------------------
# Function role: walkway contact-column discovery
# -----------------------------------------------------------------------------
# This helper searches the CSV header for foot-contact columns used to build
# Anderson-style walkway quality masks. If columns are missing, later logic
# can safely fall back to keeping all samples for walkway quality.

def find_walkway_contact_columns(columns):
    """
    Find the walkway foot-ground contact columns.

    WearGait-PD uses columns named 'L Foot Contact' and 'R Foot Contact'.
    These are the two Boolean signals used for Anderson's walkway quality mask.
    """

    contact_columns = []

    for col in columns:
        col_lower = col.lower().strip()

        if col_lower in ["l foot contact", "r foot contact"]:
            contact_columns.append(col)

    return contact_columns

# =====================================================
# BRODIE / BUCKLEY CONTINUOUS PITCH-ROLL CORRECTION
# =====================================================

# -----------------------------------------------------------------------------
# Function role: zero-phase low-pass filtering
# -----------------------------------------------------------------------------
# This function applies Butterworth filtering with filtfilt, so the filtered
# signal is smoothed without introducing temporal delay.

def apply_zero_phase_lowpass(data, fs, cutoff_hz, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter.

    This is used for:
    1. extraction of the low-frequency acceleration component (LFA);
    2. final smoothing of the corrected acceleration and gyroscope signals.

    Zero-phase filtering avoids temporal delay, which is important when
    windowing gait signals.
    """

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyquist

    b, a = butter(
        N=order,
        Wn=normalized_cutoff,
        btype="lowpass"
    )

    return filtfilt(b, a, data, axis=0)


# -----------------------------------------------------------------------------
# Function role: step-frequency estimation for Brodie/Buckley preprocessing
# -----------------------------------------------------------------------------
# This function estimates a dominant gait frequency from acceleration.
# The estimate is then used to choose the low-frequency cutoff for continuous
# pitch-roll correction.

def estimate_step_frequency_from_acc(acc, fs, vertical_axis_idx=0):
    """
    Estimate step frequency Fo from the acceleration signal.

    Brodie estimated Fo from the dominant harmonic in AP and VT acceleration.
    In this dataset, after gravity alignment, the vertical axis is axis 0.
    Since a true heading-based AP axis is not available here, the function
    combines:
    - the vertical axis;
    - the horizontal axis with the largest gait-band spectral power.

    This preserves the logic of using the most informative AP/VT-like
    components without assuming a perfectly known heading direction.
    """

    n = len(acc)

    if n < fs:
        return 1.8

    freqs, psd_all = welch(
        acc,
        fs=fs,
        nperseg=min(n, 1024),
        axis=0
    )

    gait_band_mask = (freqs >= 0.5) & (freqs <= 3.0)

    if not np.any(gait_band_mask):
        return 1.8

    # Choose the horizontal axis with the largest total gait-band power.
    horizontal_axes = [idx for idx in range(3) if idx != vertical_axis_idx]

    horizontal_powers = []
    for axis_idx in horizontal_axes:
        axis_power = np.nansum(psd_all[gait_band_mask, axis_idx])
        horizontal_powers.append(axis_power)

    best_horizontal_axis = horizontal_axes[int(np.argmax(horizontal_powers))]

    # Combine vertical + best horizontal component, as a practical AP/VT analogue.
    combined_psd = (
        psd_all[:, vertical_axis_idx] +
        psd_all[:, best_horizontal_axis]
    )

    gait_freqs = freqs[gait_band_mask]
    gait_psd = combined_psd[gait_band_mask]

    if len(gait_psd) == 0 or np.all(np.isnan(gait_psd)):
        return 1.8

    step_frequency_hz = gait_freqs[int(np.argmax(gait_psd))]

    return float(step_frequency_hz)


# -----------------------------------------------------------------------------
# Function role: vector-to-vector rotation computation
# -----------------------------------------------------------------------------
# This function computes the rotation matrix needed to align one vector to
# another. It is used sample-by-sample in the continuous tilt correction step.

def rotation_matrix_from_vectors(source_vector, target_vector):
    """
    Compute the proper rotation matrix that aligns source_vector to target_vector.

    This implements the dot-product / cross-product logic described in Brodie:
    the low-frequency acceleration vector is compared with the gravity vector,
    then a rotation is computed around the floating unit vector orthogonal to
    both vectors.

    The output is a true rotation matrix, so it does not use the invalid -I
    fallback for opposite vectors.
    """

    source = np.asarray(source_vector, dtype=float)
    target = np.asarray(target_vector, dtype=float)

    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)

    if source_norm < 1e-12 or target_norm < 1e-12:
        return np.eye(3)

    source = source / source_norm
    target = target / target_norm

    v = np.cross(source, target)
    c = np.dot(source, target)
    s = np.linalg.norm(v)

    # If vectors are already aligned, no correction is needed.
    if s < 1e-12 and c > 0:
        return np.eye(3)

    # If vectors are opposite, choose any stable orthogonal axis.
    if s < 1e-12 and c < 0:
        orthogonal = np.array([1.0, 0.0, 0.0])

        if abs(source[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0])

        v = np.cross(source, orthogonal)
        v = v / np.linalg.norm(v)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        # Rotation of pi radians.
        return np.eye(3) + 2 * (vx @ vx)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    return R

# -----------------------------------------------------------------------------
# Function role: full gap filling after gravity alignment
# -----------------------------------------------------------------------------
# This function creates a continuous numerical signal after gravity alignment.
#
# The goal is purely computational:
# - M3 correction requires low-pass filtering;
# - filtfilt cannot safely process NaN-containing signals;
# - therefore, remaining missing values must be filled before M3 and filtering.
#
# Important:
# this function does not decide whether interpolated samples are reliable.
# Reliability is still controlled later by imu_quality, walkway_quality and
# total_quality during window construction.

def fill_all_gaps_after_alignment_for_preprocessing(X_aligned):
    """
    Fill all remaining NaNs after gravity alignment so that M3 and low-pass
    filtering can be applied to the full trial.

    The returned signal is used only for numerical preprocessing.
    Quality masks still preserve the information about originally unreliable
    regions and are applied later at window level.
    """

    # Convert to DataFrame because pandas interpolation is convenient and works
    # independently on each signal channel.
    X_df = pd.DataFrame(X_aligned)

    # If one complete channel is missing, interpolation cannot reconstruct it.
    # In that case, the trial should be skipped instead of creating artificial data.
    if (X_df.notna().sum(axis=0) == 0).any():
        raise ValueError(
            "At least one IMU channel is entirely NaN. "
            "Full-trial interpolation is not possible."
        )

    # First attempt: cubic spline interpolation.
    # This provides a smooth reconstruction of missing samples.
    try:
        X_filled = X_df.interpolate(
            method="spline",
            order=3,
            axis=0,
            limit_direction="both"
        )

    except Exception:
        # If spline interpolation fails, for example because too few valid points
        # are available in a channel, fall back to linear interpolation.
        X_filled = X_df.interpolate(
            method="linear",
            axis=0,
            limit_direction="both"
        )

    # Safety fallback:
    # if residual NaNs remain after spline interpolation, use linear interpolation
    # plus backward/forward filling at signal borders.
    if X_filled.isna().any().any():
        X_filled = X_df.interpolate(
            method="linear",
            axis=0,
            limit_direction="both"
        )

        X_filled = X_filled.bfill().ffill()

    # If NaNs still remain, preprocessing cannot be performed safely.
    if X_filled.isna().any().any():
        raise ValueError(
            "NaNs remain after full gap filling. "
            "M3 and low-pass filtering cannot be applied safely."
        )

    return X_filled.to_numpy(dtype=float)

# -----------------------------------------------------------------------------
# Function role: gravity alignment with NaN-aware static reference
# -----------------------------------------------------------------------------
# This function applies the initial gravity alignment before any full-signal
# interpolation is performed.
#
# The rotation is estimated from the first static portion of the raw signal.
# If this portion contains NaNs, only finite accelerometer samples are used to
# estimate the mean gravity direction.
#
# Important:
# - NaNs are ignored only when estimating the static gravity vector.
# - The raw signal itself is not interpolated before alignment.
# - Rows containing NaNs remain NaN after rotation and will be handled later.
# - The quality masks still decide later which samples/windows are reliable.

def align_imu_to_gravity_omit_nan(
    acc,
    gyr,
    sampling_rate_hz,
    static_duration_s=1.0,
    target_axis=0,
    min_valid_static_samples=10
):
    """
    Align raw accelerometer and gyroscope signals to gravity using a NaN-aware
    static reference.

    The mean acceleration vector is computed from the first static_duration_s
    seconds, ignoring rows where at least one accelerometer axis is not finite.
    The resulting rotation is then applied to the full raw Acc and Gyr signals.

    The function does not fill missing samples. Missing rows remain missing
    after rotation and are filled later only for numerical preprocessing.
    """

    # Convert the static reference duration from seconds to samples.
    n_static = int(static_duration_s * sampling_rate_hz)

    # Avoid requesting more static samples than the trial actually contains.
    n_static = min(n_static, len(acc))

    if n_static <= 0:
        raise ValueError(
            "Static reference duration is zero or the trial is empty."
        )

    # Extract the initial reference segment used to estimate gravity direction.
    acc_reference = acc[:n_static]

    # Keep only rows where all three accelerometer axes are finite.
    # This is the equivalent of an omit-NaN strategy for gravity estimation.
    finite_reference_mask = np.isfinite(acc_reference).all(axis=1)

    # If too few finite samples are available, gravity cannot be estimated
    # reliably and the trial should be skipped.
    if np.sum(finite_reference_mask) < min_valid_static_samples:
        raise ValueError(
            "Not enough finite samples in the initial static reference "
            "to estimate gravity alignment."
        )

    # Compute the mean gravity direction using only finite samples.
    # NaN rows are ignored here, but the raw signal is not interpolated yet.
    mean_static_acc_raw = np.mean(
        acc_reference[finite_reference_mask],
        axis=0
    )

    # Check that the estimated gravity vector is numerically valid.
    gravity_norm = np.linalg.norm(mean_static_acc_raw)

    if not np.isfinite(gravity_norm) or gravity_norm < 1e-12:
        raise ValueError(
            "Invalid gravity vector estimated from the initial static reference."
        )

    # Build the target gravity vector.
    # In this pipeline convention, gravity is aligned with axis 0.
    target_vector = np.zeros(3)
    target_vector[target_axis] = gravity_norm

    # Compute the rotation that aligns the measured gravity vector to the
    # target gravity direction.
    R = rotation_matrix_from_vectors(
        source_vector=mean_static_acc_raw,
        target_vector=target_vector
    )

    # Apply the same rigid rotation to accelerometer and gyroscope signals.
    # Matrix multiplication preserves NaNs in rows where the original signal
    # was missing, because interpolation is intentionally not applied here.
    acc_aligned = acc @ R.T
    gyr_aligned = gyr @ R.T

    # Estimate the aligned gravity vector from the raw mean vector.
    # This vector is later used as trial-specific gravity reference for M3.
    mean_static_acc_aligned = R @ mean_static_acc_raw

    return (
        acc_aligned,
        gyr_aligned,
        R,
        mean_static_acc_raw,
        mean_static_acc_aligned
    )


# -----------------------------------------------------------------------------
# Function role: full-trial M3 and low-pass preprocessing
# -----------------------------------------------------------------------------
# This function applies Brodie/Buckley M3 correction and final low-pass filtering
# to the full trial after gravity alignment and total-signal-gap-filling.
# Quality masks are intentionally not used here.
# They are applied later during window construction, so preprocessing remains
# independent from the decision of which samples/windows are finally trusted.

def apply_preprocessing_full_trial_after_alignment(
    acc_aligned_filled,
    gyr_aligned_filled,
    fs,
    gravity_vector,
    lowpass_cutoff_hz
):
    """
    Apply M3 correction and final low-pass filtering to the full aligned trial.

    Input signals must be:
    - already gravity-aligned;
    - free from NaNs after interpolation.

    Output signals have the same length as the original trial and will be
    filtered later at window level using the stored quality masks.
    """

    # M3 and filtfilt cannot be applied safely if NaNs are still present.
    if np.isnan(acc_aligned_filled).any() or np.isnan(gyr_aligned_filled).any():
        raise ValueError(
            "Full-trial preprocessing received NaN values. "
            "Run full gap filling after gravity alignment first."
        )

    # Apply Brodie/Buckley continuous tilt correction.
    # This estimates a low-frequency acceleration vector and uses it to correct
    # slow pitch/roll drift before subtracting the gravity vector.
    acc_linear, _, _, _ = apply_brodie_continuous_tilt_correction(
        acc=acc_aligned_filled,
        fs=fs,
        gravity_vector=gravity_vector,
        vertical_axis_idx=0
    )

    # Apply final zero-phase low-pass filtering to linear acceleration.
    # Zero-phase filtering avoids temporal delay in the final windowed signal.
    acc_final = apply_zero_phase_lowpass(
        data=acc_linear,
        fs=fs,
        cutoff_hz=lowpass_cutoff_hz,
        order=4
    )

    # Apply the same final low-pass filtering to the gyroscope signal.
    gyr_final = apply_zero_phase_lowpass(
        data=gyr_aligned_filled,
        fs=fs,
        cutoff_hz=lowpass_cutoff_hz,
        order=4
    )

    return acc_final, gyr_final


# -----------------------------------------------------------------------------
# Function role: continuous pitch-roll correction
# -----------------------------------------------------------------------------
# This function implements the Brodie/Buckley-style correction. It estimates
# slow orientation drift from low-frequency acceleration and removes gravity
# using a trial-specific gravity vector.
# Brodie's method does not simply subtract a fixed [9.81, 0, 0] vector.
# First, it estimates a low-frequency acceleration vector (LFA) using
# a low-pass filter with cutoff Fo/4.
#
# The LFA vector is interpreted as gravity plus slow orientation drift.
# A sample-wise rotation is then computed to align LFA with the gravity
# direction. This continuously corrects pitch and roll.

def apply_brodie_continuous_tilt_correction(
    acc,
    fs,
    gravity_vector,
    vertical_axis_idx=0
):
    """
    Apply Brodie/Buckley continuous pitch-roll correction.

    Methodological steps:
    1. Estimate step frequency Fo from acceleration.
    2. Set the low-pass cutoff to Fo/4.
    3. Low-pass filter the complete acceleration signal to obtain LFA.
    4. At each sample, compute the rotation aligning LFA to the gravity vector.
    5. Apply this rotation to the measured acceleration, obtaining ACorr.
    6. Compute linear acceleration by subtracting the estimated gravity vector.

    Important:
    - gravity_vector is NOT hardcoded as [9.81, 0, 0].
    - It is estimated from the aligned static reference of the current trial.
    - This makes the correction robust to units, sign and axis convention.
    """

    step_frequency_hz = estimate_step_frequency_from_acc(
        acc=acc,
        fs=fs,
        vertical_axis_idx=vertical_axis_idx
    )

    m3_cutoff_hz = step_frequency_hz / 4.0

    # Safety bounds avoid unstable low-pass filtering in noisy/non-walking files.
    m3_cutoff_hz = float(np.clip(m3_cutoff_hz, 0.15, 1.0))

    lfa = apply_zero_phase_lowpass(
        data=acc,
        fs=fs,
        cutoff_hz=m3_cutoff_hz,
        order=4
    )

    acc_corrected = np.zeros_like(acc)

    for i in range(len(acc)):
        R_i = rotation_matrix_from_vectors(
            source_vector=lfa[i],
            target_vector=gravity_vector
        )

        acc_corrected[i] = R_i @ acc[i]

    # Brodie computes linear acceleration by subtracting gravity from ACorr.
    # Here gravity is estimated from the current trial instead of being hardcoded.
    acc_linear = acc_corrected - gravity_vector

    return acc_linear, acc_corrected, step_frequency_hz, m3_cutoff_hz

# -----------------------------------------------------------------------------
# Function role: activity-detail-to-binary-GSD conversion
# -----------------------------------------------------------------------------
# This function creates the main model target from activity-detail metadata.
# Static remains static. Walking, ascending and descending are merged into walking.
# Samples marked as none remain none and are not valid training targets.

def convert_activity_detail_to_gsd(activity_detail):
    """
    Convert activity-detail labels into binary GSD labels.

    Final binary GSD labels:
    - none    = -1
    - static  = 0
    - walking = 1

    Ascending and descending are merged into walking because the GSD task is
    binary: static versus locomotion.
    """

    labels_gsd = np.full(
        len(activity_detail),
        GSD_LABEL_MAP["none"],
        dtype=np.int32
    )

    labels_gsd[
        activity_detail == ACTIVITY_DETAIL_MAP["static"]
    ] = GSD_LABEL_MAP["static"]

    labels_gsd[
        activity_detail == ACTIVITY_DETAIL_MAP["walking"]
    ] = GSD_LABEL_MAP["walking"]

    labels_gsd[
        activity_detail == ACTIVITY_DETAIL_MAP["ascending"]
    ] = GSD_LABEL_MAP["walking"]

    labels_gsd[
        activity_detail == ACTIVITY_DETAIL_MAP["descending"]
    ] = GSD_LABEL_MAP["walking"]

    return labels_gsd

# -----------------------------------------------------------------------------
# Function role: numeric label-to-name conversion
# -----------------------------------------------------------------------------
# This utility converts integer labels back to readable names for metadata.

def get_map_name(map_dict, value):
    """
    Convert numeric map value into its string name.
    """

    for key, val in map_dict.items():
        if val == value:
            return key

    return "unknown"

# -----------------------------------------------------------------------------
# Function role: robust window-level path-type summarization
# -----------------------------------------------------------------------------
# This helper converts sample-wise path-type labels into robust window-level
# metadata. A window is assigned to straight or curved only if that path type
# covers at least a predefined fraction of the full window.

def summarize_window_path_type(path_win, window_size, min_fraction):
    """
    Summarize straight/curved path composition inside one window.

    Path-type convention:
    - 0    = straight
    - 1    = curved
    - None = path type not applicable, usually static or unlabeled samples

    The straight/curved fractions are computed over the full window length,
    not only over valid locomotor path samples. This is intentionally stricter:
    it prevents a mixed static/walking transition window from being marked as
    straight or curved only because its short locomotor portion is pure.

    Output:
    - final_path_type:
        0 if straight fraction >= min_fraction
        1 if curved fraction >= min_fraction
        None otherwise
    - path-type counters and fractions for transparent metadata inspection.
    """

    path_values = list(path_win)

    n_straight_path_samples = int(
        sum(value == 0 for value in path_values)
    )

    n_curved_path_samples = int(
        sum(value == 1 for value in path_values)
    )

    n_none_path_samples = int(
        sum(value is None for value in path_values)
    )

    n_other_path_samples = int(
        window_size
        - n_straight_path_samples
        - n_curved_path_samples
        - n_none_path_samples
    )

    straight_path_fraction = n_straight_path_samples / window_size
    curved_path_fraction = n_curved_path_samples / window_size
    none_path_fraction = n_none_path_samples / window_size

    path_type_confidence = max(
        straight_path_fraction,
        curved_path_fraction
    )

    if straight_path_fraction >= min_fraction:
        final_path_type = 0
        is_path_type_robust = True

    elif curved_path_fraction >= min_fraction:
        final_path_type = 1
        is_path_type_robust = True

    else:
        final_path_type = None
        is_path_type_robust = False

    return {
        "final_path_type": final_path_type,
        "n_straight_path_samples": n_straight_path_samples,
        "n_curved_path_samples": n_curved_path_samples,
        "n_none_path_samples": n_none_path_samples,
        "n_other_path_samples": n_other_path_samples,
        "straight_path_fraction": straight_path_fraction,
        "curved_path_fraction": curved_path_fraction,
        "none_path_fraction": none_path_fraction,
        "path_type_confidence": path_type_confidence,
        "is_path_type_robust": is_path_type_robust,
    }

# -----------------------------------------------------------------------------
# Function role: static-subtype composition inside one window
# -----------------------------------------------------------------------------
# This helper preserves standing/sitting information even in mixed windows.
# The final window static_type can still remain label-consistent, but this
# summary keeps track of how much standing or sitting is present in the window.

def summarize_window_static_context(static_type_win, window_size, min_fraction):
    """
    Summarize static-subtype composition inside one window.

    Static-type convention:
    - -1 = none
    -  0 = generic static
    -  1 = standing
    -  2 = sitting

    This function does not change the final GSD label.
    It only creates contextual metadata useful for debugging and
    misclassification analysis.

    A static subtype is considered relevant only if it covers at least
    min_fraction of the full window.
    An example of a possible window configuration label is:
    label_name = walking
    static_type_name = none
    walking_fraction = 0.70
    sitting_static_type_fraction = 0.30
    static_context_type_name = sitting
    has_sitting_context = True
    Thanks to it, it is possible to distinguish between a pure walking window and
    a walking one with a certain percentage of static samples, specifying its static_type.
    """

    n_none_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["none"])
    )

    n_generic_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["static"])
    )

    n_standing_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["standing"])
    )

    n_sitting_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["sitting"])
    )

    generic_static_type_fraction = (
        n_generic_static_type_samples / window_size
    )

    standing_static_type_fraction = (
        n_standing_static_type_samples / window_size
    )

    sitting_static_type_fraction = (
        n_sitting_static_type_samples / window_size
    )

    none_static_type_fraction = (
        n_none_static_type_samples / window_size
    )

    candidate_contexts = {
        STATIC_TYPE_MAP["static"]: generic_static_type_fraction,
        STATIC_TYPE_MAP["standing"]: standing_static_type_fraction,
        STATIC_TYPE_MAP["sitting"]: sitting_static_type_fraction,
    }

    static_context_type = max(
        candidate_contexts,
        key=candidate_contexts.get
    )

    static_context_fraction = candidate_contexts[static_context_type]

    if static_context_fraction < min_fraction:
        static_context_type = STATIC_TYPE_MAP["none"]
        static_context_fraction = 0.0
        has_static_context = False
    else:
        has_static_context = True

    has_standing_context = (
        standing_static_type_fraction >= min_fraction
    )

    has_sitting_context = (
        sitting_static_type_fraction >= min_fraction
    )

    return {
        "n_none_static_type_samples": n_none_static_type_samples,
        "n_generic_static_type_samples": n_generic_static_type_samples,
        "n_standing_static_type_samples": n_standing_static_type_samples,
        "n_sitting_static_type_samples": n_sitting_static_type_samples,
        "none_static_type_fraction": none_static_type_fraction,
        "generic_static_type_fraction": generic_static_type_fraction,
        "standing_static_type_fraction": standing_static_type_fraction,
        "sitting_static_type_fraction": sitting_static_type_fraction,
        "static_context_min_fraction": min_fraction,
        "static_context_type": static_context_type,
        "static_context_type_name": get_map_name(
            STATIC_TYPE_MAP,
            static_context_type
        ),
        "static_context_fraction": static_context_fraction,
        "has_static_context": has_static_context,
        "has_standing_context": has_standing_context,
        "has_sitting_context": has_sitting_context,
    }

# -----------------------------------------------------------------------------
# Function role: optional filtering debug visualization
# -----------------------------------------------------------------------------
# This function saves before/after filtering plots for a small number of files.
# It is disabled by default so full dataset construction remains fast.

def save_filter_debug_plot(acc_before, acc_after, fs, subject_id, task_name, file_name):
    """
    Save a short visual comparison before/after M3-like filtering.
    """

    global debug_plot_count

    if not SAVE_FILTER_DEBUG_PLOTS:
        return

    if debug_plot_count >= MAX_DEBUG_PLOTS:
        return

    n_samples = min(len(acc_before), int(10 * fs))
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axis_names = ["AP", "ML", "VT"]

    for axis_idx, ax in enumerate(axes):
        ax.plot(t, acc_before[:n_samples, axis_idx], label="Before high-pass", alpha=0.7)
        ax.plot(t, acc_after[:n_samples, axis_idx], label="After high-pass", alpha=0.7)
        ax.set_ylabel(f"Acc {axis_names[axis_idx]}")
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")

    fig.suptitle(f"Filtering debug | {subject_id} | {task_name} | {file_name}")

    safe_name = f"{debug_plot_count:02d}_{subject_id}_{task_name}_{file_name}"
    safe_name = safe_name.replace(".csv", ".png")
    safe_name = safe_name.replace(" ", "_").replace("(", "").replace(")", "")

    output_path = os.path.join(FILTER_DEBUG_DIR, safe_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    debug_plot_count += 1

    print("[FILTER DEBUG] Saved plot:", output_path)

# -----------------------------------------------------------------------------
# Function role: turn-interval mask construction from GeneralEvent annotations
# -----------------------------------------------------------------------------
# This helper detects turning samples from WearGaitPD GeneralEvent annotations.
#
# It supports two possible annotation styles:
# 1. continuous turn labels, where all samples inside the turn contain "turn";
# 2. start/end labels, where only boundaries are marked as turn_start / turn_end.
#
# The returned mask is used to:
# - treat TUG turn portions as locomotor samples;
# - assign path_type = curved to locomotor samples inside turns.

def build_turn_mask_from_general_event(labels_raw):
    """
    Build a Boolean mask identifying samples belonging to a turn.
    This mask is particularly relevant for TUG-GSD_detection.

    The function first marks every sample whose GeneralEvent contains "turn".
    Then, if explicit start/end markers are available, it also fills the interval
    between each turn start and the corresponding turn end.
    """

    labels = labels_raw.astype(str).str.lower()

    # Basic case:
    # if turn is written continuously over the whole turning interval, this is
    # already sufficient.
    turn_mask = labels.str.contains(
        "turn",
        na=False,
        regex=True
    ).to_numpy()

    # More specific case:
    # some annotation styles may store only turn-start and turn-end events.
    turn_start_mask = labels.str.contains(
        r"turn[_\s-]*start|start[_\s-]*turn|turn[_\s-]*begin|begin[_\s-]*turn",
        na=False,
        regex=True
    ).to_numpy()

    turn_end_mask = labels.str.contains(
        r"turn[_\s-]*end|end[_\s-]*turn|turn[_\s-]*stop|stop[_\s-]*turn",
        na=False,
        regex=True
    ).to_numpy()

    start_indices = np.where(turn_start_mask)[0]
    end_indices = np.where(turn_end_mask)[0]

    # If start/end markers exist, fill each interval from start to the next end.
    for start_idx in start_indices:

        candidate_ends = end_indices[end_indices >= start_idx]

        if len(candidate_ends) == 0:
            continue

        end_idx = candidate_ends[0]

        turn_mask[start_idx:end_idx + 1] = True

    return turn_mask

# -----------------------------------------------------------------------------
# Function role: task-specific raw event label assignment
# -----------------------------------------------------------------------------
# This function converts GeneralEvent strings into activity-detail labels.
# The output is not the final GSD training target.
# The final binary GSD label is built later by convert_activity_detail_to_gsd().

def assign_task_activity_detail(labels_raw, task_name, subject_id):
    """
    Create sample-wise activity labels according to the selected WearGait-PD tasks.

    Main labels:
    - none
    - static
    - walking
    - ascending
    - descending

    Task-specific logic:
    - TUG: standing/sitting as static, walk as walking.
    - Balance: standing as static.
    - SelfPace_mat and HurriedPace_mat: walk as walking.
    - SelfPace_matturn: walk and turn as walking, with turn represented in path_type.
    - FreeWalk: standing as static, walk as walking, stairs split into ascending/descending.
    """

    # Initialize every sample as none.
    # Only samples explicitly matching the expected annotations for each task
    # will be converted into static, walking, ascending or descending.
    activity_detail = np.full(len(labels_raw), ACTIVITY_DETAIL_MAP["none"], dtype=np.int32)

    # Detect standing-like annotations.
    # These samples are considered static when the task expects standing periods.
    standing_mask = labels_raw.str.contains(
        "stand|standing|feet|eo_|ec_",
        na=False,
        regex=True
    ).to_numpy()

    # Detect sitting-like annotations.
    # This is mainly useful for TUG, where sitting is a valid static condition.
    sitting_mask = labels_raw.str.contains(
        "sit|sitting|chair|seated",
        na=False,
        regex=True
    ).to_numpy()

    # Detect walking and turning annotations.
    # Turns are treated differently depending on the task.
    walk_mask = labels_raw.str.contains("walk", na=False, regex=True).to_numpy()

    # Detect turn samples or turn intervals.
    # This is especially important for TUG, where the turning portion is still
    # locomotion and should be treated as walking in the binary GSD target.
    turn_mask = build_turn_mask_from_general_event(labels_raw)

    # Detect stair annotations.
    # These are expected mainly in FreeWalk files.
    stair_mask = labels_raw.str.contains("stair", na=False, regex=True).to_numpy()

    # TUG contains walking and static phases.
    # Both standing and sitting (| is OR) are assigned to activity_detail = static.
    # The distinction between standing and sitting will not be lost since it is
    # separately saved into static_type flag.
    # Thus, they are marked using the index defined inside ACTIVITY_DETAIL_MAP.
    if task_name == "tug":
        activity_detail[standing_mask | sitting_mask] = ACTIVITY_DETAIL_MAP["static"]

        # TUG contains straight walking and a turning portion.
        # Both are locomotor samples for binary GSD, so both must become walking.
        # The straight/curved distinction is preserved later through path_type.
        activity_detail[walk_mask | turn_mask] = ACTIVITY_DETAIL_MAP["walking"]

    # Balance is used as a static task.
    # Only standing-like samples are retained.
    elif task_name == "balance":
        activity_detail[standing_mask] = ACTIVITY_DETAIL_MAP["static"]

    # SelfPace_mat and HurriedPace_mat contain straight walking.
    # Gait initiation and termination are not separate labels here:
    # they will be stored later as gait_phase metadata.
    elif task_name in ["selfpace_mat", "hurriedpace_mat"]:
        activity_detail[walk_mask] = ACTIVITY_DETAIL_MAP["walking"]

    # SelfPace_matturn contains walking plus turning.
    # Turns are kept as walking in the main label, while the curved/straight
    # distinction remains stored separately in path_type.
    elif task_name == "selfpace_matturn":
        activity_detail[walk_mask | turn_mask] = ACTIVITY_DETAIL_MAP["walking"]

    # FreeWalk may contain standing, walking and stairs.
    # Stairs are split into ascending and descending using the task sequence.
    elif task_name == "freewalk":
        activity_detail[standing_mask] = ACTIVITY_DETAIL_MAP["static"]
        activity_detail[walk_mask] = ACTIVITY_DETAIL_MAP["walking"]

    if np.any(stair_mask):
        # Stair direction depends on the acquisition site.
        # NLS subjects: ascend first, then descend.
        # W subjects: descend first, then ascend.
        #
        # The function modifies and returns activity_detail.
        # No separate label vector is created here, because the binary GSD target
        # will be built later by convert_activity_detail_to_gsd().
        activity_detail = assign_stair_ascent_descent(
            activity_detail=activity_detail,
            labels_raw=labels_raw,
            subject_id=subject_id
        )

    return activity_detail

# -----------------------------------------------------------------------------
# Function role: static-subtype metadata assignment
# -----------------------------------------------------------------------------
# This function assigns a more specific static subtype when possible.
#
# Logic:
# - standing-like annotations are marked as standing
# - sitting-like annotations are marked as sitting
# - generic static samples are marked as static
# - walking, ascending, descending and none samples remain none

def assign_static_type(labels_raw, task_name, activity_detail):
    """
    Create sample-wise static-type metadata.

    Output metadata:
    - none
    - static
    - standing
    - sitting

    This is not the main class label.
    It only describes the type of static posture when this information is available.
    """

    static_type = np.full(
        len(labels_raw),
        STATIC_TYPE_MAP["none"],
        dtype=np.int32
    )

    # Static type is meaningful only for samples whose activity detail is static.
    # Walking, ascending, descending and none samples remain static_type = none.
    static_samples = activity_detail == ACTIVITY_DETAIL_MAP["static"]

    standing_mask = labels_raw.str.contains(
        "stand|standing|balance|feet|eo_|ec_",
        na=False,
        regex=True
    ).to_numpy()

    sitting_mask = labels_raw.str.contains(
        "sit|sitting|chair|seated",
        na=False,
        regex=True
    ).to_numpy()

    generic_static_mask = labels_raw.str.contains(
        "static",
        na=False,
        regex=True
    ).to_numpy()

    # Generic static assignment.
    static_type[
        static_samples & generic_static_mask
    ] = STATIC_TYPE_MAP["static"]

    # Explicit standing assignment.
    static_type[
        static_samples & standing_mask
    ] = STATIC_TYPE_MAP["standing"]

    # Explicit sitting assignment.
    # This is especially relevant for TUG, where sitting and standing are both
    # part of the static/non-walking class but should remain distinguishable.
    static_type[
        static_samples & sitting_mask
    ] = STATIC_TYPE_MAP["sitting"]

    # Any remaining sample labelled as static in activity_detail but not matched
    # by a posture-specific keyword is treated as generic static.
    unresolved_static = (
        static_samples
        &
        (static_type == STATIC_TYPE_MAP["none"])
    )

    static_type[unresolved_static] = STATIC_TYPE_MAP["static"]

    return static_type

# -----------------------------------------------------------------------------
# Function role: gait-phase metadata assignment
# -----------------------------------------------------------------------------
# Unlike INDIP/POLITO datasets, WearGaitPD keeps whole walking bouts. Therefore
# gait initiation and gait termination can be meaningful and are marked here.

def assign_gait_phase(labels_num, task_name, fs, transient_duration_s=1.0):
    """
    Create sample-wise gait phase metadata.

    For WearGait-PD, gait initiation and gait termination are preserved
    because walking periods are not cropped as in INDIP-based datasets.

    The first second of each walking block is marked as gait_initiation.
    The last second of each walking block is marked as gait_termination.
    The central part is marked as steady_state.

    This is not the main class label.
    The main label remains walking, while this metadata specifies whether a walking
    sample belongs to the beginning, middle or end of a walking bout.
    """
    # Initialize every sample as none.
    # Non-walking samples will remain none.
    gait_phase = np.full(len(labels_num), GAIT_PHASE_MAP["none"], dtype=np.int32)

    # Gait phase is useful only in WearGait-PD walking tasks where the transient
    # phases are still present in the signal.
    if task_name not in ["selfpace_mat", "hurriedpace_mat", "selfpace_matturn"]:
        return gait_phase

    # Identify samples whose main label is walking.
    # These samples will be divided into initiation, steady-state and termination.
    # labels_num is the binary GSD label vector (0 = static, 1 = walking).
    # Therefore, walking samples must be identified using GSD_LABEL_MAP,
    # not ACTIVITY_DETAIL_MAP.
    walking_mask = labels_num == GSD_LABEL_MAP["walking"]
    walking_idx = np.where(walking_mask)[0]

    # If the task has no walking samples, no gait phase can be assigned.
    if len(walking_idx) == 0:
        return gait_phase

    # Split walking samples into contiguous walking blocks.
    # This is important because one trial may contain more than one walking segment.
    split_points = np.where(np.diff(walking_idx) > 1)[0] + 1
    walking_blocks = np.split(walking_idx, split_points)

    # Convert transient duration from seconds to samples.
    # Example: 1.0 s at 100 Hz corresponds to 100 samples.
    transient_samples = int(transient_duration_s * fs)

    # Process each walking block independently.
    for block in walking_blocks:

        # Skip empty blocks, only as a safety check.
        if len(block) == 0:
            continue

        # If the walking block is shorter than initiation + termination,
        # it cannot contain a reliable steady-state portion.
        if len(block) <= 2 * transient_samples:

            # Split the short block into two halves:
            # first half = gait initiation, second half = gait termination.
            midpoint = len(block) // 2

            gait_phase[block[:midpoint]] = GAIT_PHASE_MAP["gait_initiation"]
            gait_phase[block[midpoint:]] = GAIT_PHASE_MAP["gait_termination"]

        else:
            # First part of the walking block.
            # These samples correspond to the transition from static/non-walking
            # into walking, so they are marked as gait initiation.
            gait_phase[block[:transient_samples]] = GAIT_PHASE_MAP["gait_initiation"]

            # Middle part of the walking block.
            # These samples are far from the start and the end of the walking bout,
            # so they are marked as steady-state walking.
            gait_phase[block[transient_samples:-transient_samples]] = GAIT_PHASE_MAP["steady_state"]

            # Last part of the walking block.
            # These samples correspond to the transition from walking back to
            # static/non-walking, so they are marked as gait termination.
            gait_phase[block[-transient_samples:]] = GAIT_PHASE_MAP["gait_termination"]

    return gait_phase

# -----------------------------------------------------------------------------
# Function role: walking-bout metadata construction
# -----------------------------------------------------------------------------
# WearGaitPD does not provide INDIP ContinuousWalkingPeriod segment IDs.
# However, for metadata consistency with INDIVI, TOWalk and MOVEWISE, a local
# walking-bout identifier can be derived from contiguous regions of binary
# walking labels.
#
# Samples outside walking bouts receive -1.
# Samples inside the first walking block receive 0, the second walking block
# receive 1, and so on.
#
# These IDs are local to each trial/file and must not be interpreted as global
# subject-level or dataset-level identifiers.

def build_sample_walking_bouts(labels_gsd):
    """
    Build sample-wise walking-bout IDs from binary GSD labels.

    Output:
    - -1 for samples outside walking bouts;
    -  0, 1, 2, ... for contiguous walking blocks inside the same trial.
    """

    walking_bouts = np.full(
        len(labels_gsd),
        -1,
        dtype=np.int32
    )

    walking_mask = labels_gsd == GSD_LABEL_MAP["walking"]
    walking_indices = np.where(walking_mask)[0]

    if len(walking_indices) == 0:
        return walking_bouts

    # Split walking samples into contiguous blocks.
    split_points = np.where(np.diff(walking_indices) > 1)[0] + 1
    walking_blocks = np.split(walking_indices, split_points)

    for bout_id, block in enumerate(walking_blocks):
        walking_bouts[block] = bout_id

    return walking_bouts

# -----------------------------------------------------------------------------
# Function role: window-level binary transition-type detection
# -----------------------------------------------------------------------------
# This function assigns a transition metadata label to one window.
#
# Important:
# this function must work on binary GSD labels, not on HAR labels.
#
# Expected input:
# y_win = window of binary labels
#
# Label convention:
# 0 = static
# 1 = walking
# -1 = none
#
# Logic:
# 1. split the window into two temporal halves;
# 2. compute the dominant valid GSD label in the first half;
# 3. compute the dominant valid GSD label in the second half;
# 4. compare the two dominant labels.
#
# Possible outputs:
# - "Static2Walking" if the window starts mainly as static and ends mainly as walking
# - "Walking2Static" if the window starts mainly as walking and ends mainly as static
# - "None" if the dominant class does not change or the transition cannot be estimated
# This is only metadata. It does not change the final window label.

def assign_transition_type_majority(y_win):
    """
    Assign window-level transition metadata using binary GSD majority voting.

    The window is split into two halves:
    - first half: approximate starting state
    - second half: approximate ending state

    If the dominant binary class changes from the first half to the second half,
    a transition type is stored.
    """

    midpoint = len(y_win) // 2

    first_half = y_win[:midpoint]
    second_half = y_win[midpoint:]

    # Exclude invalid samples from both halves.
    # Only static and walking samples are allowed to vote.
    first_valid = first_half[first_half != GSD_LABEL_MAP["none"]]
    second_valid = second_half[second_half != GSD_LABEL_MAP["none"]]

    # If one half contains only none samples, the transition cannot be
    # estimated reliably.
    if len(first_valid) == 0 or len(second_valid) == 0:
        return "None"

    # Majority voting on the first half.
    first_values, first_counts = np.unique(
        first_valid,
        return_counts=True
    )

    start_label = int(
        first_values[np.argmax(first_counts)]
    )

    # Majority voting on the second half.
    second_values, second_counts = np.unique(
        second_valid,
        return_counts=True
    )

    end_label = int(
        second_values[np.argmax(second_counts)]
    )

    # If the dominant class does not change, this is not a transition window.
    if start_label == end_label:
        return "None"

    # Explicitly return only binary GSD transitions.
    if (
        start_label == GSD_LABEL_MAP["static"]
        and
        end_label == GSD_LABEL_MAP["walking"]
    ):
        return "Static2Walking"

    if (
        start_label == GSD_LABEL_MAP["walking"]
        and
        end_label == GSD_LABEL_MAP["static"]
    ):
        return "Walking2Static"

    # Safety fallback.
    # This should normally not happen if y_win contains only binary GSD labels.
    return "None"

# -----------------------------------------------------------------------------
# Function role: window-level dataset construction and saving
# -----------------------------------------------------------------------------
# This function takes the preprocessed flat trial dataset and generates fixed-
# length windows. For each window, it assigns labels, quality metadata, path
# type, static type, gait phase, activity detail and transition type.

def create_window_dataset_from_trials(trial_dataset, window_size, step_size, output_path):
    """
    Create a window-level dataset from the already preprocessed trial-level dataset.

    This avoids reloading and preprocessing the raw CSV files every time a new
    window length is needed.
    """
    X_windows = []
    Y_windows = []
    metadata_windows = []

    for trial in trial_dataset:
        X = trial["X"]
        labels = trial["labels"]
        activity_detail = trial["activity_detail"]
        path_types = trial["path_types"]
        static_type = trial["static_type"]
        gait_phase = trial["gait_phase"]
        walkway_quality = trial["walkway_quality"]
        imu_quality = trial["imu_quality"]
        total_quality = trial["total_quality"]
        walking_bouts = trial["walking_bouts"]

        for i in range(0, len(X) - window_size + 1, step_size):
            x_win = X[i:i + window_size]
            y_win = labels[i:i + window_size]
            activity_detail_win = activity_detail[i:i + window_size]
            walking_bouts_win = walking_bouts[i:i + window_size]

            # ------------------------------------------------------------
            # Window-level quality-mask application
            # ------------------------------------------------------------
            # Quality masks are applied only at this stage, after the full
            # preprocessing pipeline has already been completed.
            # For walkway-based tasks, the window is accepted only if all
            # its samples satisfy total_quality:
            # total_quality = imu_quality AND walkway_quality.
            # For non-walkway tasks, only IMU quality is used. This avoids
            # incorrectly rejecting FreeWalk or Balance windows using a
            # walkway signal that is not meaningful for those tasks.
            if trial["use_walkway_quality"]:
                quality_win = total_quality[i:i + window_size]
            else:
                quality_win = imu_quality[i:i + window_size]

            # Reject the window if at least one sample is marked as low quality.
            # This is where the quality mask actually affects the dataset.
            if not np.all(quality_win):
                continue

            # Identify only valid GSD samples.
            # Samples marked as none are ignored during window-label majority voting.
            valid_labels = y_win[y_win != GSD_LABEL_MAP["none"]]

            # ------------------------------------------------------------
            # Transition type metadata
            # ------------------------------------------------------------
            # Transition type is computed using majority voting on the
            # first and second half of the window.
            # Example: Static2Walking, Walking2Static.
            transition_type = assign_transition_type_majority(y_win)

            # Skip only windows that contain no valid GSD samples.
            if len(valid_labels) == 0:
                continue

            unique_vals, counts = np.unique(valid_labels, return_counts=True)
            final_label = int(unique_vals[np.argmax(counts)])

            # ------------------------------------------------------------
            # Robust window annotation for static/walking transitions
            # ------------------------------------------------------------
            # The final window label is still assigned by majority voting.
            # However, transition windows may contain a mixed amount of static
            # and walking samples.
            #
            # The aim is to avoid ambiguous windows during training.
            # Therefore, for each window we compute:
            # - number of static samples;
            # - number of walking samples;
            # - walking/static ratio;
            # - robust-window flag.
            #
            # Robust windows:
            # - walking_static_ratio >= 4.0  -> at least about 80% walking
            # - walking_static_ratio <= 0.25 -> at least about 80% static
            #
            # Ambiguous windows:
            # - 0.25 < walking_static_ratio < 4.0
            #
            # These windows are kept in the dataset but can be excluded from training
            # while still being used later as challenging validation/test examples.
            n_static_samples = int(
                np.sum(y_win == GSD_LABEL_MAP["static"])
            )

            n_walking_samples = int(
                np.sum(y_win == GSD_LABEL_MAP["walking"])
            )

            n_none_label_samples = int(
                np.sum(y_win == GSD_LABEL_MAP["none"])
            )

            # ------------------------------------------------------------
            # Robust GSD window composition
            # ------------------------------------------------------------
            # Robustness is now computed from explicit full-window fractions,
            # not from the walking/static ratio alone.
            #
            # This avoids a common edge case:
            # a window with many none samples and only a few walking samples
            # could have a very high walking/static ratio only because the
            # number of static samples is zero. Such a window must not be marked
            # as a clean robust walking window.
            static_fraction = n_static_samples / window_size
            walking_fraction = n_walking_samples / window_size
            none_label_fraction = n_none_label_samples / window_size

            valid_label_fraction = (
                n_static_samples + n_walking_samples
            ) / window_size

            if final_label == GSD_LABEL_MAP["static"]:
                final_label_fraction = static_fraction

            elif final_label == GSD_LABEL_MAP["walking"]:
                final_label_fraction = walking_fraction

            else:
                final_label_fraction = 0.0

            # Keep the old ratio as backward-compatible diagnostic metadata.
            # It is no longer used to decide whether the window is robust.
            eps = 1e-6

            walking_static_ratio = (
                (n_walking_samples + eps)
                /
                (n_static_samples + eps)
            )

            is_robust_window = (
                final_label_fraction >= WINDOW_LABEL_MIN_FRACTION
            )

            # No minimum label-fraction filter is applied here.
            # Windows with a low percentage of static or walking samples are
            # intentionally kept in the dataset and marked as non-robust.
            #
            # This is useful for later validation and misclassification analysis,
            # because ambiguous windows can still be inspected instead of being
            # silently removed during dataset construction.

            if np.isnan(x_win).any():
                continue

            # ------------------------------------------------------------
            # Label-consistent window-level metadata assignment
            # ------------------------------------------------------------
            # Sample-wise metadata are intentionally rich and may contain both
            # static and locomotor information inside a mixed window.
            # Window-level metadata, however, must remain semantically consistent
            # with the final binary GSD label assigned by majority voting.
            #
            # Therefore:
            # - if final_label is walking, static_type must be none;
            # - if final_label is static, path_type and gait_phase must be none;
            # - path_type is computed only from meaningful locomotor samples;
            # - static_type is computed only for windows whose final label is static.
            #
            # This prevents cases such as:
            # label_name = walking with static_type_name = standing/sitting.
            path_win = path_types[i:i + window_size]
            static_type_win = static_type[i:i + window_size]
            gait_phase_win = gait_phase[i:i + window_size]

            # ------------------------------------------------------------
            # Static-subtype composition
            # ------------------------------------------------------------
            # The final static_type remains label-consistent, but the internal
            # standing/sitting composition is preserved for later analysis.
            #
            # This is essential for mixed transition windows, for example:
            # - walking windows that contain a relevant sitting portion;
            # - static windows that contain a small walking portion.
            static_context_summary = summarize_window_static_context(
                static_type_win=static_type_win,
                window_size=window_size,
                min_fraction=STATIC_CONTEXT_MIN_FRACTION
            )

            # ------------------------------------------------------------
            # Robust window-level path-type composition
            # ------------------------------------------------------------
            # Path type is computed as a window-level metadata field using a
            # minimum fraction threshold.
            #
            # A window is assigned to:
            # - straight only if at least PATH_TYPE_MIN_FRACTION of the full
            #   window is straight;
            # - curved only if at least PATH_TYPE_MIN_FRACTION of the full
            #   window is curved;
            # - none otherwise.
            #
            # The window is not discarded when the threshold is not reached.
            # Instead, path_type remains none and the path-type percentages are
            # saved in metadata for later debugging, filtering or analysis.
            path_summary = summarize_window_path_type(
                path_win=path_win,
                window_size=window_size,
                min_fraction=PATH_TYPE_MIN_FRACTION
            )

            if final_label == GSD_LABEL_MAP["walking"]:
                # Path type is meaningful only for locomotor windows.
                # Unlike simple majority voting, the window-level path type is
                # assigned only when straight or curved covers at least the
                # required fraction of the full window.
                #
                # Ambiguous walking windows are kept, but their path_type remains
                # none. Their detailed straight/curved percentages are still
                # stored in metadata.
                final_path_type = path_summary["final_path_type"]

                # Static subtype is not applicable to walking windows.
                # Static samples inside mixed windows are already documented by:
                # n_static_samples, n_walking_samples and walking_static_ratio.
                final_static_type = STATIC_TYPE_MAP["none"]

                # Gait phase is meaningful only for walking windows.
                valid_gait_phase = gait_phase_win[
                    gait_phase_win != GAIT_PHASE_MAP["none"]
                ]

                if len(valid_gait_phase) > 0:
                    unique_phase, phase_counts = np.unique(
                        valid_gait_phase,
                        return_counts=True
                    )

                    final_gait_phase = int(
                        unique_phase[np.argmax(phase_counts)]
                    )

                else:
                    final_gait_phase = GAIT_PHASE_MAP["none"]

                # Activity detail must also be consistent with a walking GSD label.
                # Static detail is excluded here because static posture is not the
                # final window class.
                locomotor_details = activity_detail_win[
                    (activity_detail_win == ACTIVITY_DETAIL_MAP["walking"])
                    |
                    (activity_detail_win == ACTIVITY_DETAIL_MAP["ascending"])
                    |
                    (activity_detail_win == ACTIVITY_DETAIL_MAP["descending"])
                ]

                if len(locomotor_details) > 0:
                    unique_detail, detail_counts = np.unique(
                        locomotor_details,
                        return_counts=True
                    )

                    final_activity_detail = int(
                        unique_detail[np.argmax(detail_counts)]
                    )

                else:
                    final_activity_detail = ACTIVITY_DETAIL_MAP["walking"]

                # Walking-bout ID is meaningful only for walking windows.
                # Ignore -1 values because they represent samples outside walking
                # bouts and should not dominate a walking window.
                valid_bouts = walking_bouts_win[walking_bouts_win >= 0]

                if len(valid_bouts) > 0:
                    unique_bouts, bout_counts = np.unique(
                        valid_bouts,
                        return_counts=True
                    )

                    final_walking_bout = int(
                        unique_bouts[np.argmax(bout_counts)]
                    )

                else:
                    final_walking_bout = -1

            elif final_label == GSD_LABEL_MAP["static"]:

                # Path type and gait phase are locomotor metadata.
                # They are not applicable to static windows.
                final_path_type = None
                final_gait_phase = GAIT_PHASE_MAP["none"]
                final_walking_bout = -1

                # Static type is meaningful only for static windows.
                valid_static_type = static_type_win[
                    static_type_win != STATIC_TYPE_MAP["none"]
                ]

                if len(valid_static_type) > 0:
                    unique_static, static_counts = np.unique(
                        valid_static_type,
                        return_counts=True
                    )

                    final_static_type = int(
                        unique_static[np.argmax(static_counts)]
                    )

                else:
                    final_static_type = STATIC_TYPE_MAP["static"]

                # The binary GSD label is static. The detailed activity label
                # should therefore remain static at window level.
                final_activity_detail = ACTIVITY_DETAIL_MAP["static"]

            else:

                # This branch should rarely be reached because windows with no
                # valid GSD samples are skipped earlier. It is kept as a safety
                # fallback for metadata consistency.
                final_path_type = None
                final_static_type = STATIC_TYPE_MAP["none"]
                final_gait_phase = GAIT_PHASE_MAP["none"]
                final_activity_detail = ACTIVITY_DETAIL_MAP["none"]
                final_walking_bout = -1

            X_windows.append(x_win)
            Y_windows.append(final_label)

            metadata_windows.append({
                "dataset": trial["dataset"],
                "subject_id": trial["subject_id"],
                "task": trial["task"],
                "test_type": trial["test_type"],
                "challenge_type": trial.get("challenge_type", "none"),
                "file_name": trial["file_name"],
                "relative_file_path": trial["relative_file_path"],
                "window_start_sample": i,
                "window_end_sample": i + window_size,
                "window_duration_s": window_size / trial["fs"],
                "n_static_samples": int(n_static_samples),
                "n_walking_samples": int(n_walking_samples),
                "n_none_label_samples": int(n_none_label_samples),
                "static_fraction": float(static_fraction),
                "walking_fraction": float(walking_fraction),
                "none_label_fraction": float(none_label_fraction),
                "valid_label_fraction": float(valid_label_fraction),
                "final_label_fraction": float(final_label_fraction),
                "label_min_fraction": float(WINDOW_LABEL_MIN_FRACTION),
                "walking_static_ratio": float(walking_static_ratio),
                "is_robust_window": bool(is_robust_window),
                "label": final_label,
                "label_name": get_map_name(GSD_LABEL_MAP, final_label),
                "activity_detail": final_activity_detail,
                "activity_detail_name": get_map_name(ACTIVITY_DETAIL_MAP, final_activity_detail),
                "path_type": final_path_type,
                "path_type_name": trial["path_type_map"].get(final_path_type, "none"),
                "path_type_min_fraction": float(PATH_TYPE_MIN_FRACTION),
                "n_straight_path_samples": int(path_summary["n_straight_path_samples"]),
                "n_curved_path_samples": int(path_summary["n_curved_path_samples"]),
                "n_none_path_samples": int(path_summary["n_none_path_samples"]),
                "n_other_path_samples": int(path_summary["n_other_path_samples"]),
                "straight_path_fraction": float(path_summary["straight_path_fraction"]),
                "curved_path_fraction": float(path_summary["curved_path_fraction"]),
                "none_path_fraction": float(path_summary["none_path_fraction"]),
                "path_type_confidence": float(path_summary["path_type_confidence"]),
                "is_path_type_robust": bool(path_summary["is_path_type_robust"]),
                "static_type": final_static_type,
                "static_type_name": [k for k, v in STATIC_TYPE_MAP.items() if v == final_static_type][0],

                "static_context_min_fraction": float(
                    static_context_summary["static_context_min_fraction"]
                ),
                "n_none_static_type_samples": int(
                    static_context_summary["n_none_static_type_samples"]
                ),
                "n_generic_static_type_samples": int(
                    static_context_summary["n_generic_static_type_samples"]
                ),
                "n_standing_static_type_samples": int(
                    static_context_summary["n_standing_static_type_samples"]
                ),
                "n_sitting_static_type_samples": int(
                    static_context_summary["n_sitting_static_type_samples"]
                ),
                "none_static_type_fraction": float(
                    static_context_summary["none_static_type_fraction"]
                ),
                "generic_static_type_fraction": float(
                    static_context_summary["generic_static_type_fraction"]
                ),
                "standing_static_type_fraction": float(
                    static_context_summary["standing_static_type_fraction"]
                ),
                "sitting_static_type_fraction": float(
                    static_context_summary["sitting_static_type_fraction"]
                ),
                "static_context_type": int(
                    static_context_summary["static_context_type"]
                ),
                "static_context_type_name": static_context_summary[
                    "static_context_type_name"
                ],
                "static_context_fraction": float(
                    static_context_summary["static_context_fraction"]
                ),
                "has_static_context": bool(
                    static_context_summary["has_static_context"]
                ),
                "has_standing_context": bool(
                    static_context_summary["has_standing_context"]
                ),
                "has_sitting_context": bool(
                    static_context_summary["has_sitting_context"]
                ),
                "transition_type": transition_type,
                "walking_bout": final_walking_bout,
                "gait_phase": final_gait_phase,
                "gait_phase_name": [k for k, v in GAIT_PHASE_MAP.items() if v == final_gait_phase][0],
                "group": trial["group"],
            })
    X_windows = np.array(X_windows, dtype=np.float32)
    Y_windows = np.array(Y_windows, dtype=np.int32)
    metadata_windows = np.array(metadata_windows, dtype=object)

    window_bundle = {
        "DatasetName": "WearGaitPD",
        "WindowSize": window_size,
        "StepSize": step_size,
        "PathTypeMinFraction": PATH_TYPE_MIN_FRACTION,
        "label_map": GSD_LABEL_MAP,  # backward-compatible alias.
        "gsd_label_map": GSD_LABEL_MAP,
        "activity_detail_map": ACTIVITY_DETAIL_MAP,
        "static_type_map": STATIC_TYPE_MAP,
        "gait_phase_map": GAIT_PHASE_MAP,
        "path_type_map": {
            0: "straight",
            1: "curved",
            None: "none",
        },
        "X": X_windows,
        "Y": Y_windows,
        "metadata": metadata_windows,
    }

    with open(output_path, "wb") as f:
        pickle.dump(window_bundle, f)

    # Save readable metadata in CSV format using the same base name as the PKL file.
    # The PKL contains X, Y and metadata together, while the CSV is useful for
    # manual inspection, debugging and dataset summaries.
    output_path = Path(output_path)
    metadata_csv_path = output_path.with_suffix(".csv")

    metadata_df = pd.DataFrame(list(metadata_windows))

    # ------------------------------------------------------------
    # Semantic consistency checks for window-level metadata
    # ------------------------------------------------------------
    # These checks guard against metadata leakage across domains:
    # - static_type must describe only windows whose final label is static;
    # - path_type must describe only windows whose final label is walking.
    #
    # If one of these checks fails, the construction is stopped because the
    # resulting metadata would be misleading for thesis analysis and debugging.
    walking_with_static_type = (
        (metadata_df["label_name"] == "walking")
        &
        (metadata_df["static_type_name"] != "none")
    )

    static_with_path_type = (
        (metadata_df["label_name"] == "static")
        &
        (metadata_df["path_type_name"] != "none")
    )

    if walking_with_static_type.any() or static_with_path_type.any():

        debug_inconsistency_path = (
            metadata_csv_path.with_name(
                metadata_csv_path.stem + "_SEMANTIC_INCONSISTENCIES.csv"
            )
        )

        metadata_df[
            walking_with_static_type | static_with_path_type
        ].to_csv(
            debug_inconsistency_path,
            index=False
        )

        raise RuntimeError(
            "Semantic inconsistency detected in window-level metadata. "
            "A debug CSV was saved to: "
            f"{debug_inconsistency_path}"
        )

    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"[SUCCESS] Metadata CSV saved: {metadata_csv_path}")

    print(f"[SUCCESS] Window dataset saved: {output_path}")
    print("X shape:", X_windows.shape)
    print("Y shape:", Y_windows.shape)


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
# This block scans every selected CSV file, loads raw IMU data, applies quality
# checks, performs gap filling, gravity alignment, Brodie/Buckley preprocessing,
# builds sample-wise labels and appends one trial dictionary to trial_dataset.
# The output of this block is still trial-level: windowing is performed later.
# =============================================================================

def main():
    """
    Execute the complete WearGaitPD construction pipeline.

    The pipeline:
    1. loops over all selected CSV files
    2. preprocesses each trial
    3. builds the flat trial-level dataset
    4. saves the trial dataset
    5. builds and saves window datasets
    """

    # =====================================================
    # MAIN LOOP
    # =====================================================

    global debug_plot_count

    # Create output folders only when the script is executed directly.
    # This avoids filesystem side effects when the module is imported only to
    # reuse helper functions from another script.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FILTER_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize the trial-level container inside main().
    # This prevents stale data from remaining in memory across repeated imports
    # or interactive runs.
    trial_dataset = []

    # Discover CSV files only during explicit script execution.
    # Importing this file will therefore not scan the dataset folder and will not
    # start any dataset-construction step.
    raw_csv_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
        if os.path.basename(f).lower() != "manifest.csv"
    ]

    raw_csv_files = sorted(raw_csv_files)

    # ------------------------------------------------------------
    # Input-level CSV deduplication
    # ------------------------------------------------------------
    # The recursive scan can find the same logical trial in two physical folders.
    # This happened, for example, for wpd015_freewalk.csv, which was present both
    # inside the specific FreeWalk folder and inside the generic PD participant
    # folder. Without this block, the trial is preprocessed twice and all its
    # generated windows appear twice in the final metadata and X/Y arrays.
    csv_files, duplicate_csv_groups = deduplicate_weargait_csv_files(
        raw_csv_files
    )

    freewalk_csv_files = [
        f for f in csv_files
        if "freewalk" in os.path.basename(f).lower()
    ]

    print(f"[INFO] Searching CSV files inside: {DATA_DIR}")
    print(f"[INFO] Total CSV files found: {len(csv_files)}")
    print(f"[INFO] FreeWalk CSV files found inside DATA_DIR: {len(freewalk_csv_files)}")

    if len(csv_files) > 0:
        print("[INFO] First CSV example:", csv_files[0])
    else:
        raise FileNotFoundError("No CSV files found. Check DATA_DIR.")

    # Count balance files after CSV discovery. This is only a diagnostic summary
    # and should not run during module import.
    balance_files = {
        get_subject_id(f): f
        for f in csv_files
        if os.path.basename(f).lower().endswith("_balance.csv")
    }

    print(f"[INFO] Balance files found: {len(balance_files)}")

    for file_id, fpath in enumerate(csv_files, start=1):

        print(f"{file_id}/{len(csv_files)} -> {os.path.basename(fpath)}")

        # For every .csv file, get subject id and task name from the file name.
        subject_id = get_subject_id(fpath)

        # Extract task name from the file name.
        # Example: hc100_selfpace_mat.csv -> selfpace_mat
        # The same helper is used during input deduplication, so the logical
        # identity of each trial is consistent across the whole pipeline.
        task_name = get_task_name_from_file(fpath)

        # Keep only the selected WearGait-PD tasks.
        if task_name not in VALID_TASKS:
            continue

        print(f"[INFO] Selected task: {task_name} | File: {os.path.basename(fpath)}")

        # The read columns are: GeneralEvent, the tri-axial
        # forehead_acc and tri-axial forehead_gyr.
        try:
            # Read only the header first to detect optional walkway contact columns.
            header_columns = pd.read_csv(fpath, nrows=0).columns.tolist()

            '''debug_walkway_cols = [
                col for col in header_columns
                if any(key in col.lower() for key in [
                    "walkway", "foot", "left", "right", "contact", "pressure", "force"
                ])
            ]

            print("\n[DEBUG WALKWAY COLUMNS]", os.path.basename(fpath))
            for col in debug_walkway_cols:
                print(" -", col)'''

            walkway_contact_columns = find_walkway_contact_columns(header_columns)

            # Store the actual column names found in the CSV header.
            # This keeps the code robust to harmless spacing/case differences in
            # the source files while still using the expected WearGait-PD fields.
            l_contact_col = next(
                (
                    col for col in walkway_contact_columns
                    if col.lower().strip() == "l foot contact"
                ),
                None
            )

            r_contact_col = next(
                (
                    col for col in walkway_contact_columns
                    if col.lower().strip() == "r foot contact"
                ),
                None
            )

            # Walkway-based tasks require both contact columns because their
            # quality mask depends on L/R foot-ground contact. If one column is
            # missing, the file is skipped instead of silently accepting windows.
            if task_name in WALKWAY_QUALITY_TASKS and (
                l_contact_col is None or r_contact_col is None
            ):
                print(
                    f"[WARNING] Skipped file {os.path.basename(fpath)} | "
                    "Reason: missing L/R Foot Contact columns required for walkway quality."
                )
                continue

            usecols = [
                          "GeneralEvent",
                          "Forehead_Acc_X", "Forehead_Acc_Y", "Forehead_Acc_Z",
                          "Forehead_Gyr_X", "Forehead_Gyr_Y", "Forehead_Gyr_Z"
                      ] + walkway_contact_columns

            df = pd.read_csv(fpath, usecols=usecols, low_memory=False)

        except Exception as e:
            print(f"[WARNING] Skipped file {os.path.basename(fpath)} | Reason: {e}")
            continue

        # ---------------------------------------------
        # Load IMU
        # ---------------------------------------------
        acc = df[[
            "Forehead_Acc_X",
            "Forehead_Acc_Y",
            "Forehead_Acc_Z"
        ]].values

        gyr = df[[
            "Forehead_Gyr_X",
            "Forehead_Gyr_Y",
            "Forehead_Gyr_Z"
        ]].values

        # ------------------------------------------------------------
        # Gravity alignment before full-signal interpolation
        # ------------------------------------------------------------
        # The initial gravity alignment is performed on the raw IMU signal.
        # Missing samples are ignored only when estimating the mean gravity
        # vector from the first static second.
        #
        # This prevents interpolated samples from influencing the initial
        # reference-frame alignment.
        static_duration_s = 1.0
        alignment_reference = "first_1s_raw_omit_nan"

        try:
            (
                acc_aligned_raw,
                gyr_aligned_raw,
                R,
                mean_static_acc_raw,
                mean_static_acc
            ) = align_imu_to_gravity_omit_nan(
                acc=acc,
                gyr=gyr,
                sampling_rate_hz=FS,
                static_duration_s=static_duration_s,
                target_axis=0
            )

        except ValueError as e:
            print(
                f"[WARNING] Skipped file {os.path.basename(fpath)} | "
                f"Reason: gravity alignment failed before interpolation. {e}"
            )
            continue

        # ------------------------------------------------------------
        # Gravity alignment check
        # ------------------------------------------------------------
        # After rotation, the estimated static acceleration should be aligned
        # with axis 0. The normalized mean is used only as a diagnostic check.
        mean_static_acc_norm = mean_static_acc / np.linalg.norm(mean_static_acc)

        if np.isnan(mean_static_acc).any() or np.isnan(mean_static_acc_norm).any():
            print(
                f"[WARNING] Skipped file {os.path.basename(fpath)} | "
                "Reason: gravity alignment produced NaN values."
            )
            continue

        # Alignment debug can be useful when checking preprocessing quality,
        # but it is commented during full dataset construction to reduce console output.

        print(f"\n[ALIGNMENT CHECK] Subject {subject_id} | File: {os.path.basename(fpath)}")
        print("Alignment reference:", alignment_reference)
        print("Raw mean static acceleration:", mean_static_acc_raw)
        print("Aligned mean static acceleration:", mean_static_acc)
        print("Normalized mean aligned static acceleration:", mean_static_acc_norm)

        # ------------------------------------------------------------
        # Full gap filling after gravity alignment
        # ------------------------------------------------------------
        # At this point, the raw IMU signal has already been rotated into the
        # gravity-aligned frame.
        # Remaining NaNs are now filled only to make the signal numerically
        # compatible with M3 correction and zero-phase filtering.
        # This interpolation does not make the samples automatically reliable:
        # imu_quality, walkway_quality and total_quality are still stored and
        # will be applied later during window construction.
        X_aligned_raw = np.concatenate(
            [acc_aligned_raw, gyr_aligned_raw],
            axis=1
        )

        try:
            X_aligned_filled = fill_all_gaps_after_alignment_for_preprocessing(
                X_aligned_raw
            )

        except ValueError as e:
            print(
                f"[WARNING] Skipped file {os.path.basename(fpath)} | "
                f"Reason: full gap filling failed after gravity alignment. {e}"
            )
            continue

        acc_aligned = X_aligned_filled[:, 0:3]
        gyr_aligned = X_aligned_filled[:, 3:6]

        # ------------------------------------------------------------
        # Brodie / Buckley M3 + final low-pass preprocessing
        # ------------------------------------------------------------
        # M3 and low-pass filtering are applied to the full trial after:
        # - raw gravity alignment;
        # - full gap filling for numerical continuity.
        #
        # The quality masks are not used here. They are applied only later,
        # during window construction, to decide which windows are accepted.
        gravity_vector = mean_static_acc.copy()

        try:
            acc_final, gyr_final = apply_preprocessing_full_trial_after_alignment(
                acc_aligned_filled=acc_aligned,
                gyr_aligned_filled=gyr_aligned,
                fs=FS,
                gravity_vector=gravity_vector,
                lowpass_cutoff_hz=LOWPASS_CUTOFF_HZ
            )

        except ValueError as e:
            print(
                f"[WARNING] Skipped file {os.path.basename(fpath)} | "
                f"Reason: full-trial preprocessing failed. {e}"
            )
            continue

        # ------------------------------------------------------------
        # Final 6-channel signal
        # ------------------------------------------------------------
        # Accelerometer:
        # - gravity-aligned;
        # - gap-filled for numerical continuity;
        # - continuously pitch/roll corrected with Brodie M3;
        # - gravity removed using the trial-specific estimated gravity vector;
        # - low-pass filtered.
        #
        # Gyroscope:
        # - gravity-aligned with the same initial rotation;
        # - low-pass filtered.
        X = np.concatenate([acc_final, gyr_final], axis=1)

        # ---------------------------------------------
        # Labels and metadata from GeneralEvent
        # ---------------------------------------------
        labels_raw = df["GeneralEvent"].astype(str).str.lower()

        # ------------------------------------------------------------
        # WearGaitPD quality assessment
        # ------------------------------------------------------------
        # This block computes the signal-quality masks described in Anderson's
        # supplementary materials. The quality check is based on two independent
        # sources of poor data quality:
        #
        # 1. IMU quality:
        #    Long wireless dropouts are detected as regions with consecutive NaN values
        #    in either accelerometer or gyroscope channels. Regions with 7 or more
        #    consecutive missing frames are marked as low quality.
        #
        # 2. Walkway quality:
        #    The instrumented walkway provides two Boolean foot-ground contact signals:
        #    L Foot Contact and R Foot Contact. These signals are combined using an OR
        #    operation to obtain the "Either Foot-Ground Contact" waveform.
        #
        #    During walking, at least one foot should be detected in contact with the
        #    ground/walkway at each instant. Therefore, if both contact signals are zero
        #    inside a dynamic walking region, that portion is considered unreliable
        #    because it may correspond to an off-walkway step or another walkway artifact.
        #
        #    A 50-frame buffer is removed from the beginning and end of each valid
        #    walkway-contact region. This follows Anderson's supplementary description
        #    and reduces the risk of keeping heel strikes occurring near the walkway
        #    boundaries.
        #
        # 3. Total quality:
        #    The final quality mask is obtained as:
        #    total_quality = imu_quality AND walkway_quality
        #
        # Important:
        # The quality masks are computed after preprocessing in the code order,
        # but they are computed from the original raw IMU and walkway-contact
        # signals. This is essential because gap filling removes NaNs from the
        # preprocessed signal, while IMU quality must still describe the original
        # wireless-dropout pattern.
        #
        # Important pipeline rule:
        # - quality masks are not applied here;
        # - the preprocessed signal is not cut or modified by quality masks;
        # - quality masks are stored and applied only later during window creation.
        X_quality_raw = np.concatenate([acc, gyr], axis=1)

        # Decide whether walkway quality must be computed for the current task.
        # Walkway quality is meaningful only for instrumented-walkway tasks.
        # For Balance and FreeWalk, the walkway request is skipped completely:
        # no artificial L/R foot-contact signals are created.
        use_walkway_quality = task_name in WALKWAY_QUALITY_TASKS

        if use_walkway_quality:

            # Walkway-based tasks must have both contact columns.
            # This should already be guaranteed by the earlier header check,
            # but the condition is repeated here for safety and readability.
            if l_contact_col is None or r_contact_col is None:
                print(
                    f"[WARNING] Skipped file {os.path.basename(fpath)} | "
                    "Reason: missing L/R Foot Contact columns required for walkway quality."
                )
                continue

            # Convert the left-foot walkway contact column to a numeric array.
            # Non-numeric values are converted to NaN and then replaced by 0
            # inside utils.WearGaitPD_quality.compute_weargait_quality_masks.
            l_foot_contact = pd.to_numeric(
                df[l_contact_col],
                errors="coerce"
            ).to_numpy()

            # Convert the right-foot walkway contact column to a numeric array.
            r_foot_contact = pd.to_numeric(
                df[r_contact_col],
                errors="coerce"
            ).to_numpy()

        else:

            # Non-walkway tasks skip walkway-quality computation.
            # No neutral/artificial foot-contact waveform is generated.
            # The quality function will compute IMU quality only and will return:
            # walkway_quality = None
            # total_quality = imu_quality
            l_foot_contact = None
            r_foot_contact = None

        # Compute the sample-wise quality masks from the original raw signals.
        # For walkway tasks:
        # - imu_quality is computed from raw IMU missing-data regions;
        # - walkway_quality is computed from L/R foot-contact signals;
        # - total_quality = imu_quality AND walkway_quality.
        #
        # For non-walkway tasks:
        # - only imu_quality is computed;
        # - walkway_quality is None;
        # - total_quality = imu_quality.
        imu_quality, walkway_quality, total_quality, quality_debug_info = compute_weargait_quality_masks(
            X_raw=X_quality_raw,
            l_foot_contact=l_foot_contact,
            r_foot_contact=r_foot_contact,
            gap_threshold_frames=7,
            edge_buffer_frames=50,
            use_walkway_quality=use_walkway_quality
        )

        # ------------------------------------------------------------
        # Optional quality-check debug plot
        # ------------------------------------------------------------
        # This plot allows visual verification of the quality mask against:
        # - raw acceleration;
        # - L/R foot contact;
        # - GeneralEvent annotations.
        if (SAVE_FILTER_DEBUG_PLOTS
            and debug_plot_count < MAX_DEBUG_PLOTS
            and use_walkway_quality):
                debug_plot_path = save_weargait_quality_debug_plot(
                output_dir=FILTER_DEBUG_DIR,
                file_name=os.path.basename(fpath),
                acc_signal=acc,
                l_foot_contact=l_foot_contact,
                r_foot_contact=r_foot_contact,
                imu_quality=imu_quality,
                walkway_quality=walkway_quality,
                final_quality=total_quality,
                labels_raw=labels_raw.to_numpy(),
                fs=FS,
                max_duration_s=60
            )

                debug_plot_count += 1

                print("[QUALITY DEBUG] Saved:", debug_plot_path)

        '''
        # Optional FreeWalk label debug.
        # Keep this block commented during full dataset construction,
        # because it prints many lines and slows down execution.
        if task_name == "freewalk":
            print("\n[DEBUG FREEWALK EVENTS]", subject_id, os.path.basename(fpath))
            print(labels_raw.value_counts().head(30))
            print("Contains stair:", labels_raw.str.contains("stair", na=False).sum())
            print("Contains stairs:", labels_raw.str.contains("stairs", na=False).sum())
            print("Contains ascend:", labels_raw.str.contains("ascend|ascent|up", na=False, regex=True).sum())
            print("Contains descend:", labels_raw.str.contains("descend|descent|down", na=False, regex=True).sum())
        '''

        # Create sample-wise activity-detail metadata from GeneralEvent annotations.
        # This preserves the original activity information:
        # none, static, walking, ascending and descending.
        activity_detail = assign_task_activity_detail(
            labels_raw=labels_raw,
            task_name=task_name,
            subject_id=subject_id
        )

        # Convert activity detail into final binary GSD labels.
        # Static remains static, while walking, ascending and descending are merged
        # into the walking class.
        labels_num = convert_activity_detail_to_gsd(activity_detail)

        # Create sample-wise static-type metadata.
        # This distinguishes generic static, standing and sitting when meaningful.
        # Static type is assigned only where activity_detail is static.
        static_type = assign_static_type(
            labels_raw=labels_raw,
            task_name=task_name,
            activity_detail=activity_detail
        )

        # Create sample-wise gait phase metadata.
        # This preserves gait initiation and termination in WearGait-PD.
        gait_phase = assign_gait_phase(
            labels_num=labels_num,
            task_name=task_name,
            fs=FS,
            transient_duration_s=1.0
        )

        # Create sample-wise walking-bout metadata.
        # Since WearGaitPD has no INDIP CWP segment IDs, bout IDs are derived from
        # contiguous walking regions in the binary GSD label vector.
        walking_bouts = build_sample_walking_bouts(labels_num)

        # ------------------------------------------------------------
        # Path type definition
        # ------------------------------------------------------------
        # Path type is defined sample by sample as locomotor-path metadata.
        #
        # Convention:
        # - None = path type not applicable, used for static or unlabeled samples
        # - 0    = straight, used for locomotor samples outside turn annotations
        # - 1    = curved, used for locomotor samples inside turn annotations
        #
        # This avoids assigning "straight" to static samples, which would incorrectly
        # make static windows look like locomotor straight-path windows.

        path_types = np.full(
            len(labels_raw),
            None,
            dtype=object
        )

        # Locomotor samples are identified from the HAR-level labels.
        # Ascending and descending are also locomotor activities, so they receive a
        # path type as well. Static and none/unlabeled samples remain None.
        locomotor_mask = (
                (activity_detail == ACTIVITY_DETAIL_MAP["walking"])
                |
                (activity_detail == ACTIVITY_DETAIL_MAP["ascending"])
                |
                (activity_detail == ACTIVITY_DETAIL_MAP["descending"])
        )

        # Locomotor samples are straight by default.
        path_types[locomotor_mask] = 0

        # Curved samples are identified through turn annotations, but only when they
        # also belong to a locomotor portion. This prevents static turn-like annotations
        # or unlabeled turn samples from receiving a locomotor path type.
        # Build a robust turn mask from GeneralEvent.
        # This supports both continuous turn labels and start/end turn markers.
        turn_mask = build_turn_mask_from_general_event(labels_raw)

        path_types[locomotor_mask & turn_mask] = 1

        # trial_dataset is a trial information dictionary.
        trial_dataset.append({
            "dataset": "WearGaitPD",
            "subject_id": subject_id,
            "task": task_name,
            "test_type": task_name,
            "challenge_type": "none",
            "file_name": os.path.basename(fpath),
            "file_path": fpath,  # absolute path kept only for local debugging.
            "relative_file_path": make_project_relative_path(fpath),
            "fs": FS,
            "X": X,
            "labels": labels_num,  # final binary GSD labels: none/static/walking.
            "activity_detail": activity_detail,  # metadata preserving none/static/walking/ascending/descending.
            "static_type": static_type,  # none/static/standing/sitting.
            "gait_phase": gait_phase,  # gait initiation, steady_state, gait termination
            "walking_bouts": walking_bouts,
            "SampleWalkingBouts": walking_bouts,
            "walkway_quality": walkway_quality,
            "imu_quality": imu_quality,
            "total_quality": total_quality,
            "use_walkway_quality": use_walkway_quality,
            "path_types": path_types,  # None = not applicable, 0 = straight, 1 = curved.
            "label_map": GSD_LABEL_MAP,  # backward-compatible alias for the binary GSD target map.
            "gsd_label_map": GSD_LABEL_MAP,
            "activity_detail_map": ACTIVITY_DETAIL_MAP,
            "static_type_map": STATIC_TYPE_MAP,
            "gait_phase_map": GAIT_PHASE_MAP,
            "path_type_map": {
                0: "straight",
                1: "curved",
                None: "none",
            },
            "general_event": labels_raw.to_numpy(),
            "group": get_group_from_path(fpath)
        })


    # =============================================================================

    # =====================================================
    # FINAL SAVE
    # =====================================================
    # This block saves the flat trial-level dataset and then creates window-level
    # datasets for all configured window lengths. Each window bundle should contain
    # X, Y and metadata, while a CSV file stores the readable metadata table.

    TRIAL_OUTPUT_PATH = RESULTS_DIR / "WearGaitPD_trial_dataset.pkl"
    with open(TRIAL_OUTPUT_PATH, "wb") as f:
        pickle.dump(trial_dataset, f)

    print(f"[SUCCESS] Trial-level dataset saved to: {TRIAL_OUTPUT_PATH}")

    print("[SUCCESS] Saved.")

    WINDOW_CONFIGS = [
        ("1s_50p_overlap", 100, 50),
        ("2s_50p_overlap", 200, 100),
        ("5s_50p_overlap", 500, 250),
        ("10s_50p_overlap", 1000, 500),
    ]

    for window_config_name, window_size, step_size in WINDOW_CONFIGS:

        output_path = RESULTS_DIR / f"WearGaitPD_window_dataset_{window_config_name}.pkl"

        create_window_dataset_from_trials(
            trial_dataset=trial_dataset,
            window_size=window_size,
            step_size=step_size,
            output_path=output_path
        )

if __name__ == "__main__":
    main()