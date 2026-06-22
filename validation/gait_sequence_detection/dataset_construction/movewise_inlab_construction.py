"""
MOVEWISE InLab trial-level dataset construction.

This script builds the trial-level dataset from MOVEWISE InLab MATLAB files.

Current goal:
- load InLab data.mat files
- use all useful tests except Test3
- extract LeftEar Acc, Gyr, Mag, Timestamp and Fs
- build walking / non-walking labels
- build walking bout IDs when INDIP annotations are available
- build path type from turn annotations when available
- apply gravity alignment using the first second of each trial
- save the complete trial-level dataset as pickle
"""

import os
import pickle
import scipy.io
import numpy as np
import pandas as pd

from utils.rotations import align_imu_to_gravity
from pathlib import Path

def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "data").exists():
            return parent

    raise FileNotFoundError(
        "Project root not found. Expected folders: src/ and data/."
    )


PROJECT_ROOT = find_project_root(Path(__file__))
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

ROOT_PATH = DATA_ROOT / "MOVEWISE"

TEST_TYPE_MAP = {
    "Test1": "static",
    "Test2": "standing",
    "Test3": "data personalization",
    "Test4": "straight_walking",
    "Test5": "DT_texting",
    "Test6": "DT_browsing",
    "Test7": "DT_calling",
    "Test8": "DT_audio_comprehension",
    "Test9": "DT_audio_disturb",
    "Test10": "DT_visual_exploration",
}

# =====================================================
# STANDARD LABEL / METADATA MAPS
# =====================================================
# These maps follow the same convention used in WearGaitPD.
# The GSD target is binary:
# - static
# - walking
# MOVEWISE InLab currently emits only static and walking activity_detail values.
# The ascending and descending codes are kept in the map only for cross-dataset
# schema compatibility, but they are not expected to appear in this dataset.
# The specific InLab protocol condition is represented by test_type.

GSD_LABEL_MAP = {
    "static": 0,
    "walking": 1,
}

GSD_LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}

ACTIVITY_DETAIL_MAP = {
    "static": 0,
    "walking": 1,
    "ascending": 2,
    "descending": 3,
}

ACTIVITY_DETAIL_NAME_MAP = {
    0: "static",
    1: "walking",
    2: "ascending",
    3: "descending",
}

STATIC_TYPE_MAP = {
    "none": -1,
    "static": 0,
    "standing": 1,
    "sitting": 2,
}

STATIC_TYPE_NAME_MAP = {
    -1: "none",
    0: "static",
    1: "standing",
    2: "sitting",
}

GAIT_PHASE_MAP = {
    "none": -1,
    "gait_initiation": 0,
    "steady_state": 1,
    "gait_termination": 2,
}

GAIT_PHASE_NAME_MAP = {
    -1: "none",
    0: "gait_initiation",
    1: "steady_state",
    2: "gait_termination",
}

PATH_TYPE_MAP = {
    0: "straight",
    1: "curved",
    None: "none",
}

# Minimum fraction required to assign a robust window-level path type.
# A window is considered straight or curved only if at least this fraction
# of the full window is assigned to that path type.
PATH_TYPE_MIN_FRACTION = 0.80

# Minimum fraction required to consider the final GSD window label robust.
# The final label is still assigned by majority voting, but the window is marked
# as robust only when the final class covers at least this fraction of the
# complete window.
#
# This replaces the old walking/static-ratio logic, because that ratio can be
# misleading when a window contains many invalid or non-applicable samples.
WINDOW_LABEL_MIN_FRACTION = 0.80

# Minimum fraction required to preserve a static subtype as contextual metadata.
# This does not change the final window label. It only stores whether a mixed
# window contains a meaningful amount of standing or sitting/static samples.
STATIC_CONTEXT_MIN_FRACTION = 0.10

# Human-readable description of how window-level transition metadata are built.
# In INDIP-derived datasets, transitions are derived from ContinuousWalkingPeriod
# boundaries. Static samples within 1 second before a walking bout are marked as
# Static2Walking, while static samples within 1 second after a walking bout are
# marked as Walking2Static. The sample labels are not modified.
TRANSITION_DEFINITION = (
    "CWP boundary +/-1 s on static samples around INDIP walking bouts"
)

# -----------------------------------------------------------------------------
# Function role: robust window-level path-type majority voting
# -----------------------------------------------------------------------------
# This helper assigns one path-type value to a window starting from the
# sample-wise path-type vector.
#
# It is needed because SamplePathTypes can contain mixed Python objects:
# - None = path type not applicable, usually static / non-walking samples
# - 0    = straight walking
# - 1    = curved walking
#
# np.unique is intentionally avoided here because arrays containing both None
# and integers can be fragile, especially when object dtype is used.
#
# Tie-breaking policy:
# - None is preferred in ties involving static/non-applicable samples, so a
#   borderline static/walking window is not incorrectly forced to straight.
# - curved is preferred over straight in ties between walking path types, because
#   preserving turning information is usually more informative for later analysis.
#
# This function is metadata-only: it does not affect the binary GSD label.

def majority_path_type(window_path_types):
    """
    Compute the dominant path type inside one window.

    Path-type conventions:
    None = path type not applicable
    0    = straight walking
    1    = curved walking
    """

    if window_path_types is None:
        return None

    values = list(window_path_types)

    if len(values) == 0:
        return None

    n_none = sum(value is None for value in values)
    n_straight = sum(value == 0 for value in values)
    n_curved = sum(value == 1 for value in values)

    path_counts = {
        None: n_none,
        0: n_straight,
        1: n_curved,
    }

    # Priority is used only when two path types have the same count.
    # None is the most conservative choice for static/non-applicable windows.
    # Curved is preferred over straight when the walking portion is equally split.
    tie_priority = {
        None: 2,
        1: 1,
        0: 0,
    }

    return max(
        path_counts,
        key=lambda key: (path_counts[key], tie_priority[key])
    )

# -----------------------------------------------------------------------------
# Function role: robust unique path-type extraction for debug printing
# -----------------------------------------------------------------------------
# This helper extracts the unique path-type values from a sample-wise path-type
# vector without using np.unique.
#
# This is necessary because SamplePathTypes is an object array that may contain:
# - None = path type not applicable
# - 0    = straight
# - 1    = curved
#
# np.unique can fail on mixed object arrays containing None because it internally
# tries to sort the values. This helper preserves the intended path-type logic
# and is used only for readable debug printing.

def unique_path_types_safe(sample_path_types):
    """
    Return unique path-type values in a safe and ordered way.

    Output order:
    None -> 0 -> 1

    This makes debug output stable and avoids np.unique on object arrays.
    """

    if sample_path_types is None:
        return []

    found_values = []

    for value in sample_path_types:

        # None means path type is not applicable.
        if value is None:
            normalized_value = None

        else:
            # Convert numpy integer objects to standard Python integers.
            # This keeps compatibility with PATH_TYPE_MAP.
            normalized_value = int(value)

        if normalized_value not in found_values:
            found_values.append(normalized_value)

    ordered_values = [
        value for value in [None, 0, 1]
        if value in found_values
    ]

    return ordered_values

# -----------------------------------------------------------------------------
# Function role: numeric label-to-name conversion
# -----------------------------------------------------------------------------
# This helper converts numeric map values back to readable names.
# It is used when creating metadata fields such as static_context_type_name.

def get_map_name(map_dict, value):
    """
    Convert a numeric map value into its corresponding string name.

    Example:
    STATIC_TYPE_MAP["standing"] = 1
    get_map_name(STATIC_TYPE_MAP, 1) -> "standing"
    """

    for key, val in map_dict.items():
        if val == value:
            return key

    return "unknown"


# -----------------------------------------------------------------------------
# Function role: robust window-level path-type summarization
# -----------------------------------------------------------------------------
# This helper replaces simple majority voting for path type.
#
# A window receives path_type = straight or curved only if that condition covers
# at least PATH_TYPE_MIN_FRACTION of the complete window.
#
# This is intentionally stricter than majority voting. It prevents a window with
# a small walking portion from being classified as straight only because that
# small walking portion is entirely straight.

def summarize_window_path_type(path_win, window_size, min_fraction):
    """
    Summarize straight/curved path composition inside one window.

    Path-type convention:
    - 0    = straight
    - 1    = curved
    - None = path type not applicable, usually static/non-walking samples

    Fractions are computed over the full window length.
    This keeps path_type consistent with the real composition of the window.
    """

    if path_win is None:
        path_values = [None] * window_size
    else:
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
# This helper preserves standing/sitting/static information even when the final
# window label is walking.
#
# Example:
# a window can have final label = walking, static_type_name = none,
# but still contain 20% standing samples. In that case this helper preserves
# standing_static_type_fraction and has_standing_context for error analysis.

def summarize_window_static_context(static_type_win, window_size, min_fraction):
    """
    Summarize static-subtype composition inside one window.

    Static-type convention:
    - -1 = none
    -  0 = generic static
    -  1 = standing
    -  2 = sitting

    This function does not change the final GSD label.
    It only creates contextual metadata useful for misclassification analysis.
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

    none_static_type_fraction = (
        n_none_static_type_samples / window_size
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
# Function role: subject discovery
# -----------------------------------------------------------------------------
# This function scans the dataset root directory and identifies all valid
# MOVEWISE subject folders. Only folders with 4-digit numeric names are kept.

def get_subject_list(root_path):
    """
    Return valid MOVEWISE subject folders.
    Subjects are expected to have 4-digit numeric folder names.
    """
    return sorted([
        folder for folder in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, folder))
        and folder.isdigit()
        and len(folder) == 4
    ])


# -----------------------------------------------------------------------------
# Function role: MATLAB loading
# -----------------------------------------------------------------------------
# This function loads the original MATLAB structure for one subject.
# The .mat file is converted into a Python-accessible structure using scipy.

def load_subject_mat(root_path, subject):
    """
    Load MOVEWISE InLab data.mat for one subject.
    Missing files are handled outside this function.
    """
    file_path = os.path.join(root_path, subject, "InLab", "data.mat")

    mat = scipy.io.loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False
    )

    return mat["data"]


def matlab_datenum_to_seconds(timestamp):
    """
    Convert MATLAB datenum timestamps into relative seconds.
    The first sample is used as time zero.
    """
    timestamp = np.asarray(timestamp).squeeze()
    return (timestamp - timestamp[0]) * 24 * 3600


# -----------------------------------------------------------------------------
# Function role: IMU signal extraction
# -----------------------------------------------------------------------------
# This function extracts LeftEar inertial signals from the MATLAB structure.
# The extracted channels are Acc, Gyr, Mag, timestamps and sampling frequencies.

def extract_left_ear_signals(trial_struct, subject, test_name, trial_name):
    """
    Extract LeftEar IMU signals from one InLab trial.
    Acc and Gyr are the six input channels used by the model.
    """
    imu = trial_struct.SU_INDIP.LeftEar

    acc = np.asarray(imu.Acc)
    gyr = np.asarray(imu.Gyr)
    mag = np.asarray(imu.Mag)
    timestamp = np.asarray(imu.Timestamp).squeeze()
    time = matlab_datenum_to_seconds(timestamp)

    fs = {
        "Acc": imu.Fs.Acc,
        "Gyr": imu.Fs.Gyr,
        "Mag": imu.Fs.Mag,
    }

    return {
        "Dataset": "MOVEWISE_InLab",
        "Subject": subject,
        "Task": test_name,
        "TestType": TEST_TYPE_MAP.get(test_name, test_name),
        "Trial": trial_name,
        "Acc": acc,
        "Gyr": gyr,
        "Mag": mag,
        "Timestamp": timestamp,
        "Time": time,
        "Fs": fs,
    }


# -----------------------------------------------------------------------------
# Function role: INDIP annotation extraction
# -----------------------------------------------------------------------------
# This function extracts ContinuousWalkingPeriod annotations generated by INDIP.
# Walking intervals and turn intervals are converted into Python dictionaries.

def extract_indip_annotations(trial_struct, test_name):
    """
    Extract INDIP ContinuousWalkingPeriod annotations when available.
    Static tests do not contain walking bouts and therefore return None.
    """
    if not hasattr(trial_struct, "Standards"):
        return None

    standards = trial_struct.Standards

    if not hasattr(standards, "INDIP"):
        return None

    indip = standards.INDIP

    if not hasattr(indip, "ContinuousWalkingPeriod"):
        return None

    cwp = indip.ContinuousWalkingPeriod

    if isinstance(cwp, np.ndarray):
        cwp_elements = cwp
    else:
        cwp_elements = [cwp]

    segments = []

    for i, elem in enumerate(cwp_elements):
        segment_dict = {
            "segment_id": i,
            "start": getattr(elem, "Start", None),
            "end": getattr(elem, "End", None),
            "turn_start": getattr(elem, "Turn_Start", None),
            "turn_end": getattr(elem, "Turn_End", None),
        }

        segments.append(segment_dict)

    return {
        "Task": test_name,
        "ReferenceSystem": "INDIP",
        "INDIP_Fs": getattr(indip, "Fs", None),
        "ContinuousWalkingSegments": segments,
    }


# -----------------------------------------------------------------------------
# Function role: sample-wise binary labels
# -----------------------------------------------------------------------------
# This function creates one binary label for each sample of the recording.
# Samples inside INDIP walking bouts become walking, others remain static.

def build_sample_labels(trial_data):
    """
    Build sample-wise labels for MOVEWISE InLab.

    Labels:
    0 = static
    1 = walking
    """
    time = trial_data["Time"]
    # Default label is static.
    # MOVEWISE InLab Test2: standing portions are represented as static.
    labels = np.full(
        len(time),
        GSD_LABEL_MAP["static"], # = 0
        dtype=np.int32
    )

    annotations = trial_data["Annotations"]

    if annotations is None:
        return labels

    for seg in annotations["ContinuousWalkingSegments"]:
        start = seg["start"]
        end = seg["end"]

        if start is None or end is None:
            continue

        idx_walking = (time >= start) & (time <= end)
        labels[idx_walking] = GSD_LABEL_MAP["walking"] # = 1

    return labels

# -----------------------------------------------------------------------------
# Function role: CWP-boundary transition metadata
# -----------------------------------------------------------------------------
# This function marks static samples immediately before and after each INDIP
# ContinuousWalkingPeriod boundary.
#
# It does NOT estimate gait initiation or gait termination.
# Labels are NOT modified: these samples remain static in SampleLabels.
#
# Possible transition types:
# - Static2Walking
# - Walking2Static
# - None

def build_sample_transition_types(trial_data, margin_s=1.0):
    """
    Mark static samples immediately before and after each Continuous Walking Period.

    The main SampleLabels are NOT modified.
    Samples before CWP start remain static.
    Samples after CWP end remain static.

    TransitionType is only metadata:
    - Static2Walking
    - Walking2Static
    """

    labels = trial_data["SampleLabels"]
    time = trial_data["Time"]
    annotations = trial_data["Annotations"]

    transition_types = np.array([None] * len(labels), dtype=object)

    if annotations is None:
        return transition_types

    margin_s = float(margin_s)

    for seg in annotations["ContinuousWalkingSegments"]:

        start = seg["start"]
        end = seg["end"]

        if start is None or end is None:
            continue

        # Static portion immediately before walking.
        pre_idx = (time >= start - margin_s) & (time < start)

        # Static portion immediately after walking.
        post_idx = (time > end) & (time <= end + margin_s)

        transition_types[pre_idx & (labels == GSD_LABEL_MAP["static"])] = "Static2Walking"
        transition_types[post_idx & (labels == GSD_LABEL_MAP["static"])] = "Walking2Static"
    return transition_types


# -----------------------------------------------------------------------------
# Function role: walking-bout indexing
# -----------------------------------------------------------------------------
# This function assigns one walking-bout ID to each sample.
# Samples outside walking bouts are assigned value -1.

def build_sample_walking_bouts(trial_data):
    """
    Assign a walking-bout ID to each sample.
    Samples outside walking bouts are marked as -1.
    """
    time = trial_data["Time"]
    walking_bouts = np.full(len(time), -1, dtype=int)

    annotations = trial_data["Annotations"]

    if annotations is None:
        return walking_bouts

    for seg in annotations["ContinuousWalkingSegments"]:
        start = seg["start"]
        end = seg["end"]
        segment_id = seg["segment_id"]

        if start is None or end is None:
            continue

        idx = (time >= start) & (time <= end)
        walking_bouts[idx] = segment_id

    return walking_bouts

def build_sample_path_types(trial_data):
    """
    Build sample-wise path type.

    Path types:
    None = path type not applicable, used for static / non-walking samples
    0    = straight walking
    1    = curved walking

    Static samples must not be initialized as straight.
    Therefore, the vector starts as None for all samples. Only samples labelled
    as walking are set to straight by default. Walking samples inside INDIP
    Turn_Start / Turn_End intervals are then overwritten as curved.
    """

    time = trial_data["Time"]
    labels = trial_data["SampleLabels"]
    annotations = trial_data["Annotations"]

    # Initialize every sample as None.
    # This prevents static windows from being incorrectly interpreted as straight.
    path_types = np.full(
        len(time),
        None,
        dtype=object
    )

    # Path geometry is meaningful only during walking.
    walking_idx = labels == GSD_LABEL_MAP["walking"]

    # Walking samples are straight by default.
    # Curved intervals, when available, will overwrite this value.
    path_types[walking_idx] = 0

    if annotations is None:
        return path_types

    for seg in annotations["ContinuousWalkingSegments"]:
        turn_start = seg.get("turn_start", None)
        turn_end = seg.get("turn_end", None)

        if turn_start is None or turn_end is None:
            continue

        turn_start_arr = np.atleast_1d(turn_start)
        turn_end_arr = np.atleast_1d(turn_end)

        if turn_start_arr.size == 0 or turn_end_arr.size == 0:
            continue

        for ts, te in zip(turn_start_arr, turn_end_arr):

            # Only walking samples inside turn intervals are marked as curved.
            # Static samples around walking-bout borders remain None.
            idx_turn = (
                (time >= ts)
                &
                (time <= te)
                &
                walking_idx
            )

            path_types[idx_turn] = 1

    return path_types


# -----------------------------------------------------------------------------
# Function role: activity-detail metadata
# -----------------------------------------------------------------------------
# This function preserves activity information before binary GSD merging.
# In MOVEWISE InLab only static and walking are currently available.

def build_sample_activity_detail(trial_data):
    """
    Build sample-wise activity-detail labels.

    activity_detail preserves the effective activity before GSD merging.
    In MOVEWISE InLab, the available activities are:
    - Test2: standing (static)
    - Test4-Test10: walking tasks with different test_type metadata

    There are no ascending/descending labels in MOVEWISE InLab.
    Therefore, activity_detail can only be static or walking here.
    """

    labels = trial_data["SampleLabels"]

    activity_detail = np.full(
        len(labels),
        ACTIVITY_DETAIL_MAP["static"],
        dtype=np.int32
    )

    activity_detail[
        labels == GSD_LABEL_MAP["walking"]
    ] = ACTIVITY_DETAIL_MAP["walking"]

    return activity_detail


# -----------------------------------------------------------------------------
# Function role: static-type metadata
# -----------------------------------------------------------------------------
# This function distinguishes different types of static behavior.
# Test2 is treated as standing while other static periods remain generic.

def build_sample_static_types(trial_data):
    """
    Build sample-wise static-type metadata.

    In MOVEWISE InLab:
    - Test2 is a pure standing static task;
    - static samples inside walking tasks are generic static periods;
    - walking samples remain none.
    """

    labels = trial_data["SampleLabels"]

    static_type = np.full(
        len(labels),
        STATIC_TYPE_MAP["none"],
        dtype=np.int32
    )

    static_mask = labels == GSD_LABEL_MAP["static"]

    if trial_data["Task"] == "Test2":
        static_type[static_mask] = STATIC_TYPE_MAP["standing"]
    else:
        static_type[static_mask] = STATIC_TYPE_MAP["static"]

    return static_type


# -----------------------------------------------------------------------------
# Function role: gait-phase metadata
# -----------------------------------------------------------------------------
# This function would normally (in WearGaitPD) identify gait initiation and termination.
# MOVEWISE InLab does not reliably provide these annotations.

def build_sample_gait_phases(trial_data):
    """
    Build sample-wise gait-phase metadata.

    MOVEWISE InLab does not provide reliable gait initiation,
    steady-state, or gait termination annotations in the dataset acquisition.

    Therefore, all samples are marked as none.
    """

    labels = trial_data["SampleLabels"]

    gait_phase = np.full(
        len(labels),
        GAIT_PHASE_MAP["none"],
        dtype=np.int32
    )

    return gait_phase

# -----------------------------------------------------------------------------
# Function role: metadata standardization
# -----------------------------------------------------------------------------
# This function creates standardized fields shared across datasets.
# The goal is compatibility with generic downstream ML pipelines.

def update_standard_trial_fields(trial_data):
    """
    Add WearGaitPD-like lowercase fields to the MOVEWISE trial dictionary.

    Original MOVEWISE keys are preserved, while standardized keys are added
    for compatibility with the generic ml_pipeline.
    """

    trial_data["X"] = np.hstack([
        trial_data["Acc"],
        trial_data["Gyr"]
    ])

    trial_data["dataset"] = trial_data["Dataset"]
    trial_data["subject_id"] = trial_data["Subject"]
    trial_data["task"] = trial_data["Task"]
    trial_data["test_type"] = trial_data["TestType"]
    # Challenge type is a future-compatible metadata field.
    # MOVEWISE InLab does not currently encode a specific challenge condition,
    # so all trials are marked as "none".
    trial_data["challenge_type"] = trial_data.get("ChallengeType", "none")

    trial_data["file_name"] = trial_data.get("FileName", "data.mat")
    trial_data["file_path"] = trial_data.get("FilePath", None)

    # Store a project-relative path for portable metadata.
    trial_data["relative_file_path"] = trial_data.get(
        "RelativeFilePath",
        str(
            Path("data")
            / "MOVEWISE"
            / trial_data["Subject"]
            / "InLab"
            / trial_data["file_name"]
        )
    )

    trial_data["fs"] = trial_data["Fs"]["Acc"]

    trial_data["labels"] = trial_data["SampleLabels"]
    trial_data["labels_har"] = trial_data["SampleActivityDetail"]

    trial_data["activity_detail"] = trial_data["SampleActivityDetail"]
    trial_data["static_type"] = trial_data["SampleStaticTypes"]
    trial_data["gait_phase"] = trial_data["SampleGaitPhases"]
    trial_data["path_types"] = trial_data["SamplePathTypes"]
    trial_data["transition_types"] = trial_data["SampleTransitionTypes"]

    # Walking-bout metadata aligned with TOWalk, INDIVI and MOVEWISE OutOfLab.
    # It is useful for bout-level analyses and for tracing each sample back to
    # the corresponding INDIP ContinuousWalkingPeriod.
    trial_data["walking_bouts"] = trial_data["SampleWalkingBouts"]

    trial_data["label_map"] = GSD_LABEL_MAP
    trial_data["gsd_label_map"] = GSD_LABEL_MAP
    trial_data["har_label_map"] = ACTIVITY_DETAIL_MAP
    trial_data["activity_detail_map"] = ACTIVITY_DETAIL_MAP
    trial_data["static_type_map"] = STATIC_TYPE_MAP
    trial_data["gait_phase_map"] = GAIT_PHASE_MAP
    trial_data["path_type_map"] = PATH_TYPE_MAP
    trial_data["transition_definition"] = TRANSITION_DEFINITION

    trial_data["group"] = trial_data.get("Group", "UNKNOWN")

    return trial_data

# -----------------------------------------------------------------------------
# Function role: gravity alignment
# -----------------------------------------------------------------------------
# This function rotates IMU signals so gravity is aligned to a reference axis.
# The first second of the current recording is used as static reference.

def apply_gravity_alignment_to_trial(trial_data, static_duration_s=1.0):
    """
    Apply gravity alignment using the first second of the current trial.
    This avoids dependency on a separate static acquisition.
    """
    acc = trial_data["Acc"]
    gyr = trial_data["Gyr"]
    sampling_rate_hz = trial_data["Fs"]["Acc"]

    acc_aligned, gyr_aligned, R = align_imu_to_gravity(
        acc=acc,
        gyr=gyr,
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=static_duration_s,
        gravity_ideal=np.array([1.0, 0.0, 0.0])
    )

    n_static = int(static_duration_s * sampling_rate_hz)

    mean_static_acc = np.mean(acc_aligned[:n_static], axis=0)
    mean_static_acc_norm = mean_static_acc / np.linalg.norm(mean_static_acc)

    '''print(
        f"\n[ALIGNMENT CHECK] "
        f"Subject {trial_data['Subject']} | "
        f"Task {trial_data['Task']} | "
        f"Trial {trial_data['Trial']}"
    )
    print("Mean aligned acceleration (g-unit):", mean_static_acc)
    print("Normalized mean aligned acceleration:", mean_static_acc_norm)
    '''
    trial_data["RawAcc"] = acc
    trial_data["RawGyr"] = gyr

    trial_data["Acc"] = acc_aligned
    trial_data["Gyr"] = gyr_aligned

    # Keep the standardized 6-channel representation aligned
    # with the final preprocessed Acc/Gyr signals.
    trial_data["X"] = np.hstack([
        trial_data["Acc"],
        trial_data["Gyr"]
    ])

    trial_data["RotationMatrix"] = R

    trial_data["GravityAlignment"] = {
        "reference": "first_1s_of_current_recording",
        "type": "self_initial_segment",
        "duration_s": static_duration_s,
        "mean_static_acc": mean_static_acc.tolist(),
        "mean_static_acc_norm": mean_static_acc_norm.tolist(),
        "gravity_ideal": [1.0, 0.0, 0.0],
    }

    return trial_data


# -----------------------------------------------------------------------------
# Function role: subject-level dataset construction
# -----------------------------------------------------------------------------
# This function processes all valid tests for one subject.
# Signal extraction, labeling and preprocessing are executed sequentially.

def build_subject_dataset(data_struct, subject):
    """
    Build MOVEWISE InLab trial-level dataset for one subject.

    Test1, Test3, Test8, Test9 is skipped because not significant.
    Test2 is kept as non-walking.
    Test4-Test10 are kept as walking/dynamic trials when available.
    """
    # A flat list is used so that the trial dataset has the same structure
    # as WearGaitPD and can be processed by the generic ml_pipeline.
    subject_dataset = []

    if not hasattr(data_struct, "TimeMeasure1"):
        print(f"[WARNING] Subject {subject}: TimeMeasure1 not found.")
        return subject_dataset

    time_measure = data_struct.TimeMeasure1

    for test_name in time_measure._fieldnames:

        if test_name in ["Test1", "Test3", "Test8", "Test9"]:
            continue

        if test_name not in TEST_TYPE_MAP:
            continue

        test_struct = getattr(time_measure, test_name)

        if not hasattr(test_struct, "Trial1"):
            print(f"[WARNING] Subject {subject} | {test_name}: Trial1 not found.")
            continue

        trial_name = "Trial1"
        trial_struct = test_struct.Trial1

        try:
            if not hasattr(trial_struct, "SU_INDIP"):
                print(f"[WARNING] Subject {subject} | {test_name}: SU_INDIP not found.")
                continue

            if not hasattr(trial_struct.SU_INDIP, "LeftEar"):
                print(f"[WARNING] Subject {subject} | {test_name}: LeftEar not found.")
                continue

            signal_data = extract_left_ear_signals(
                trial_struct=trial_struct,
                subject=subject,
                test_name=test_name,
                trial_name=trial_name
            )

            annotation_data = extract_indip_annotations(
                trial_struct=trial_struct,
                test_name=test_name
            )

            trial_data = {
                **signal_data,
                "Annotations": annotation_data,
            }

            trial_data["SampleLabels"] = build_sample_labels(trial_data)

            trial_data["SampleActivityDetail"] = build_sample_activity_detail(trial_data)

            # MOVEWISE InLab uses INDIP-derived ContinuousWalkingPeriod annotations.
            # These annotations do not reliably describe gait initiation or gait termination,
            # so SampleGaitPhases remains set to "none".
            trial_data["SampleGaitPhases"] = build_sample_gait_phases(trial_data)

            # Samples immediately before and after each CWP are still useful
            # as transition metadata. They are NOT relabelled: they remain static/non-walking.
            # This allows later analysis of windows close to walking-bout boundaries.
            trial_data["SampleTransitionTypes"] = build_sample_transition_types(
                trial_data,
                margin_s=1.0
            )

            trial_data["SampleWalkingBouts"] = build_sample_walking_bouts(trial_data)
            trial_data["SamplePathTypes"] = build_sample_path_types(trial_data)
            trial_data["SampleStaticTypes"] = build_sample_static_types(trial_data)

            trial_data = apply_gravity_alignment_to_trial(
                trial_data,
                static_duration_s=1.0
            )

            trial_data = update_standard_trial_fields(trial_data)

            subject_dataset.append(trial_data)

        except Exception as e:
            print(f"[WARNING] Failed subject {subject} | {test_name} | {trial_name}: {e}")

    return subject_dataset


# -----------------------------------------------------------------------------
# Function role: full dataset construction
# -----------------------------------------------------------------------------
# This function loops across all subjects and aggregates trial dictionaries
# into one flat dataset compatible with the ML pipeline.

def build_dataset(root_path, selected_subjects=None):
    """
    Build MOVEWISE InLab dataset across subjects.
    Missing InLab files are skipped safely.
    """
    # Flat trial-level dataset.
    # Each element is one trial dictionary.
    dataset = []

    subjects = get_subject_list(root_path)

    if selected_subjects is not None:
        subjects = [s for s in subjects if s in selected_subjects]

    for subject in subjects:
        print(f"\nProcessing subject {subject}...")

        file_path = os.path.join(root_path, subject, "InLab", "data.mat")

        if not os.path.exists(file_path):
            print(f"[SKIP] Subject {subject}: InLab data.mat not found.")
            continue

        try:
            data_struct = load_subject_mat(root_path, subject)
            subject_dataset = build_subject_dataset(data_struct, subject)

            if len(subject_dataset) > 0:
                dataset.extend(subject_dataset)

        except Exception as e:
            print(f"[WARNING] Failed to process subject {subject}: {e}")

    return dataset

# -----------------------------------------------------------------------------
# Function role: sliding-window segmentation
# -----------------------------------------------------------------------------
# This function converts continuous trials into fixed-length windows.
# Majority voting is used to assign labels and metadata to each window.

def build_windows_from_trial(trial_data, window_size=200, step_size=100):
    """
    Build fixed-length windows from one MOVEWISE trial.

    The final window label remains binary for GSD:
    - static
    - walking

    In addition to the final label, this function stores detailed window
    composition metadata:
    - static/walking/none sample counts and fractions;
    - robust-window flag based on final-label fraction;
    - robust path-type composition;
    - static-context composition;
    - transition metadata;
    - walking-bout metadata.
    """

    acc = trial_data["Acc"]
    gyr = trial_data["Gyr"]
    time = trial_data["Time"]

    sample_labels = trial_data["SampleLabels"]
    sample_activity_detail = trial_data["SampleActivityDetail"]
    sample_static_types = trial_data["SampleStaticTypes"]
    sample_gait_phases = trial_data["SampleGaitPhases"]
    sample_transition_types = trial_data["SampleTransitionTypes"]
    sample_walking_bouts = trial_data["SampleWalkingBouts"]
    sample_path_types = trial_data["SamplePathTypes"]

    # Acc and Gyr are concatenated to obtain the final 6-channel input.
    signal_6ch = np.hstack([acc, gyr])

    windows = []

    for start_idx in range(0, len(signal_6ch) - window_size + 1, step_size):

        end_idx = start_idx + window_size

        window_signal = signal_6ch[start_idx:end_idx]
        window_labels = sample_labels[start_idx:end_idx]
        window_activity_detail = sample_activity_detail[start_idx:end_idx]
        window_static_types = sample_static_types[start_idx:end_idx]
        window_gait_phases = sample_gait_phases[start_idx:end_idx]
        window_transition_types = sample_transition_types[start_idx:end_idx]
        window_walking_bouts = sample_walking_bouts[start_idx:end_idx]
        window_time = time[start_idx:end_idx]

        if sample_path_types is not None:
            window_path_types = sample_path_types[start_idx:end_idx]
        else:
            window_path_types = None

        # ------------------------------------------------------------
        # Final binary GSD label by majority voting
        # ------------------------------------------------------------
        # The final label is still assigned by majority voting over the
        # sample-wise binary labels. This preserves the original GSD logic.
        unique_labels, label_counts = np.unique(
            window_labels,
            return_counts=True
        )

        majority_label = int(
            unique_labels[np.argmax(label_counts)]
        )

        # ------------------------------------------------------------
        # Internal label composition
        # ------------------------------------------------------------
        # These metadata describe how the window is internally composed.
        # They are not used as features. They are saved only for filtering,
        # validation and misclassification analysis.
        n_static_samples = int(
            np.sum(window_labels == GSD_LABEL_MAP["static"])
        )

        n_walking_samples = int(
            np.sum(window_labels == GSD_LABEL_MAP["walking"])
        )

        # MOVEWISE labels are expected to be only static/walking.
        # This counter is still created for compatibility with WearGaitPD,
        # where none-labelled samples may exist.
        n_none_label_samples = int(
            window_size
            - n_static_samples
            - n_walking_samples
        )

        static_fraction = n_static_samples / window_size
        walking_fraction = n_walking_samples / window_size
        none_label_fraction = n_none_label_samples / window_size

        valid_label_fraction = (
            n_static_samples + n_walking_samples
        ) / window_size

        if majority_label == GSD_LABEL_MAP["static"]:
            final_label_fraction = static_fraction

        elif majority_label == GSD_LABEL_MAP["walking"]:
            final_label_fraction = walking_fraction

        else:
            final_label_fraction = 0.0

        # The old walking/static ratio is kept only as diagnostic metadata.
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

        # ------------------------------------------------------------
        # Static-context composition
        # ------------------------------------------------------------
        # This preserves static subtype information even when the final window
        # label is walking.
        #
        # Example:
        # label_name = walking, static_type_name = none,
        # standing_static_type_fraction = 0.20.
        static_context_summary = summarize_window_static_context(
            static_type_win=window_static_types,
            window_size=window_size,
            min_fraction=STATIC_CONTEXT_MIN_FRACTION
        )

        # ------------------------------------------------------------
        # Robust path-type composition
        # ------------------------------------------------------------
        # Path type is assigned only if straight or curved covers at least
        # PATH_TYPE_MIN_FRACTION of the full window.
        # The detailed fractions are always saved, even when the final
        # WindowPathType remains none.
        path_summary = summarize_window_path_type(
            path_win=window_path_types,
            window_size=window_size,
            min_fraction=PATH_TYPE_MIN_FRACTION
        )

        # ------------------------------------------------------------
        # Transition metadata
        # ------------------------------------------------------------
        # Transition type is obtained from sample-wise transition annotations.
        # This does not change the final GSD label.
        valid_transition_types = [
            t for t in window_transition_types
            if t is not None and t != "None"
        ]

        if len(valid_transition_types) == 0:
            # Store the string "None" instead of Python None to keep CSV metadata
            # explicit and consistent across all dataset construction scripts.
            window_transition_type = "None"
            window_is_transition = 0

        else:
            unique_transition_types, transition_counts = np.unique(
                valid_transition_types,
                return_counts=True
            )

            window_transition_type = unique_transition_types[
                np.argmax(transition_counts)
            ]

            window_is_transition = 1

        # ------------------------------------------------------------
        # Label-consistent final metadata
        # ------------------------------------------------------------
        # Final metadata must remain coherent with the final binary label.
        #
        # Therefore:
        # - final walking windows can have path_type and walking_bout;
        # - final walking windows must have static_type = none;
        # - final static windows can have static_type;
        # - final static windows must have path_type = none and walking_bout = -1.
        if majority_label == GSD_LABEL_MAP["walking"]:

            final_path_type = path_summary["final_path_type"]
            final_static_type = STATIC_TYPE_MAP["none"]

            valid_gait_phases = window_gait_phases[
                window_gait_phases != GAIT_PHASE_MAP["none"]
            ]

            if len(valid_gait_phases) > 0:
                unique_phase, phase_counts = np.unique(
                    valid_gait_phases,
                    return_counts=True
                )

                final_gait_phase = int(
                    unique_phase[np.argmax(phase_counts)]
                )

            else:
                final_gait_phase = GAIT_PHASE_MAP["none"]

            locomotor_details = window_activity_detail[
                (window_activity_detail == ACTIVITY_DETAIL_MAP["walking"])
                |
                (window_activity_detail == ACTIVITY_DETAIL_MAP["ascending"])
                |
                (window_activity_detail == ACTIVITY_DETAIL_MAP["descending"])
            ]

            if len(locomotor_details) > 0:
                unique_activity, activity_counts = np.unique(
                    locomotor_details,
                    return_counts=True
                )

                final_activity_detail = int(
                    unique_activity[np.argmax(activity_counts)]
                )

            else:
                final_activity_detail = ACTIVITY_DETAIL_MAP["walking"]

            valid_bouts = window_walking_bouts[
                window_walking_bouts >= 0
            ]

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

        elif majority_label == GSD_LABEL_MAP["static"]:

            final_path_type = None
            final_gait_phase = GAIT_PHASE_MAP["none"]
            final_activity_detail = ACTIVITY_DETAIL_MAP["static"]
            final_walking_bout = -1

            valid_static_types = window_static_types[
                window_static_types != STATIC_TYPE_MAP["none"]
            ]

            if len(valid_static_types) > 0:
                unique_static, static_counts = np.unique(
                    valid_static_types,
                    return_counts=True
                )

                final_static_type = int(
                    unique_static[np.argmax(static_counts)]
                )

            else:
                final_static_type = STATIC_TYPE_MAP["static"]

        else:

            final_path_type = None
            final_static_type = STATIC_TYPE_MAP["none"]
            final_gait_phase = GAIT_PHASE_MAP["none"]
            final_activity_detail = ACTIVITY_DETAIL_MAP["static"]
            final_walking_bout = -1

        window_dict = {
            "Dataset": trial_data["Dataset"],
            "Subject": trial_data["Subject"],
            "Task": trial_data["Task"],
            "TestType": trial_data["TestType"],
            "Trial": trial_data["Trial"],
            "ChallengeType": trial_data.get("challenge_type", "none"),

            # File provenance metadata.
            "FileName": trial_data.get("file_name", None),
            "FilePath": trial_data.get("file_path", None),
            "RelativeFilePath": trial_data.get("relative_file_path", None),

            # Explicit description of the transition-detection rule used by this dataset.
            "TransitionDefinition": trial_data.get(
                "transition_definition",
                TRANSITION_DEFINITION
            ),

            "WindowStartIdx": start_idx,
            "WindowEndIdx": end_idx,
            "WindowStartTime": float(window_time[0]),
            "WindowEndTime": float(window_time[-1]),

            "Signal": window_signal,

            "SampleLabels": window_labels,
            "WindowLabel": int(majority_label),

            "SampleActivityDetail": window_activity_detail,
            "WindowActivityDetail": int(final_activity_detail),

            "NStaticSamples": int(n_static_samples),
            "NWalkingSamples": int(n_walking_samples),
            "NNoneLabelSamples": int(n_none_label_samples),

            "StaticFraction": float(static_fraction),
            "WalkingFraction": float(walking_fraction),
            "NoneLabelFraction": float(none_label_fraction),
            "ValidLabelFraction": float(valid_label_fraction),
            "FinalLabelFraction": float(final_label_fraction),

            "LabelMinFraction": float(WINDOW_LABEL_MIN_FRACTION),
            "WalkingStaticRatio": float(walking_static_ratio),
            "IsRobustWindow": bool(is_robust_window),

            "SamplePathTypes": window_path_types,
            "WindowPathType": final_path_type,

            "PathTypeMinFraction": float(PATH_TYPE_MIN_FRACTION),
            "NStraightPathSamples": int(
                path_summary["n_straight_path_samples"]
            ),
            "NCurvedPathSamples": int(
                path_summary["n_curved_path_samples"]
            ),
            "NNonePathSamples": int(
                path_summary["n_none_path_samples"]
            ),
            "NOtherPathSamples": int(
                path_summary["n_other_path_samples"]
            ),
            "StraightPathFraction": float(
                path_summary["straight_path_fraction"]
            ),
            "CurvedPathFraction": float(
                path_summary["curved_path_fraction"]
            ),
            "NonePathFraction": float(
                path_summary["none_path_fraction"]
            ),
            "PathTypeConfidence": float(
                path_summary["path_type_confidence"]
            ),
            "IsPathTypeRobust": bool(
                path_summary["is_path_type_robust"]
            ),

            "SampleWalkingBouts": window_walking_bouts,
            "WindowWalkingBout": int(final_walking_bout),

            "SampleTransitionTypes": window_transition_types,
            "WindowIsTransition": int(window_is_transition),
            "WindowTransitionType": window_transition_type,

            "WindowStaticType": int(final_static_type),
            "WindowGaitPhase": int(final_gait_phase),

            "StaticContextMinFraction": float(
                static_context_summary["static_context_min_fraction"]
            ),
            "NNoneStaticTypeSamples": int(
                static_context_summary["n_none_static_type_samples"]
            ),
            "NGenericStaticTypeSamples": int(
                static_context_summary["n_generic_static_type_samples"]
            ),
            "NStandingStaticTypeSamples": int(
                static_context_summary["n_standing_static_type_samples"]
            ),
            "NSittingStaticTypeSamples": int(
                static_context_summary["n_sitting_static_type_samples"]
            ),
            "NoneStaticTypeFraction": float(
                static_context_summary["none_static_type_fraction"]
            ),
            "GenericStaticTypeFraction": float(
                static_context_summary["generic_static_type_fraction"]
            ),
            "StandingStaticTypeFraction": float(
                static_context_summary["standing_static_type_fraction"]
            ),
            "SittingStaticTypeFraction": float(
                static_context_summary["sitting_static_type_fraction"]
            ),
            "StaticContextType": int(
                static_context_summary["static_context_type"]
            ),
            "StaticContextTypeName": static_context_summary[
                "static_context_type_name"
            ],
            "StaticContextFraction": float(
                static_context_summary["static_context_fraction"]
            ),
            "HasStaticContext": bool(
                static_context_summary["has_static_context"]
            ),
            "HasStandingContext": bool(
                static_context_summary["has_standing_context"]
            ),
            "HasSittingContext": bool(
                static_context_summary["has_sitting_context"]
            ),

            "FileName": trial_data.get("file_name", None),
            "FilePath": trial_data.get("file_path", None),
            "Group": trial_data.get("group", "UNKNOWN"),
        }

        windows.append(window_dict)

    return windows

# -----------------------------------------------------------------------------
# Function role: dataset windowing
# -----------------------------------------------------------------------------
# This function applies sliding-window segmentation to the whole dataset.
# All trial windows are aggregated into one final list.

def build_window_dataset(dataset, window_size, step_size):
    """
    Build the full window-level MOVEWISE InLab dataset.

    The input dataset is a flat list of trial dictionaries.
    """

    window_dataset = []

    for trial_data in dataset:

        trial_windows = build_windows_from_trial(
            trial_data=trial_data,
            window_size=window_size,
            step_size=step_size
        )

        window_dataset.extend(trial_windows)

    return window_dataset

# -----------------------------------------------------------------------------
# Function role: ML-ready array conversion
# -----------------------------------------------------------------------------
# This function converts list-based windows into numpy arrays and metadata.
# Output structures are ready for machine-learning training pipelines.

def convert_windows_to_arrays(window_dataset):
    """
    Convert list-based windows into arrays ready for model training.

    Metadata names are aligned with WearGaitPD, INDIVI, TOWalk and MOVEWISE
    OutOfLab so that downstream ml_pipeline scripts can be reused.
    """

    X = np.stack([w["Signal"] for w in window_dataset], axis=0)
    y = np.asarray([w["WindowLabel"] for w in window_dataset], dtype=int)

    metadata = []

    for w in window_dataset:

        metadata.append({
            "dataset": w.get("Dataset", "MOVEWISE_InLab"),
            "subject_id": w["Subject"],
            "task": w["Task"],
            "test_type": w["TestType"],
            "challenge_type": w.get("ChallengeType", "none"),
            "file_name": w.get("FileName", None),
            "file_path": w.get("FilePath", None),
            "relative_file_path": w.get("RelativeFilePath", None),
            "trial": w["Trial"],

            "window_start_sample": w["WindowStartIdx"],
            "window_end_sample": w["WindowEndIdx"],
            "window_start_time": w["WindowStartTime"],
            "window_end_time": w["WindowEndTime"],
            "window_duration_s": w["WindowEndTime"] - w["WindowStartTime"],

            "label": w["WindowLabel"],
            "label_name": GSD_LABEL_NAME_MAP.get(
                w["WindowLabel"],
                "unknown"
            ),

            "activity_detail": w["WindowActivityDetail"],
            "activity_detail_name": ACTIVITY_DETAIL_NAME_MAP.get(
                w["WindowActivityDetail"],
                "unknown"
            ),

            "n_static_samples": w["NStaticSamples"],
            "n_walking_samples": w["NWalkingSamples"],
            "n_none_label_samples": w["NNoneLabelSamples"],
            "static_fraction": w["StaticFraction"],
            "walking_fraction": w["WalkingFraction"],
            "none_label_fraction": w["NoneLabelFraction"],
            "valid_label_fraction": w["ValidLabelFraction"],
            "final_label_fraction": w["FinalLabelFraction"],
            "label_min_fraction": w["LabelMinFraction"],
            "walking_static_ratio": w["WalkingStaticRatio"],
            "is_robust_window": w["IsRobustWindow"],

            "path_type": w["WindowPathType"],
            "path_type_name": PATH_TYPE_MAP.get(
                w["WindowPathType"],
                "none"
            ),
            "path_type_min_fraction": w["PathTypeMinFraction"],
            "n_straight_path_samples": w["NStraightPathSamples"],
            "n_curved_path_samples": w["NCurvedPathSamples"],
            "n_none_path_samples": w["NNonePathSamples"],
            "n_other_path_samples": w["NOtherPathSamples"],
            "straight_path_fraction": w["StraightPathFraction"],
            "curved_path_fraction": w["CurvedPathFraction"],
            "none_path_fraction": w["NonePathFraction"],
            "path_type_confidence": w["PathTypeConfidence"],
            "is_path_type_robust": w["IsPathTypeRobust"],

            "static_type": w["WindowStaticType"],
            "static_type_name": STATIC_TYPE_NAME_MAP.get(
                w["WindowStaticType"],
                "none"
            ),

            "static_context_min_fraction": w["StaticContextMinFraction"],
            "n_none_static_type_samples": w["NNoneStaticTypeSamples"],
            "n_generic_static_type_samples": w["NGenericStaticTypeSamples"],
            "n_standing_static_type_samples": w["NStandingStaticTypeSamples"],
            "n_sitting_static_type_samples": w["NSittingStaticTypeSamples"],
            "none_static_type_fraction": w["NoneStaticTypeFraction"],
            "generic_static_type_fraction": w["GenericStaticTypeFraction"],
            "standing_static_type_fraction": w["StandingStaticTypeFraction"],
            "sitting_static_type_fraction": w["SittingStaticTypeFraction"],
            "static_context_type": w["StaticContextType"],
            "static_context_type_name": w["StaticContextTypeName"],
            "static_context_fraction": w["StaticContextFraction"],
            "has_static_context": w["HasStaticContext"],
            "has_standing_context": w["HasStandingContext"],
            "has_sitting_context": w["HasSittingContext"],

            "is_transition": bool(w["WindowIsTransition"]),
            "transition_type": w.get("WindowTransitionType", "None"),
            "transition_definition": w.get(
                "TransitionDefinition",
                TRANSITION_DEFINITION
            ),

            "gait_phase": w["WindowGaitPhase"],
            "gait_phase_name": GAIT_PHASE_NAME_MAP.get(
                w["WindowGaitPhase"],
                "none"
            ),

            "walking_bout": w["WindowWalkingBout"],
            "group": w.get("Group", "UNKNOWN"),
        })

    return X, y, metadata

# -----------------------------------------------------------------------------
# Function role: dataset inspection
# -----------------------------------------------------------------------------
# This function prints high-level dataset statistics.
# It helps verify subjects, tasks and label distributions.

def summarize_dataset(dataset):
    """
    Print trial-level dataset summary and sample-label distribution.

    The input dataset is a flat list of trial dictionaries.
    """

    subjects = sorted({
        trial_data["Subject"]
        for trial_data in dataset
    })

    tasks = sorted({
        trial_data["Task"]
        for trial_data in dataset
    })

    all_labels = []

    for trial_data in dataset:
        all_labels.extend(trial_data["SampleLabels"])

    print("\n----- MOVEWISE INLAB TRIAL DATASET SUMMARY -----")
    print("Number of subjects:", len(subjects))
    print("Number of tasks:", len(tasks))
    print("Number of valid trials:", len(dataset))

    print("\nTasks:")
    for task in tasks:
        n_task_trials = sum(
            trial_data["Task"] == task
            for trial_data in dataset
        )
        print(f"  {task}: {n_task_trials} trials")

    if len(all_labels) > 0:
        unique_labels, counts = np.unique(all_labels, return_counts=True)

        print("\nSample-label distribution:")
        for label, count in zip(unique_labels, counts):
            label_name = GSD_LABEL_NAME_MAP.get(int(label), "unknown")
            print(f"  Label {label} ({label_name}): {count}")


# -----------------------------------------------------------------------------
# Function role: dataset serialization
# -----------------------------------------------------------------------------
# This function saves the dataset as pickle for later reuse.
# Pickle preserves Python dictionaries and numpy arrays efficiently.

def save_dataset(dataset, save_path):
    """
    Save dataset as pickle.
    """
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print("\nDataset saved to:", save_path)



# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
# The following section executes the complete MOVEWISE InLab pipeline:
# 1. subject loading
# 2. trial construction
# 3. gravity alignment
# 4. metadata generation
# 5. window segmentation
# 6. export of trial-level and window-level datasets
# =============================================================================

if __name__ == "__main__":

    selected_subjects = [str(i) for i in range(5001, 5022)]

    dataset = build_dataset(
        root_path=ROOT_PATH,
        selected_subjects=selected_subjects
    )

    summarize_dataset(dataset)

    # ------------------------------------------------------------
    # Results directory
    # ------------------------------------------------------------

    results_dir = RESULTS_ROOT / "MOVEWISE_InLab_dataset"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Save trial dataset
    # ------------------------------------------------------------
    trial_save_path = (
            results_dir /
            "movewise_inlab_trial_dataset.pkl"
    )

    save_dataset(
        dataset,
        trial_save_path
    )

    print(f"\n[SUCCESS] Trial dataset saved to:")
    print(trial_save_path)

    WINDOW_CONFIGS = {
        "1s_50p_overlap": {"window_size": 100, "step_size": 50},
        "2s_50p_overlap": {"window_size": 200, "step_size": 100},
        "5s_50p_overlap": {"window_size": 500, "step_size": 250},
        "10s_50p_overlap": {"window_size": 1000, "step_size": 500},
    }


    for config_name, cfg in WINDOW_CONFIGS.items():

        print(f"\n[INFO] Building MOVEWISE InLab window dataset: {config_name}")

        window_dataset = build_window_dataset(
            dataset=dataset,
            window_size=cfg["window_size"],
            step_size=cfg["step_size"]
        )

        X, y, metadata = convert_windows_to_arrays(window_dataset)

        # Save metadata CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_save_path = results_dir / f"MOVEWISE_InLab_metadata_{config_name}.csv"

        # ------------------------------------------------------------
        # Semantic consistency checks for window-level metadata
        # ------------------------------------------------------------
        # These checks verify that final metadata remain coherent with the
        # final binary GSD label.
        #
        # A walking window must not have a final static_type different from none.
        # A static window must not have a final path_type different from none.
        #
        # Static-context metadata are allowed in both cases because they describe
        # internal composition, not the final class.
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
                metadata_save_path.with_name(
                    metadata_save_path.stem + "_SEMANTIC_INCONSISTENCIES.csv"
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

        metadata_df.to_csv(metadata_save_path, index=False)

        print("\nPath type distribution by GSD label:")

        print("\nPath type distribution by GSD label:")
        print(
            metadata_df
            .groupby(["label_name", "path_type_name"])
            .size()
            .reset_index(name="n_windows")
        )

        print(f"\n----- WINDOW DATASET SUMMARY ({config_name}) -----")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Number of metadata entries:", len(metadata))

        unique_labels, counts = np.unique(y, return_counts=True)

        print("\nWindow-label distribution:")
        for lbl, cnt in zip(unique_labels, counts):
            label_name = "static" if lbl == 0 else "walking"
            percentage = 100 * cnt / len(y)
            print(f"  Label {lbl} ({label_name}): {cnt} windows ({percentage:.2f}%)")

        window_bundle = {
            "DatasetName": "MOVEWISE_InLab",
            "WindowConfig": config_name,
            "WindowSize": cfg["window_size"],
            "StepSize": cfg["step_size"],
            "SamplingFrequencyHz": 100,
            "PathTypeMinFraction": PATH_TYPE_MIN_FRACTION,
            "WindowLabelMinFraction": WINDOW_LABEL_MIN_FRACTION,
            "StaticContextMinFraction": STATIC_CONTEXT_MIN_FRACTION,
            "TransitionDefinition": TRANSITION_DEFINITION,
            "label_map": GSD_LABEL_MAP,
            "gsd_label_map": GSD_LABEL_MAP,
            "har_label_map": ACTIVITY_DETAIL_MAP,
            "activity_detail_map": ACTIVITY_DETAIL_MAP,
            "static_type_map": STATIC_TYPE_MAP,
            "gait_phase_map": GAIT_PHASE_MAP,
            "path_type_map": PATH_TYPE_MAP,
            "X": X,
            "Y": y,
            "metadata": metadata,
        }

        window_save_path = results_dir / f"movewise_inlab_window_dataset_{config_name}.pkl"
        save_dataset(window_bundle, window_save_path)

        print(f"[SUCCESS] Saved MOVEWISE InLab {config_name} window dataset")

    for trial_data in dataset:

        print(f"\nSubject {trial_data['Subject']}")
        print(f"  Task: {trial_data['Task']}")
        print(f"  Trial: {trial_data['Trial']}")
        print(f"  TestType: {trial_data['TestType']}")
        print(f"  Acc shape: {trial_data['Acc'].shape}")
        print(f"  Gyr shape: {trial_data['Gyr'].shape}")
        print(f"  Mag shape: {trial_data['Mag'].shape}")
        print(f"  Time shape: {trial_data['Time'].shape}")
        print(f"  Fs: {trial_data['Fs']}")
        print(f"  X shape: {trial_data['X'].shape}")
        print(f"  SampleLabels shape: {trial_data['SampleLabels'].shape}")
        print(f"  Unique labels: {np.unique(trial_data['SampleLabels'])}")

        print(f"  Fs: {trial_data['Fs']}")
        print(f"  X shape: {trial_data['X'].shape}")

        print(f"  SampleLabels shape: {trial_data['SampleLabels'].shape}")

        label_names = [
            GSD_LABEL_NAME_MAP[int(x)]
            for x in np.unique(trial_data["SampleLabels"])
        ]

        print(f"  Unique labels: {label_names}")

        activity_names = [
            ACTIVITY_DETAIL_NAME_MAP[int(x)]
            for x in np.unique(trial_data["SampleActivityDetail"])
        ]

        print(f"  Activity detail: {activity_names}")

        static_type_names = [
            STATIC_TYPE_NAME_MAP[int(x)]
            for x in np.unique(trial_data["SampleStaticTypes"])
        ]

        print(f"  Static type: {static_type_names}")

        gait_phase_names = [
            GAIT_PHASE_NAME_MAP[int(x)]
            for x in np.unique(trial_data["SampleGaitPhases"])
        ]

        print(f"  Gait phases: {gait_phase_names}")

        print(f"  Task: {trial_data['Task']}")
        print(f"  TestType: {trial_data['TestType']}")


        # ------------------------------------------------------------
        # Safe path-type debug print
        # ------------------------------------------------------------
        # SamplePathTypes can contain None, 0 and 1.
        # np.unique is intentionally avoided here because mixed object arrays containing
        # None can raise a TypeError during sorting.
        if trial_data["SamplePathTypes"] is None:

            print("  Path types: None")
            print("  SamplePathTypes: None")

        else:

            unique_path_values = unique_path_types_safe(
                trial_data["SamplePathTypes"]
            )

            path_names = [
                PATH_TYPE_MAP.get(value, "unknown")
                for value in unique_path_values
            ]

            print(f"  Path types: {path_names}")
            print(f"  Unique path types: {unique_path_values}")