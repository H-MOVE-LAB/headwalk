"""
This module handles dataset construction starting from the raw MATLAB files.

Its goal is to load the original subject-level acquisitions, extract the
relevant signals and metadata, and organize them into a consistent Python
representation ready for the following pipeline blocks.

Main operations include:
- loading raw .mat files
- selecting the relevant sensor and task
- extracting inertial signals and timestamps
- converting timestamps into relative time
- organizing data into structured arrays or dictionaries

This module defines the input data representation used by the rest of the project.
"""

import os
import scipy.io
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from utils.quality_check import check_trial_quality, summarize_quality_report
from utils.rotations import align_imu_to_gravity_with_external_file
from utils.rotations import align_imu_to_gravity
from pathlib import Path

LABEL_MAP = {
    "static": 0,
    "walking": 1,
}

# Explicit GSD alias used to keep the INDIVI construction aligned with the
# other dataset construction scripts.
# LABEL_MAP is preserved because the current INDIVI code already uses it
# internally, while GSD_LABEL_MAP is added for common bundle metadata.
GSD_LABEL_MAP = LABEL_MAP

LABEL_NAME_MAP = {v: k for k, v in LABEL_MAP.items()}

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

GAIT_PHASE_NAME_MAP = {v: k for k, v in GAIT_PHASE_MAP.items()}

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
# In INDIVI, transitions are not inferred from a change inside the window.
# They are derived from INDIP ContinuousWalkingPeriod boundaries: static samples
# within 1 second before a walking bout are marked as Static2Walking, while
# static samples within 1 second after a walking bout are marked as Walking2Static.
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
# Function role: robust path-type listing for consistency checks
# -----------------------------------------------------------------------------
# This helper extracts the distinct path-type values from a sample-level
# path-type vector without using np.unique.
#
# This is necessary because SamplePathTypes may contain mixed Python objects:
# - None = path type not applicable, usually static / non-walking samples;
# - 0    = straight walking;
# - 1    = curved walking.
#
# np.unique is fragile in this case because it tries to sort the array. Sorting
# fails when Python needs to compare None and integer values. This helper keeps
# the same information but handles None and numeric values separately.

def unique_path_types_safe(path_values):
    """
    Return distinct path-type values without sorting mixed None/int arrays.

    Output order:
    - None first, if present;
    - numeric path types after that, sorted increasingly.
    """

    if path_values is None:
        return [None]

    values = list(path_values)

    has_none = any(value is None for value in values)

    numeric_values = sorted({
        int(value)
        for value in values
        if value is not None
    })

    unique_values = []

    if has_none:
        unique_values.append(None)

    unique_values.extend(numeric_values)

    return unique_values

# -----------------------------------------------------------------------------
# Function role: numeric label-to-name conversion
# -----------------------------------------------------------------------------
# This helper converts numeric map values back to readable names.
# It is used when creating metadata fields such as static_context_type_name.
#
# Example:
# STATIC_TYPE_MAP["standing"] = 1
# get_map_name(STATIC_TYPE_MAP, 1) -> "standing"
#
# This helper is intentionally generic, so it can be reused with any dictionary
# that maps readable names to numerical codes.

def get_map_name(map_dict, value):
    """
    Convert a numeric map value into its corresponding string name.

    Parameters
    ----------
    map_dict : dict
        Dictionary where keys are readable names and values are numeric codes.

    value : int, float, None
        Numeric value to be converted back into a readable name.

    Returns
    -------
    str
        Readable name associated with the numeric value.
        If the value is not found, "unknown" is returned.
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
# This is stricter than majority voting. It avoids assigning "straight" or
# "curved" to windows where only a small part of the window is actually walking.
#
# Example:
# - 60% None, 40% straight -> final_path_type = None
# - 85% straight, 15% None -> final_path_type = straight
# - 50% straight, 50% curved -> final_path_type = None unless one reaches 80%

def summarize_window_path_type(path_win, window_size, min_fraction):
    """
    Summarize straight/curved path composition inside one window.

    Path-type convention:
    - 0    = straight
    - 1    = curved
    - None = path type not applicable, usually static/non-walking samples

    Fractions are computed over the full window length, not only over walking
    samples. This makes the final path_type conservative and consistent with
    the real composition of the window.
    """

    # If the whole trial has no path-type information, the current window is
    # treated as fully non-applicable from the path-type point of view.
    if path_win is None:
        path_values = [None] * window_size

    else:
        path_values = list(path_win)

    # Count straight samples.
    n_straight_path_samples = int(
        sum(value == 0 for value in path_values)
    )

    # Count curved samples.
    n_curved_path_samples = int(
        sum(value == 1 for value in path_values)
    )

    # Count samples where path type is not applicable.
    # This usually corresponds to static/non-walking samples.
    n_none_path_samples = int(
        sum(value is None for value in path_values)
    )

    # Count unexpected values, if any.
    # This should normally be zero, but it is useful as defensive metadata.
    n_other_path_samples = int(
        window_size
        - n_straight_path_samples
        - n_curved_path_samples
        - n_none_path_samples
    )

    # Fractions are computed over the full window.
    straight_path_fraction = n_straight_path_samples / window_size
    curved_path_fraction = n_curved_path_samples / window_size
    none_path_fraction = n_none_path_samples / window_size

    # Confidence is the largest fraction among the two meaningful locomotor
    # path types. None is intentionally not used as confidence, because the goal
    # is to quantify how reliable a possible straight/curved assignment is.
    path_type_confidence = max(
        straight_path_fraction,
        curved_path_fraction
    )

    # Assign straight only if straight covers enough of the full window.
    if straight_path_fraction >= min_fraction:
        final_path_type = 0
        is_path_type_robust = True

    # Assign curved only if curved covers enough of the full window.
    elif curved_path_fraction >= min_fraction:
        final_path_type = 1
        is_path_type_robust = True

    # Otherwise, keep path_type as none.
    # This means that the window is mixed, ambiguous, mostly static, or not
    # reliable enough for a final straight/curved label.
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
# This is important for misclassification analysis.
#
# Example:
# a window can have:
# - final label = walking
# - final static_type = none
# - standing_static_type_fraction = 0.20
#
# In this case the final label remains semantically coherent, but the internal
# static component is not lost.

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

    # Count samples where static type is not applicable.
    # This usually corresponds to walking samples.
    n_none_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["none"])
    )

    # Count generic static samples.
    n_generic_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["static"])
    )

    # Count standing samples.
    n_standing_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["standing"])
    )

    # Count sitting samples.
    # INDIVI usually does not contain sitting, but the field is kept for
    # compatibility with WearGaitPD and with the common metadata schema.
    n_sitting_static_type_samples = int(
        np.sum(static_type_win == STATIC_TYPE_MAP["sitting"])
    )

    # Fractions are computed over the full window.
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

    # Candidate static contexts.
    # "none" is not included here because the goal is to detect whether a
    # meaningful static subtype is present inside the window.
    candidate_contexts = {
        STATIC_TYPE_MAP["static"]: generic_static_type_fraction,
        STATIC_TYPE_MAP["standing"]: standing_static_type_fraction,
        STATIC_TYPE_MAP["sitting"]: sitting_static_type_fraction,
    }

    # Select the most represented static context.
    static_context_type = max(
        candidate_contexts,
        key=candidate_contexts.get
    )

    static_context_fraction = candidate_contexts[static_context_type]

    # If the best static context does not reach the minimum fraction, the
    # context is considered too weak and is reset to none.
    if static_context_fraction < min_fraction:
        static_context_type = STATIC_TYPE_MAP["none"]
        static_context_fraction = 0.0
        has_static_context = False

    else:
        has_static_context = True

    # Boolean flags for quick filtering during error analysis.
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

# =====================================================
# PROJECT ROOT DISCOVERY
# =====================================================

# -----------------------------------------------------------------------------
# Function role: automatic project-root discovery
# -----------------------------------------------------------------------------
# This function avoids hard-coded absolute Windows paths.
# Starting from the current file location, it walks upward through parent
# folders until it finds the project folder containing both 'src' and 'data'.
# This is important because the same script can then run on different PCs
# even if the user folder or OneDrive path is different.

def find_project_root(start_path: Path) -> Path:
    """
    Find the project root automatically.

    The project root is identified as the first parent folder
    containing both:
    - src/
    - data/

    This avoids hard-coded user-specific Windows paths.
    """

    current = start_path.resolve()

    for parent in [current] + list(current.parents):

        if (
            (parent / "src").exists()
            and
            (parent / "data").exists()
        ):
            return parent

    raise FileNotFoundError(
        "Project root not found. Expected folders: src/ and data/."
    )

PROJECT_ROOT = find_project_root(Path(__file__))

# -----------------------------------------------------------------------------
# Function role: subject discovery from INDIVI raw files
# -----------------------------------------------------------------------------
# INDIVI stores all subject .mat files inside one flat folder.
# This function identifies valid subjects by parsing filenames such as
# '001_data.mat'. The extracted subject IDs are returned in sorted order so
# that dataset construction is reproducible and easy to debug.

def get_subject_list(root_path):
    """
    Return the sorted list of valid subject IDs found in the flat ALL data folder.
    A valid subject is identified by files named like 001_data.mat.
    """
    # The ALL data folder does not contain one subfolder per subject.
    # Subject IDs are extracted from file names such as 001_data.mat.
    subjects = sorted({
        # subjects is the sorted list of the first [0] term contained in filename, when
        # it is split using "_" separator.
        filename.split("_")[0]
        # For loop over filenames that endswith "data_mat" and whose first portion,
        # separated by "_", is digit.
        for filename in os.listdir(root_path)
        if filename.endswith("_data.mat") and filename.split("_")[0].isdigit()
    })
    return subjects


# -----------------------------------------------------------------------------
# Function role: loading one subject-level MATLAB file
# -----------------------------------------------------------------------------
# This function loads one INDIVI .mat file and returns the top-level MATLAB
# 'data' structure. The scipy options are chosen to simplify MATLAB struct
# navigation from Python: singleton dimensions are removed and fields can be
# accessed with dot notation.

def load_subject_mat(root_path, subject):
    """
    Load the subject MATLAB file from the flat ALL data folder
    and return the top-level 'data' struct.
    """
    # In the ALL data folder, each subject file is stored directly as
    # subjectID_data.mat, for example 001_data.mat.
    file_path = os.path.join(root_path, f"{subject}_data.mat")

    # The MATLAB file is loaded with squeeze_me=True to remove useless
    # singleton dimensions and with struct_as_record=False to allow
    # attribute-style access to MATLAB structs.
    mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return mat["data"]


# -----------------------------------------------------------------------------
# Function role: timestamp conversion
# -----------------------------------------------------------------------------
# MATLAB datenum timestamps are absolute time values expressed in days.
# For signal processing and annotation matching, a relative time axis in
# seconds is easier to use. The first sample is set to time zero and all
# following samples are converted to elapsed seconds.

def matlab_datenum_to_seconds(timestamp):
    """
    Convert MATLAB datenum timestamps into relative time in seconds.
    Time starts from zero at the first sample.
    """
    # Up to now, timestamp is given as datenum: it means that it is
    # referred to the seconds passed from January 0th, 0.
    # It needs to be converted into relative seconds.
    timestamp = np.asarray(timestamp).squeeze()
    return (timestamp - timestamp[0]) * 24 * 3600


# -----------------------------------------------------------------------------
# Function role: LeftEar IMU extraction
# -----------------------------------------------------------------------------
# This function extracts the head/ear-worn IMU channels used by the pipeline.
# The selected sensor is SU_INDIP.LeftEar, and the extracted signals are
# accelerometer, gyroscope, magnetometer, timestamps and sampling frequencies.
# The returned dictionary is the basic trial-level signal container used by
# all later labeling, preprocessing and windowing steps.

def extract_left_ear_signals(trial_struct, subject, task_name, trial_name):
    # Data structure:
    # data
    # └── TimeMeasure1
    #     └── Stairs
    #         └── Trial1 (trial_struct)
    #             └── ...
    #             └── SU_INDIP
    #                 └── LeftEar
    #                     ├── Acc
    #                     ├── Gyr
    #                     ├── Mag
    #                     ├── Timestamp
    #                     └── Fs
    #         └── Trial2
    #             ├── StartDateTime
    #             ├── TimeZone
    #             ├── SU_INDIP
    #             └── Standards
    #                 ├── BarometricData_raw
    #                 ├── Temperature_raw
    #                 ├── PressureInsoles_raw
    #                 ├── DistanceModule_raw
    #                 └── INDIP
    #                     ├── ContinuousWalkingPeriod
    #                     │   ├── [0]
    #                     │   │   ├── Start = 16.27
    #                     │   │   ├── End = 23.37
    #                     │   │   ├── Incline_Start = 16.27
    #                     │   │   ├── Incline_End = 23.37
    #                     │   │   └── ...
    #                     │   └── [1]
    #                     │       ├── Start = 41.87
    #                     │       ├── End = 48.27
    #                     │       ├── Incline_Start = 41.87
    #                     │       ├── Incline_End = 48.27
    #                     │       └── ...
    #                     └── Fs
    #     └── ...
    """
    Extract LeftEar IMU signals and metadata from one trial.

    Returns a dictionary with:
    - Subject
    - Task
    - Trial
    - Acc
    - Gyr
    - Mag
    - Timestamp
    - Time
    - Fs
    """
    imu = trial_struct.SU_INDIP.LeftEar

    acc = np.asarray(imu.Acc)
    gyr = np.asarray(imu.Gyr)
    mag = np.asarray(imu.Mag)
    timestamp = np.asarray(imu.Timestamp).squeeze()
    time = matlab_datenum_to_seconds(timestamp)

    # Each type of signal is sampled at 100 Hz.
    fs = {
        "Acc": imu.Fs.Acc,
        "Gyr": imu.Fs.Gyr,
        "Mag": imu.Fs.Mag,
    }

    # Returned output dictionary including both metadata and numerical data.
    return {
        "Dataset": "INDIVI",
        "Subject": subject,
        "Task": task_name,
        "Trial": trial_name,
        "Acc": acc,
        "Gyr": gyr,
        "Mag": mag,
        "Timestamp": timestamp,
        "Time": time,
        "Fs": fs,
    }

# -----------------------------------------------------------------------------
# Function role: INDIP reference-annotation extraction
# -----------------------------------------------------------------------------
# This function reads the INDIP branch of the MATLAB structure and extracts
# ContinuousWalkingPeriod information. These annotations are the temporal
# reference used to create walking labels, walking-bout metadata, turn/path
# metadata and stair-related activity-detail labels.
# The output is intentionally converted into a list of Python dictionaries so
# that single-CWP and multiple-CWP trials can be handled with the same logic.

def extract_indip_annotations(trial_struct, task_name):
    """
    Extract INDIP reference annotations from one trial.

    This function is responsible only for reference information stored in
    Standards.INDIP. These annotations are later used to define labels
    and task-specific metadata for the dataset.
    """

    # data
    # └── TimeMeasure1
    #     └── Stairs
    #         └── Trial2
    #             ├── StartDateTime
    #             ├── TimeZone
    #             ├── SU_INDIP
    #             └── Standards
    #                 ├── BarometricData_raw
    #                 ├── Temperature_raw
    #                 ├── PressureInsoles_raw
    #                 ├── DistanceModule_raw
    #                 └── INDIP
    #                     ├── ContinuousWalkingPeriod
    #                     │   ├── [0]
    #                     │   │   ├── Start = 16.27
    #                     │   │   ├── End = 23.37
    #                     │   │   ├── Incline_Start = 16.27
    #                     │   │   ├── Incline_End = 23.37
    #                     │   │   └── ...
    #                     │   └── [1]
    #                     │       ├── Start = 41.87
    #                     │       ├── End = 48.27
    #                     │       ├── Incline_Start = 41.87
    #                     │       ├── Incline_End = 48.27
    #                     │       └── ...
    #                     └── Fs

    # The Standards branch does not necessarily contain INDIP annotations
    # for every task. Static tasks such as Test2 may not provide the same
    # reference structure used by dynamic gait tasks.
    standards = trial_struct.Standards

    # If INDIP is not available, no reference annotations can be extracted
    # for the current trial. In that case, None is returned and the trial
    # remains usable as signal-only data.
    if not hasattr(standards, "INDIP"):
        return None

    indip = standards.INDIP

    # ContinuousWalkingPeriod is the main annotation container currently
    # used to define walking segments and task-specific temporal intervals.
    if not hasattr(indip, "ContinuousWalkingPeriod"):
        return None

    cwp = indip.ContinuousWalkingPeriod

    # A list of annotation segments is created so that all tasks can be handled
    # with the same Python representation, independently of how MATLAB stored them.
    segments = []

    # Tasks such as Stairs or TUG contain multiple execution phases within
    # the same trial. In that case, scipy loads ContinuousWalkingPeriod as a
    # numpy array of MATLAB structs. In other words: if cwp is a struct-array (1x2 MATLAB struct)

    if isinstance(cwp, np.ndarray):
        cwp_elements = cwp
    else:
        # Tasks such as SixMinutesWalkingTest may contain a single continuous
        # walking block. In that case, the MATLAB struct is wrapped into a list
        # to keep a uniform format.
        cwp_elements = [cwp]

    # Each walking segment is converted into a Python dictionary with the main
    # fields currently relevant for gait sequence detection and task parsing.
    for i, elem in enumerate(cwp_elements):

        # Each walking segment is stored as a dictionary so that all task types
        # can later be handled with a common Python representation.
        segment_dict = {
            "segment_id": i,
            "start": getattr(elem, "Start", None),
            "end": getattr(elem, "End", None),

            "turn_start": getattr(elem, "Turn_Start", None),
            "turn_end": getattr(elem, "Turn_End", None),

            "incline_start": getattr(elem, "Incline_Start", None),
            "incline_end": getattr(elem, "Incline_End", None),
            "incline_duration": getattr(elem, "Incline_Duration", None),
            "incline_number": getattr(elem, "Incline_Number", None),
            "incline_number_strides": getattr(elem, "Incline_NumberStrides", None),
            "incline_positive_elevation": getattr(elem, "Incline_PositiveElevation", None),
            "incline_negative_elevation": getattr(elem, "Incline_NegativeElevation", None),

            "initial_contact_event": getattr(elem, "InitialContact_Event", None),
            "initial_contact_left_right": getattr(elem, "InitialContact_LeftRight", None),
            "final_contact_event": getattr(elem, "FinalContact_Event", None),
            "final_contact_left_right": getattr(elem, "FinalContact_LeftRight", None),
        }
        segments.append(segment_dict)

    # INDIP sampling frequency is stored separately from the sensor sampling
    # frequency and may become useful later when comparing annotations and signals.
    indip_fs = getattr(indip, "Fs", None)

    # A task-level annotation dictionary is returned.
    # The segments field always has the same Python format:
    # a list of dictionaries, even when only one segment exists.
    return {
        "Task": task_name,
        "ReferenceSystem": "INDIP",
        "INDIP_Fs": indip_fs,
        "ContinuousWalkingSegments": segments,
    }

# -----------------------------------------------------------------------------
# Function role: annotation quality and consistency check
# -----------------------------------------------------------------------------
# This function does not judge clinical correctness. Its purpose is to catch
# obvious technical problems before window generation, such as missing start
# or end times, inverted intervals, intervals outside the signal duration, or
# turn/incline annotations that are inconsistent with the parent walking bout.
# The result is stored in metadata so questionable trials can be reviewed.

def check_annotation_consistency(trial_data):
    """
    Perform a basic consistency check on INDIP annotations.

    The goal is not to validate the clinical correctness of the annotations,
    but to detect clearly invalid or unusable cases before window generation.
    """
    # A default report is created so that the output format is always stable,
    # even when no annotations are available for the current trial.
    report = {
        "has_annotations": False,
        "is_consistent": True,
        "issues": [],
    }

    annotations = trial_data.get("Annotations", None)

    # Trials without INDIP annotations are allowed at this stage because
    # some static or auxiliary tasks may still be useful for preprocessing.
    if annotations is None:
        report["issues"].append("No INDIP annotations available.")
        return report

    # If annotations is not None, has_annotations flag is set to True to
    # take into account a potential problem.
    report["has_annotations"] = True

    segments = annotations.get("ContinuousWalkingSegments", [])

    # Some static tasks may legitimately contain INDIP metadata without
    # annotated walking bouts. This is not considered an inconsistency.
    static_tasks = ["StandingBalance", "Test2"]

    task_name = trial_data.get("Task", None)

    # At least one segment in expected to be when considering a dynamic task.
    # When considering a static task, absence of walking segments is normal.
    if len(segments) == 0:
        if task_name in static_tasks:
            report["issues"].append(
                "No walking segments expected for static task."
            )
            return report
        else:
            report["is_consistent"] = False
            report["issues"].append(
                "Annotations are present but no walking segments were extracted."
            )
            return report

    # Trial duration is derived from the relative time axis of the LeftEar signal.
    time = trial_data.get("Time", None)
    if time is None or len(time) == 0:
        report["is_consistent"] = False
        report["issues"].append("Missing or empty trial time axis.")
        return report

    trial_end_time = float(time[-1])

    # Each walking segment is checked for basic temporal consistency.
    for segment in segments:
        segment_id = segment.get("segment_id", None)
        start = segment.get("start", None)
        end = segment.get("end", None)

        # Start and end must exist and define a valid positive interval.
        if start is None or end is None:
            report["is_consistent"] = False
            report["issues"].append(f"Segment {segment_id}: missing start or end.")
            continue

        if start >= end:
            report["is_consistent"] = False
            report["issues"].append(f"Segment {segment_id}: start is not smaller than end.")

        # The segment should lie within the temporal support of the signal.
        if start < 0 or end > trial_end_time:
            report["is_consistent"] = False
            report["issues"].append(
                f"Segment {segment_id}: interval [{start}, {end}] is outside trial duration [0, {trial_end_time:.2f}]."
            )

        # Turn and incline intervals, when available, should remain inside the
        # corresponding walking segment. Arrays are accepted for turn intervals.
        turn_start = segment.get("turn_start", None)
        turn_end = segment.get("turn_end", None)
        incline_start = segment.get("incline_start", None)
        incline_end = segment.get("incline_end", None)

        if turn_start is not None and turn_end is not None:
            turn_start_arr = np.atleast_1d(turn_start) # Convert inputs to arrays with at least one dimension
            turn_end_arr = np.atleast_1d(turn_end)

            if len(turn_start_arr) != len(turn_end_arr):
                report["is_consistent"] = False
                report["issues"].append(
                    f"Segment {segment_id}: turn_start and turn_end have different lengths."
                )
            else:
                for ts, te in zip(turn_start_arr, turn_end_arr):
                    if ts >= te:
                        report["is_consistent"] = False
                        report["issues"].append(
                            f"Segment {segment_id}: invalid turn interval [{ts}, {te}]."
                        )
                    if ts < start or te > end:
                        report["is_consistent"] = False
                        report["issues"].append(
                            f"Segment {segment_id}: turn interval [{ts}, {te}] falls outside walking segment [{start}, {end}]."
                        )

    if incline_start is not None and incline_end is not None:

        incline_start_arr = np.atleast_1d(incline_start)
        incline_end_arr = np.atleast_1d(incline_end)

        # Empty arrays are allowed and simply mean that no incline interval
        # was annotated for the current segment.
        if incline_start_arr.size == 0 or incline_end_arr.size == 0:
            pass

        elif incline_start_arr.size != incline_end_arr.size:
            report["is_consistent"] = False
            report["issues"].append(
                f"Segment {segment_id}: incline_start and incline_end have different lengths."
            )

        else:
            incline_duration = segment.get("incline_duration", None)
            incline_duration_arr = np.atleast_1d(incline_duration) if incline_duration is not None else None

            # Each inclined interval is checked separately because a segment may
            # contain one or more incline intervals.
            for incline_idx, (istart, iend) in enumerate(zip(incline_start_arr, incline_end_arr)):

                if istart >= iend:
                    report["is_consistent"] = False
                    report["issues"].append(
                        f"Segment {segment_id}: invalid incline interval [{istart}, {iend}]."
                    )

                if istart < start or iend > end:
                    report["is_consistent"] = False
                    report["issues"].append(
                        f"Segment {segment_id}: incline interval [{istart}, {iend}] falls outside walking segment [{start}, {end}]."
                    )

                # If duration is available, it should approximately match
                # incline_end - incline_start.
                if incline_duration_arr is not None and incline_duration_arr.size > 0:
                    stored_duration = (
                        incline_duration_arr[incline_idx]
                        if incline_idx < incline_duration_arr.size
                        else incline_duration_arr[0]
                    )

                    estimated_duration = iend - istart

                    if abs(stored_duration - estimated_duration) > 0.5:
                        report["issues"].append(
                            f"Segment {segment_id}: incline duration mismatch "
                            f"(stored={stored_duration}, estimated={estimated_duration:.2f})."
                        )

    return report


# -----------------------------------------------------------------------------
# Function role: sample-wise binary GSD label creation
# -----------------------------------------------------------------------------
# This function creates the main target used for Gait Sequence Detection.
# Every sample starts as static. Samples inside INDIP walking bouts are then
# marked as walking for dynamic tasks. Stairs are merged into generic walking
# at the binary GSD level, while stair direction is preserved separately in
# SampleActivityDetail.

def build_sample_labels(trial_data):
    """
    Create one label for each sample of the trial.

    Labels:
    0 = static
    1 = walking

    Expected INDIVI task-label logic:
    - Test2 and StandingBalance -> static
    - SixMinutesWalkingTest, UTurn, TUG -> walking inside INDIP walking segments
    - Stairs -> static + walking
    """

    time = trial_data["Time"]
    task = trial_data["Task"]
    annotations = trial_data["Annotations"]

    # Default label is static, not generic non-walking.
    # This is coherent with the selected INDIVI tasks, where the retained
    # non-locomotor condition corresponds to standing/static.
    labels = np.full(len(time), LABEL_MAP["static"], dtype=np.int32)

    # Static tasks are entirely labeled as static.
    if task in ["Test2", "StandingBalance"]:
        return labels

    # If annotations are missing for a dynamic task, keep static labels.
    # The consistency check will already report the annotation problem.
    if annotations is None:
        return labels

    segments = annotations["ContinuousWalkingSegments"]

    if len(segments) == 0:
        return labels

    if task in ["SixMinutesWalkingTest", "UTurn", "TUG"]:
        for seg in segments:
            start = seg["start"]
            end = seg["end"]

            idx = (time >= start) & (time <= end)
            labels[idx] = LABEL_MAP["walking"]

    elif task == "Stairs":
        for i, seg in enumerate(segments):
            start = seg["start"]
            end = seg["end"]

            idx = (time >= start) & (time <= end)

            # Stairs are merged into generic walking for GSD.
            # The original stair condition can still be recovered from task metadata.
            if i == 0:
                labels[idx] = LABEL_MAP["walking"]
            elif i == 1:
                labels[idx] = LABEL_MAP["walking"]
            else:
                labels[idx] = LABEL_MAP["walking"]

    return labels

# -----------------------------------------------------------------------------
# Function role: sample-wise activity-detail metadata
# -----------------------------------------------------------------------------
# This function preserves locomotor details that are not kept in the binary
# GSD label. For ordinary walking tasks it mirrors static/walking. For Stairs,
# it assigns ascending and descending labels based on the segment order.
# This makes it possible to train a binary walking detector while still
# keeping richer HAR-style metadata for later analysis.

def build_sample_activity_detail(trial_data):
    """
    Create HAR-level sample annotations preserving:
    - static
    - walking
    - ascending
    - descending

    This metadata is preserved independently from the
    binary GSD labels used for training.
    """

    time = trial_data["Time"]
    task = trial_data["Task"]
    annotations = trial_data["Annotations"]

    activity_detail = np.full(
        len(time),
        LABEL_MAP["static"],
        dtype=np.int32
    )

    if task in ["Test2", "StandingBalance"]:
        return activity_detail

    if annotations is None:
        return activity_detail

    segments = annotations["ContinuousWalkingSegments"]

    if len(segments) == 0:
        return activity_detail

    if task in ["SixMinutesWalkingTest", "UTurn", "TUG"]:

        for seg in segments:

            start = seg["start"]
            end = seg["end"]

            idx = (time >= start) & (time <= end)

            activity_detail[idx] = LABEL_MAP["walking"]

    elif task == "Stairs":

        for i, seg in enumerate(segments):

            start = seg["start"]
            end = seg["end"]

            idx = (time >= start) & (time <= end)

            if i == 0:
                activity_detail[idx] = 2

            elif i == 1:
                activity_detail[idx] = 3

            else:
                activity_detail[idx] = LABEL_MAP["walking"]

    return activity_detail

# -----------------------------------------------------------------------------
# Function role: sample-wise path-type metadata
# -----------------------------------------------------------------------------
# This function creates sample-wise straight/curved path metadata.
#
# Important convention:
# - controlled static tasks return None because path geometry is not meaningful;
# - static samples inside dynamic trials remain None;
# - walking samples are marked as straight by default;
# - walking samples inside INDIP Turn_Start / Turn_End intervals are marked as curved;
# - stairs are treated as straight locomotor samples, but only where the binary
#   GSD label indicates walking.
#
# This prevents static windows from incorrectly inheriting path_type = straight
# only because the trial contains walking elsewhere.

def build_sample_path_types(trial_data):
    """
    Create one path-type value for each sample of the trial.

    Path types:
    None = path type not applicable, used for static samples.
    0    = straight, used for walking samples outside turn intervals
    1    = curved, used for walking samples inside INDIP turn intervals

    Curved samples are identified using Turn_Start and Turn_End annotations.
    If no turn interval is available, all samples remain straight.

    INDIVI task-specific logic:
    - Test2 and StandingBalance are controlled static tasks, so path type is
      not meaningful and the function returns None.
    - Stairs (no turn samples for protocol acquisition) is a locomotor task
      performed along a straight path, so all samples are marked as straight.
    - SixMinutesWalkingTest, UTurn and TUG may contain both straight and curved
      portions. Straight is the default value, while curved samples are marked
      using INDIP Turn_Start and Turn_End annotations.
    """

    task = trial_data["Task"]
    time = trial_data["Time"]
    labels = trial_data["SampleLabels"]

    # Path type is not meaningful for controlled static tasks because
    # no locomotor path geometry exists during standing acquisitions.
    if task in ["Test2", "StandingBalance"]:
        return None

    # Initialize all samples as None.
    # This is important because path geometry is meaningful only during walking.
    # Static samples before, after or between walking bouts must not be interpreted
    # as straight.
    path_types = np.full(
        len(time),
        None,
        dtype=object
    )

    # Identify walking samples from the binary GSD labels.
    # These are the only samples for which path geometry is meaningful.
    walking_idx = labels == LABEL_MAP["walking"]

    # Walking samples are straight by default.
    # This also handles Stairs correctly: stair walking is locomotion along a
    # straight path, while non-walking samples in the same trial remain None.
    path_types[walking_idx] = 0

    # Stairs do not require turn parsing in the current INDIVI convention.
    # The walking part has already been marked as straight above.
    if task == "Stairs":
        return path_types

    annotations = trial_data["Annotations"]

    # If no annotations are available, walking samples remain straight and
    # static samples remain None.
    if annotations is None:
        return path_types

    segments = annotations["ContinuousWalkingSegments"]

    if len(segments) == 0:
        return path_types

    for seg in segments:
        turn_start = seg.get("turn_start", None)
        turn_end = seg.get("turn_end", None)

        if turn_start is None or turn_end is None:
            continue

        # Convert inputs to arrays with at least one dimension so that both
        # single-turn and multi-turn annotations are handled with the same code.
        turn_start_arr = np.atleast_1d(turn_start)
        turn_end_arr = np.atleast_1d(turn_end)

        if turn_start_arr.size == 0 or turn_end_arr.size == 0:
            continue

        # Only walking samples inside turn intervals are marked as curved.
        # The walking_idx condition prevents static samples around CWP borders
        # from receiving a locomotor path type.
        for ts, te in zip(turn_start_arr, turn_end_arr):
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
# Function role: sample-wise static-type metadata
# -----------------------------------------------------------------------------
# This function adds a more specific description for static samples.
# Controlled static acquisitions are marked as standing, while static samples
# belonging to dynamic tasks are marked as generic static. Walking samples
# remain marked as none because static type is not meaningful during walking.
#
# The "sitting" category is included in STATIC_TYPE_MAP for cross-dataset
# compatibility, mainly because WearGaitPD TUG can contain explicit sitting
# annotations. INDIVI does not provide sitting annotations here, so no sample is
# assigned to sitting in this dataset.

def build_sample_static_types(trial_data):
    """
    Create sample-wise static-type metadata.

    Static-type metadata provide a subtype only for samples whose main GSD
    label is static. Walking samples remain marked as "none", because a static
    subtype is not meaningful during locomotion.

    INDIVI task-specific logic:
    - Test2 and StandingBalance are controlled standing tasks, so their
      static samples are marked as "standing".
    - SixMinutesWalkingTest, UTurn, TUG and Stairs are dynamic tasks. Static
      samples outside ContinuousWalkingPeriods represent generic pauses,
      waiting periods, stops or non-walking portions, so they are marked as
      "static".
    - Samples inside ContinuousWalkingPeriods are walking samples and remain
      marked as "none" for static type.
    """

    labels = trial_data["SampleLabels"]
    task = trial_data["Task"]

    # Default value is "none".
    # This is correct for walking samples because static type is not applicable.
    static_type = np.full(len(labels), STATIC_TYPE_MAP["none"], dtype=np.int32)

    static_idx = labels == LABEL_MAP["static"]

    # Controlled static tasks are treated as standing.
    if task in ["Test2", "StandingBalance"]:
        static_type[static_idx] = STATIC_TYPE_MAP["standing"]

    # In dynamic tasks, static samples are generic static/non-walking periods
    # outside the annotated ContinuousWalkingPeriods.
    else:
        static_type[static_idx] = STATIC_TYPE_MAP["static"]

    return static_type


# -----------------------------------------------------------------------------
# Function role: gait-phase metadata placeholder
# -----------------------------------------------------------------------------
# This function keeps the dataset compatible with pipelines where gait
# initiation, steady state and termination may be explicitly available.
#
# In INDIP-derived datasets, ContinuousWalkingPeriod annotations are already
# selective and may exclude the first and last steps of a walking bout.
# Therefore, gait initiation and gait termination are not inferred here.
# All samples are marked as none.

def build_sample_gait_phases(trial_data):
    """
    Build sample-wise gait-phase metadata.

    In INDIP-derived datasets, gait initiation and termination are not
    reliably annotated. Therefore, all samples are marked as none.
    """

    labels = trial_data["SampleLabels"]

    return np.full(
        len(labels),
        GAIT_PHASE_MAP["none"],
        dtype=np.int32
    )

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
    Mark static samples immediately before and after each CWP boundary.
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

        pre_idx = (time >= start - margin_s) & (time < start)
        post_idx = (time > end) & (time <= end + margin_s)

        transition_types[
            pre_idx & (labels == LABEL_MAP["static"])
        ] = "Static2Walking"

        transition_types[
            post_idx & (labels == LABEL_MAP["static"])
        ] = "Walking2Static"

    return transition_types

# -----------------------------------------------------------------------------
# Function role: sample-to-walking-bout association
# -----------------------------------------------------------------------------
# This function assigns each sample to its ContinuousWalkingPeriod segment.
# Samples outside walking bouts receive -1. Samples inside a walking bout receive
# the segment_id extracted from INDIP annotations.
#
# This metadata is not used as a training label. It is only used for traceability,
# bout-level analysis, debugging of mixed windows, and reconstruction of the
# temporal origin of each window.

def build_sample_walking_bouts(trial_data):
    """
    Assign one walking-bout ID to each sample.

    Samples outside ContinuousWalkingPeriods are marked as -1.
    Samples inside a walking bout receive the corresponding INDIP segment_id.
    """

    time = trial_data["Time"]
    annotations = trial_data["Annotations"]

    # Default value:
    # -1 means that the sample does not belong to any walking bout.
    walking_bouts = np.full(len(time), -1, dtype=int)

    # Static tasks or trials without annotations remain entirely set to -1.
    if annotations is None:
        return walking_bouts

    segments = annotations["ContinuousWalkingSegments"]

    if len(segments) == 0:
        return walking_bouts

    for seg in segments:

        start = seg.get("start", None)
        end = seg.get("end", None)
        segment_id = seg.get("segment_id", None)

        if start is None or end is None or segment_id is None:
            continue

        # Samples inside the current INDIP ContinuousWalkingPeriod receive
        # the corresponding walking-bout identifier.
        idx = (time >= start) & (time <= end)
        walking_bouts[idx] = int(segment_id)

    return walking_bouts

# -----------------------------------------------------------------------------
# Function role: standardized trial-level field creation
# -----------------------------------------------------------------------------
# This function keeps the original INDIVI keys but also adds lower-case fields
# with names aligned to WearGaitPD and other datasets. This is important because
# downstream feature extraction and ML scripts can use the same key names
# independently of the original dataset.

def update_standard_trial_fields(trial_data):
    """
    Add a standardized trial-level representation.

    The original INDIVI keys are preserved because the construction
    functions still use them internally. Additional lowercase keys are added
    to match the WearGaitPD trial dataset format as closely as possible.

    This makes the trial dataset easier to reuse in the generic ml_pipeline.
    """

    acc = trial_data["Acc"]
    gyr = trial_data["Gyr"]

    # X always contains the 6-channel IMU representation used by the
    # downstream pipeline: accelerometer + gyroscope.
    trial_data["X"] = np.hstack([acc, gyr])

    # Lowercase metadata fields aligned with WearGaitPD.
    trial_data["dataset"] = trial_data["Dataset"]
    trial_data["subject_id"] = trial_data["Subject"]
    trial_data["task"] = trial_data["Task"]
    trial_data["test_type"] = trial_data.get(
        "TestType", None)

    # Challenge type is a dataset-compatible metadata field.
    # In INDIVI no specific challenge condition is currently encoded,
    # so all trials are marked as "none".
    trial_data["challenge_type"] = trial_data.get("ChallengeType", "none")

    trial_file_name = f"{trial_data['Subject']}_data.mat"

    trial_data["file_name"] = trial_data.get("FileName", trial_file_name)
    trial_data["file_path"] = trial_data.get("FilePath", None)

    # Store a project-relative path for portable metadata.
    # This avoids making the constructed dataset dependent on a specific Windows
    # user folder, while still preserving file provenance.
    trial_data["relative_file_path"] = trial_data.get(
        "RelativeFilePath",
        str(Path("data") / "INDIVI" / "ALL data" / trial_data["file_name"])
    )

    # Use the accelerometer sampling frequency as the common sampling rate.
    # INDIVI Acc/Gyr/Mag are expected to be sampled at 100 Hz.
    trial_data["fs"] = trial_data["Fs"]["Acc"]

    # labels are the current sample-level labels used by the pipeline.
    # labels_har preserves the original HAR-level classes.
    # Binary GSD labels.
    trial_data["labels"] = trial_data["SampleLabels"]

    # HAR-level labels preserving ascending/descending.
    trial_data["labels_har"] = trial_data["SampleActivityDetail"]

    # activity_detail preserves the original locomotion class
    # before GSD merging.
    trial_data["activity_detail"] = trial_data["SampleActivityDetail"]

    trial_data["static_type"] = trial_data["SampleStaticTypes"]
    trial_data["gait_phase"] = trial_data["SampleGaitPhases"]
    trial_data["path_types"] = trial_data["SamplePathTypes"]
    trial_data["transition_types"] = trial_data["SampleTransitionTypes"]

    # Walking-bout metadata aligned with TOWalk and MOVEWISE.
    # It is useful for bout-level analyses and for tracing each sample back to the
    # corresponding INDIP ContinuousWalkingPeriod.
    trial_data["walking_bouts"] = trial_data["SampleWalkingBouts"]

    # Store both generic and explicit GSD map names.
    # This keeps the trial-level dictionary compatible with scripts that expect
    # either label_map or gsd_label_map.
    trial_data["label_map"] = GSD_LABEL_MAP
    trial_data["gsd_label_map"] = GSD_LABEL_MAP

    trial_data["har_label_map"] = ACTIVITY_DETAIL_MAP
    trial_data["activity_detail_map"] = ACTIVITY_DETAIL_MAP

    trial_data["static_type_map"] = STATIC_TYPE_MAP
    trial_data["gait_phase_map"] = GAIT_PHASE_MAP
    trial_data["path_type_map"] = PATH_TYPE_MAP

    trial_data["transition_definition"] = TRANSITION_DEFINITION

    # INDIVI group information is not currently encoded here.
    # The field is still kept for compatibility with WearGaitPD.
    trial_data["group"] = trial_data.get("Group", "UNKNOWN")

    return trial_data

# -----------------------------------------------------------------------------
# Function role: sliding-window construction from one trial
# -----------------------------------------------------------------------------
# This function splits one continuous trial into fixed-length windows.
# Each window contains accelerometer and gyroscope channels concatenated into
# a 6-channel signal. Labels and metadata are assigned by majority voting or
# window-level summaries. This is the key step that transforms continuous
# recordings into samples usable for machine-learning training.

def build_windows_from_trial(trial_data, window_size=200, step_size=100):
    """
    Build fixed-length windows from one INDIVI trial.

    The final window label remains binary for Gait Sequence Detection:
    - static
    - walking

    However, the internal composition of each window is now preserved through
    additional metadata. This is important because a window can be finally
    labelled as walking while still containing a relevant static portion, or
    vice versa.

    The function stores:
    - static/walking/none sample counts;
    - static/walking/none fractions;
    - final-label fraction;
    - robust-window flag;
    - straight/curved path-type composition;
    - static-context composition;
    - transition metadata;
    - walking-bout metadata.
    """

    # Accelerometer and gyroscope are concatenated to form the final
    # 6-channel representation expected by the current pipeline.
    acc = trial_data["Acc"]
    gyr = trial_data["Gyr"]

    # Sample-wise binary GSD labels:
    # 0 = static
    # 1 = walking
    sample_labels = trial_data["SampleLabels"]

    # Sample-wise detailed activity labels:
    # 0 = static
    # 1 = walking
    # 2 = ascending
    # 3 = descending
    #
    # These labels are metadata. They do not replace the binary GSD label.
    sample_activity_detail = trial_data["SampleActivityDetail"]

    # Sample-wise path-type metadata:
    # None = path type not applicable
    # 0    = straight
    # 1    = curved
    sample_path_types = trial_data["SamplePathTypes"]

    # Sample-wise static subtype:
    # -1 = none
    #  0 = generic static
    #  1 = standing
    #  2 = sitting
    sample_static_types = trial_data["SampleStaticTypes"]

    # Sample-wise gait phase metadata.
    # In INDIVI this is usually none, because gait initiation and termination
    # are not explicitly inferred from INDIP annotations.
    sample_gait_phases = trial_data["SampleGaitPhases"]

    # Sample-wise transition metadata around ContinuousWalkingPeriod boundaries.
    # These metadata do not change the sample labels.
    sample_transition_types = trial_data["SampleTransitionTypes"]

    # Sample-wise walking-bout identifier.
    # -1 means outside a walking bout.
    # Non-negative values correspond to INDIP ContinuousWalkingPeriod segment IDs.
    sample_walking_bouts = trial_data["SampleWalkingBouts"]

    time = trial_data["Time"]

    # ------------------------------------------------------------------
    # Task-specific windowing offset
    # ------------------------------------------------------------------
    # For StandingBalance, only the last 30 seconds are used according to
    # the current INDIVI dataset protocol.
    # All other tasks are windowed over the full trial.
    start_offset = 0

    if trial_data["Task"] == "StandingBalance":
        trial_end_time = time[-1]
        standing_start_time = max(0, trial_end_time - 30)

        # Convert the selected starting time into a sample index.
        start_offset = int(
            np.searchsorted(time, standing_start_time)
        )

    # Build the 6-channel input signal:
    # columns 0-2 = Acc
    # columns 3-5 = Gyr
    signal_6ch = np.hstack([acc, gyr])

    windows = []

    # ------------------------------------------------------------------
    # Sliding-window segmentation
    # ------------------------------------------------------------------
    # Each iteration extracts one fixed-length window.
    # step_size controls the overlap between consecutive windows.
    for start_idx in range(
        start_offset,
        len(signal_6ch) - window_size + 1,
        step_size
    ):

        end_idx = start_idx + window_size

        window_signal = signal_6ch[start_idx:end_idx]
        window_labels = sample_labels[start_idx:end_idx]
        window_activity_detail = sample_activity_detail[start_idx:end_idx]
        window_static_types = sample_static_types[start_idx:end_idx]
        window_gait_phases = sample_gait_phases[start_idx:end_idx]
        window_transition_types = sample_transition_types[start_idx:end_idx]
        window_walking_bouts = sample_walking_bouts[start_idx:end_idx]
        window_time = time[start_idx:end_idx]

        # Path type is available only for tasks where path geometry is meaningful.
        # Static tasks may have sample_path_types = None.
        if sample_path_types is not None:
            window_path_types = sample_path_types[start_idx:end_idx]
        else:
            window_path_types = None

        # ------------------------------------------------------------------
        # Final binary GSD label by majority voting
        # ------------------------------------------------------------------
        # The final class of the window is still assigned by majority voting.
        # This keeps the original GSD construction logic unchanged.
        unique_labels, counts = np.unique(
            window_labels,
            return_counts=True
        )

        majority_label = int(
            unique_labels[np.argmax(counts)]
        )

        # ------------------------------------------------------------------
        # Internal label composition
        # ------------------------------------------------------------------
        # These counters describe the full internal composition of the window.
        # They are not used as signal features.
        #
        # They are useful later for:
        # - filtering robust windows during training;
        # - separating clean windows from ambiguous windows;
        # - analyzing misclassified windows;
        # - identifying transition windows.
        n_static_samples = int(
            np.sum(window_labels == LABEL_MAP["static"])
        )

        n_walking_samples = int(
            np.sum(window_labels == LABEL_MAP["walking"])
        )

        # INDIVI usually has no -1/none labels in SampleLabels.
        # However, the field is still created for compatibility with WearGaitPD.
        # Any sample that is neither static nor walking is counted here.
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

        # The final-label fraction tells how much of the full window is covered
        # by the class selected through majority voting.
        #
        # Example:
        # majority_label = walking
        # walking_fraction = 0.85
        # final_label_fraction = 0.85
        if majority_label == LABEL_MAP["static"]:
            final_label_fraction = static_fraction

        elif majority_label == LABEL_MAP["walking"]:
            final_label_fraction = walking_fraction

        else:
            final_label_fraction = 0.0

        # The old walking/static ratio is kept only as diagnostic metadata.
        # It is no longer used to decide whether the window is robust.
        #
        # This avoids misleading cases where the ratio becomes very large only
        # because the number of static samples is close to zero.
        eps = 1e-6

        walking_static_ratio = (
            (n_walking_samples + eps)
            /
            (n_static_samples + eps)
        )

        # A window is robust only if the final class covers at least 80% of the
        # complete window, or whatever threshold is defined in
        # WINDOW_LABEL_MIN_FRACTION.
        is_robust_window = (
            final_label_fraction >= WINDOW_LABEL_MIN_FRACTION
        )

        # ------------------------------------------------------------------
        # Static-context composition
        # ------------------------------------------------------------------
        # This block preserves static subtype information even in mixed windows.
        #
        # Example:
        # a final walking window can still contain 20% standing samples.
        # In that case:
        # - final static_type remains none, because the window is walking;
        # - standing_static_type_fraction stores the internal standing component.
        static_context_summary = summarize_window_static_context(
            static_type_win=window_static_types,
            window_size=window_size,
            min_fraction=STATIC_CONTEXT_MIN_FRACTION
        )

        # ------------------------------------------------------------------
        # Robust path-type composition
        # ------------------------------------------------------------------
        # Path type is not assigned by simple majority voting anymore.
        #
        # A window is classified as straight or curved only if that path type
        # covers at least PATH_TYPE_MIN_FRACTION of the full window.
        #
        # If the threshold is not reached, the final path_type remains none,
        # but the straight/curved fractions are still saved in metadata.
        if window_path_types is None:
            path_win_for_summary = np.array(
                [None] * window_size,
                dtype=object
            )
        else:
            path_win_for_summary = window_path_types

        path_summary = summarize_window_path_type(
            path_win=path_win_for_summary,
            window_size=window_size,
            min_fraction=PATH_TYPE_MIN_FRACTION
        )

        # ------------------------------------------------------------------
        # Transition metadata
        # ------------------------------------------------------------------
        # Transition type is derived from sample-wise transition annotations.
        # This does not change the final binary label.
        valid_transition_types = [
            t for t in window_transition_types
            if t is not None and t != "None"
        ]

        if len(valid_transition_types) == 0:
            # Store the string "None" instead of Python None to keep CSV metadata
            # explicit and consistent with the other construction scripts.
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

        # ------------------------------------------------------------------
        # Label-consistent final metadata assignment
        # ------------------------------------------------------------------
        # The window-level metadata must remain semantically consistent with
        # the final binary GSD label.
        #
        # Therefore:
        # - final walking windows can have path_type and walking_bout;
        # - final walking windows must have static_type = none;
        # - final static windows can have static_type;
        # - final static windows must have path_type = none and walking_bout = -1.
        if majority_label == LABEL_MAP["walking"]:

            # Path type is meaningful only for locomotor windows.
            # It is assigned only if the robust path-type threshold is satisfied.
            final_path_type = path_summary["final_path_type"]

            # Static subtype is not applicable to a final walking window.
            # Static content, if present, is preserved separately in
            # static_context metadata.
            final_static_type = STATIC_TYPE_MAP["none"]

            # Gait phase is meaningful only for walking windows.
            # In INDIVI it will usually remain none, but this block keeps the
            # function compatible with the generic pipeline.
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

            # Activity detail must remain consistent with a walking final label.
            # Therefore, only locomotor details are allowed to vote here:
            # walking, ascending, descending.
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
                # Safety fallback:
                # if the final label is walking but no locomotor activity detail
                # is found, the window is treated as generic walking.
                final_activity_detail = ACTIVITY_DETAIL_MAP["walking"]

            # Walking-bout ID is meaningful only for walking windows.
            # Values equal to -1 are ignored because they represent samples outside
            # any INDIP ContinuousWalkingPeriod.
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

        elif majority_label == LABEL_MAP["static"]:

            # Path type and gait phase are locomotor metadata.
            # They are therefore not applicable to a final static window.
            final_path_type = None
            final_gait_phase = GAIT_PHASE_MAP["none"]
            final_walking_bout = -1

            # Static type is meaningful only for final static windows.
            # Values equal to none are ignored because they correspond mainly
            # to walking samples inside mixed windows.
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
                # Safety fallback:
                # if the final label is static but no static subtype is available,
                # the window is treated as generic static.
                final_static_type = STATIC_TYPE_MAP["static"]

            # A final static window must also have static activity detail.
            final_activity_detail = ACTIVITY_DETAIL_MAP["static"]

        else:

            # Safety fallback for unexpected labels.
            # This branch should not be reached in the current INDIVI pipeline,
            # because SampleLabels are expected to contain only static/walking.
            final_path_type = None
            final_static_type = STATIC_TYPE_MAP["none"]
            final_gait_phase = GAIT_PHASE_MAP["none"]
            final_activity_detail = ACTIVITY_DETAIL_MAP["static"]
            final_walking_bout = -1

        # ------------------------------------------------------------------
        # Final window dictionary
        # ------------------------------------------------------------------
        # The dictionary stores both:
        # - the signal and final labels used by the ML pipeline;
        # - rich metadata for traceability and misclassification analysis.
        window_dict = {
            "Dataset": trial_data["Dataset"],
            "Subject": trial_data["Subject"],
            "Task": trial_data["Task"],
            "TestType": trial_data.get("TestType", "unknown"),
            "ChallengeType": trial_data.get("challenge_type", "none"),
            "Trial": trial_data["Trial"],

            # File provenance metadata.
            # file_path may remain None because the portable field is relative_file_path.
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

            # Final binary GSD label.
            "SampleLabels": window_labels,
            "WindowLabel": int(majority_label),

            # Final activity-detail metadata, kept consistent with WindowLabel.
            "SampleActivityDetail": window_activity_detail,
            "WindowActivityDetail": int(final_activity_detail),

            # ----------------------------------------------------------
            # Label-composition metadata
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # Path-type metadata
            # ----------------------------------------------------------
            "SamplePathTypes": window_path_types,
            "WindowPathType": (
                None
                if final_path_type is None
                else int(final_path_type)
            ),

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

            # ----------------------------------------------------------
            # Static-context metadata
            # ----------------------------------------------------------
            "WindowStaticType": int(final_static_type),

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

            # ----------------------------------------------------------
            # Transition metadata
            # ----------------------------------------------------------
            "WindowIsTransition": int(window_is_transition),
            "WindowTransitionType": window_transition_type,
            "SampleTransitionTypes": window_transition_types,

            # ----------------------------------------------------------
            # Walking-bout and gait-phase metadata
            # ----------------------------------------------------------
            "SampleWalkingBouts": window_walking_bouts,
            "WindowWalkingBout": int(final_walking_bout),

            "WindowGaitPhase": int(final_gait_phase),

            # INDIVI currently has no diagnostic group in this script.
            # Keep UNKNOWN unless a real group mapping is added later.
            "Group": trial_data.get("Group", "UNKNOWN"),
        }

        windows.append(window_dict)

    return windows


# -----------------------------------------------------------------------------
# Function role: gravity alignment preprocessing
# -----------------------------------------------------------------------------
# This function applies trial-wise gravity alignment using the first second of
# each recording as the static reference. The goal is to reduce orientation
# variability between trials and subjects by rotating acceleration and angular
# velocity into a consistent gravity-aligned frame.

def apply_gravity_alignment_to_dataset(dataset, static_duration_s=1.0):
    """
    Apply gravity alignment independently to each trial.

    The input dataset is a flat list of trial dictionaries.
    This keeps the format compatible with the generic ml_pipeline scripts.
    """

    for trial_data in dataset:

        subject = trial_data["Subject"]
        task = trial_data["Task"]
        trial = trial_data["Trial"]

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

        print(f"\n[ALIGNMENT CHECK] Subject {subject} | Task {task} | Trial {trial}")
        print("Alignment reference: first_1s_of_current_recording")
        print("Mean aligned acceleration:", mean_static_acc)
        print("Normalized mean aligned acceleration:", mean_static_acc_norm)

        trial_data["RawAcc"] = acc
        trial_data["RawGyr"] = gyr
        trial_data["Acc"] = acc_aligned
        trial_data["Gyr"] = gyr_aligned
        # Add WearGaitPD-like trial-level fields.
        # This does not remove the original INDIVI fields; it only adds a
        # standardized representation for the generic pipeline.
        trial_data = update_standard_trial_fields(trial_data)
        trial_data["RotationMatrix"] = R

        trial_data["GravityAlignment"] = {
            "reference": "first_1s_of_current_recording",
            "type": "self_initial_segment",
            "duration_s": static_duration_s,
            "mean_static_acc": mean_static_acc.tolist(),
            "mean_static_acc_norm": mean_static_acc_norm.tolist(),
            "gravity_ideal": [1.0, 0.0, 0.0],
        }

    return dataset

# -----------------------------------------------------------------------------
# Function role: window generation for the full dataset
# -----------------------------------------------------------------------------
# This function loops over all trial dictionaries and applies the single-trial
# windowing function. The resulting list contains all windows from all subjects,
# tasks and trials.

def build_window_dataset(dataset, window_size=200, step_size=100):
    """
    Build the full window-based dataset from a flat trial dataset.

    Each element of dataset is one trial dictionary.
    Each output element is one fixed-length window.
    """

    window_dataset = []

    for trial_data in dataset:

        trial_windows = build_windows_from_trial(
            trial_data,
            window_size=window_size,
            step_size=step_size
        )

        window_dataset.extend(trial_windows)

    return window_dataset

# -----------------------------------------------------------------------------
# Function role: conversion to ML-ready arrays and metadata
# -----------------------------------------------------------------------------
# This function converts the list of window dictionaries into:
# - X: numerical input array with shape (windows, samples, channels)
# - y: class-label vector
# - metadata: list of dictionaries for traceability and analysis.
# Keeping metadata aligned across datasets is essential for later splits,
# diagnostics and performance analysis by task/path/subject.

def convert_windows_to_arrays(window_dataset):
    """
    Convert the list-based window dataset into arrays ready for model training.

    X has shape (num_windows, window_size, num_channels).
    y contains one class label for each window.
    Metadata uses the same field names adopted in WearGaitPD.
    """

    X = np.stack([w["Signal"] for w in window_dataset], axis=0)
    y = np.array([w["WindowLabel"] for w in window_dataset], dtype=int)

    metadata = []

    for w in window_dataset:
        # Metadata structure aligned with WearGaitPD.
        # The same field names are reused across datasets
        # to keep the ml_pipeline dataset-independent.
        metadata.append({
            "dataset": w.get("Dataset", "INDIVI"),
            "subject_id": w["Subject"],
            "task": w["Task"],
            "test_type": w.get("TestType", "unknown"),
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

            "label": w["WindowLabel"],
            "label_name": LABEL_NAME_MAP.get(
                w["WindowLabel"],
                "unknown"
            ),

            "activity_detail": w["WindowActivityDetail"],
            "activity_detail_name": ACTIVITY_DETAIL_NAME_MAP.get(
                w["WindowActivityDetail"],
                "unknown"
            ),

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

            "is_transition": bool(w.get("WindowIsTransition", 0)),
            "transition_type": w.get("WindowTransitionType", "None"),
            "transition_definition": w.get(
                "TransitionDefinition",
                TRANSITION_DEFINITION
            ),

            "walking_bout": w["WindowWalkingBout"],

            "gait_phase": w["WindowGaitPhase"],
            "gait_phase_name": GAIT_PHASE_NAME_MAP.get(
                w["WindowGaitPhase"],
                "none"
            ),

            "group": w["Group"],
        })

    return X, y, metadata

# -----------------------------------------------------------------------------
# Function role: subject-level INDIVI dataset construction
# -----------------------------------------------------------------------------
# This function processes all selected tasks and trials for one subject.
# For each trial it extracts signals, extracts INDIP annotations, builds labels
# and metadata, standardizes fields and performs quality checks.
# The output is still nested by task and trial, then flattened later.

def build_subject_dataset(data_struct, subject, selected_tasks=None):
    """
    Build a structured dataset for one subject, given as input to the function.

    Parameters
    data_struct: mat_struct
        Top-level MATLAB 'data' structure for one subject.
    subject: subject ID (3-numeric digits)
    selected_tasks: list or None
        If provided, only these tasks are processed.

    Returns
    subject_dataset: flat list
        organized as subject_dataset.trial data
    """
    subject_dataset = []

    # data_struct is a MATLAB structure, then the access to its fields is
    # possible using (data_struct.field).
    # Extract time_measure (contained in TimeMeasure1 field) in datenum format, yet.
    time_measure = data_struct.TimeMeasure1
    # Extract available_tasks using ._fieldnames (built-in function in Python
    # that allows access to the fields contained in a structure).
    available_tasks = time_measure._fieldnames

    # Loop over all tasks available inside TimeMeasure1.
    for task_name in available_tasks:
        # If a task selection is provided and the name of the current task is
        # not contained into the list, skip this loop for the current task.
        if selected_tasks is not None and task_name not in selected_tasks:
            continue

        # Skip calibration information, since it is not a gait task.
        if task_name == "infoForCalibration":
            continue

        # Access the current task contained inside time_measure, as a MATLAB
        # struct (e.g. time_measure.TUG)
        task_struct = getattr(time_measure, task_name)

        # Avoid processing invalid or non-structured entries.
        # If task_struct has no fields, skip this processing for the current task.
        if not hasattr(task_struct, "_fieldnames"):
            continue

        # Initialize a flat list to store all valid trials of the current task.
        task_trials = []

        # Loop over all trials available inside task_struct (Trial1, Trial2).
        for trial_name in task_struct._fieldnames:

            # Access task_struct.trial_name (e.g. TUG.Trial1) which is a structure.
            trial_struct = getattr(task_struct, trial_name)

            # Try to extract left ear IMU signals and metadata and store them
            # into a dictionary called task_trials, specific for each task.
            try:
                # Sensor inputs and reference annotations are extracted separately
                # and then merged into a single trial-level dictionary.
                signal_data = extract_left_ear_signals(
                    trial_struct=trial_struct,
                    subject=subject,
                    task_name=task_name,
                    trial_name=trial_name
                )

                annotation_data = extract_indip_annotations(
                    trial_struct=trial_struct,
                    task_name=task_name
                )

                # A single trial dictionary is built by combining signal-derived inputs
                # with INDIP reference annotations and trial metadata.
                trial_data = {
                    # **signal_data serves the purpose of copying into trial_data
                    # dictionary all the couples key-values inside signal_data.
                    **signal_data,
                    "Annotations": annotation_data}

                # TestType is stored explicitly to keep metadata consistent across datasets.
                # In INDIVI, it corresponds to the experimental task name, e.g. TUG, Stairs, or SixMinutesWalkingTest.
                trial_data["TestType"] = task_name

                # A basic consistency check is also performed on the extracted
                # reference annotations. This helps identify missing or invalid
                # temporal intervals before window generation.
                annotation_qc = check_annotation_consistency(trial_data)
                trial_data["annotation_check"] = annotation_qc

                # One label is assigned to each sample of the trial using
                # the extracted INDIP temporal annotations and the task name.
                # The resulting vector has the same length as the time axis.
                trial_data["SampleLabels"] = build_sample_labels(trial_data)

                # HAR-level labels preserving ascending/descending.
                # These labels are stored separately from binary GSD labels.
                trial_data["SampleActivityDetail"] = build_sample_activity_detail(trial_data)

                # One path-type value is assigned to each sample when path geometry is meaningful.
                # Static tasks return None because no locomotor path exists.
                # In dynamic trials, static samples remain None, while walking samples are
                # marked as straight by default.
                # Stairs walking samples are straight because they represent locomotion
                # without curved-path annotations.
                # SixMinutesWalkingTest, UTurn and TUG can contain curved walking, so
                # samples inside Turn_Start/Turn_End intervals are marked as curved.
                trial_data["SamplePathTypes"] = build_sample_path_types(trial_data)

                trial_data["SampleStaticTypes"] = build_sample_static_types(trial_data)
                trial_data["SampleGaitPhases"] = build_sample_gait_phases(trial_data)

                trial_data["SampleTransitionTypes"] = build_sample_transition_types(
                    trial_data,
                    margin_s=1.0
                )

                # One walking-bout ID is assigned to each sample.
                # Samples outside INDIP ContinuousWalkingPeriods receive -1.
                # Samples inside a CWP receive the corresponding segment_id.
                # This is auxiliary metadata and is not used as a training label.
                trial_data["SampleWalkingBouts"] = build_sample_walking_bouts(trial_data)

                # Add WearGaitPD-like trial-level fields.
                # This does not remove the original INDIVI fields; it only adds a
                # standardized representation for the generic ml_pipeline.
                trial_data = update_standard_trial_fields(trial_data)

                # quality check (qc) result is performed using check_trial_quality
                # function that computes complete checkup on the signals avoiding
                # potential wrong format signal analysis.
                qc = check_trial_quality(trial_data, expected_fs=100, min_duration_sec=3.0)

                # The trial is kept only if the signal quality check is passed.
                # Annotation consistency is stored as additional information and
                # can later be used to decide which tasks are suitable for
                # label generation.
                if qc["is_valid"]:
                    trial_data["quality_check"] = qc
                    subject_dataset.append(trial_data)

                    if not annotation_qc["is_consistent"]:
                        print(
                            f"[WARNING] {task_name} - {trial_name} | "
                            f"Inconsistent annotations: {annotation_qc['issues']}"
                        )
                else:
                    print(f"[WARNING] {task_name} - {trial_name} | {summarize_quality_report(qc)}")

            except Exception as e:
                # If extraction fails, print an error and skip the current trial extraction.
                print(f"[WARNING] Failed to extract {task_name} - {trial_name}: {e}")

    # Final subject_dataset structure:
    #
    # subject_dataset = [
    #     trial_data_1,
    #     trial_data_2,
    #     trial_data_3,
    #     ...
    # ]
    #
    # Each element of the list corresponds to one valid trial and already
    # contains:
    # - signals
    # - annotations
    # - labels
    # - metadata
    # - preprocessing outputs
    #
    # This flat representation is consistent with the generic ML pipeline
    # and avoids additional nested-dictionary flattening steps.

    return subject_dataset


# -----------------------------------------------------------------------------
# Function role: full INDIVI trial-level dataset construction
# -----------------------------------------------------------------------------
# This function loops over all selected subjects, loads their MATLAB files and
# calls the subject-level builder. The nested subject output is flattened into
# a list of trial dictionaries to match the structure used by the generic ML
# pipeline.

def build_dataset(root_path, selected_subjects=None, selected_tasks=None):
    """
    Build the dataset across multiple subjects.

    Returns
    dataset : list
        dataset.append(trial_data)
    """
    # ------------------------------------------------------------
    # Trial dataset container
    # ------------------------------------------------------------
    # A flat list structure is used instead of nested dictionaries.
    #
    # This keeps the dataset fully compatible with:
    # - feature computation;
    # - train/test split;
    # - feature ranking;
    # - wrapper selection;
    #
    # Each list element corresponds to one trial.

    dataset = []

    # Extract subject list from root path.
    subjects = get_subject_list(root_path)

    # If is provided a list of selected subjects, subjects is transformed
    # into selected_subjects.
    if selected_subjects is not None:
        subjects = [s for s in subjects if s in selected_subjects]

    # Loop over all subjects in the subject list.
    for subject in subjects:
        print(f"Processing subject {subject}...")

        # Try to load subject data structure and to build subject
        # task-specific dataset.
        try:
            data_struct = load_subject_mat(root_path, subject)

            # build_subject_dataset still returns a task/trial nested dictionary.
            # Here it is flattened into a list so that the final dataset has the
            # same trial-level structure used by the generic pipeline.
            subject_dataset = build_subject_dataset(
                data_struct=data_struct,
                subject=subject,
                selected_tasks=selected_tasks
            )

            dataset.extend(subject_dataset)

        except Exception as e:
            print(f"[WARNING] Failed to process subject {subject}: {e}")

    return dataset


# -----------------------------------------------------------------------------
# Function role: trial-level dataset summary
# -----------------------------------------------------------------------------
# This function prints the number of subjects, tasks and valid trials.
# It is a simple but important sanity check after raw-data loading and quality
# filtering.

def summarize_dataset(dataset):
    """
    Print a simple summary of the flat trial dataset.
    """

    n_trials = len(dataset)

    subjects = sorted({
        trial_data["Subject"]
        for trial_data in dataset
    })

    tasks = sorted({
        trial_data["Task"]
        for trial_data in dataset
    })

    print("\n----- DATASET SUMMARY -----")
    print("Number of subjects:", len(subjects))
    print("Number of tasks:", len(tasks))
    print("Number of valid trials:", n_trials)

    print("\nTasks:")
    for task in tasks:
        n_task_trials = sum(
            trial_data["Task"] == task
            for trial_data in dataset
        )
        print(f"  {task}: {n_task_trials} trials")

# -----------------------------------------------------------------------------
# Function role: dataset serialization
# -----------------------------------------------------------------------------
# This function saves Python objects to disk using pickle.
# Pickle is useful here because the dataset contains dictionaries, numpy arrays
# and nested metadata that should be reloaded without reconstruction.

def save_dataset(dataset, save_path):
    """
    Save the constructed dataset as a pickle file.
    """
    # pickle format allows easier access to the dataset.
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {save_path}")

def check_indivi_path_types(dataset):
    """
    Check path-type assignment for INDIVI tasks.

    The input dataset is a flat list of trial dictionaries.

    Important convention:
    - Test2 and StandingBalance are static tasks, therefore path type is not
      meaningful and should contain only None values.
    - Stairs can contain static portions and stair-walking portions. Static
      samples may be None, while stair-walking samples are expected to be
      straight, because straight/curved turning geometry is not the main
      descriptor for stairs.
    - SixMinutesWalkingTest, UTurn and TUG can contain straight and curved
      walking portions, depending on INDIP turn annotations.
    """

    print("\n----- INDIVI PATH-TYPE CONSISTENCY CHECK -----")

    static_only_tasks = ["Test2", "StandingBalance"]
    straight_only_tasks = ["Stairs"]
    straight_curved_tasks = ["SixMinutesWalkingTest", "UTurn", "TUG"]

    for trial_data in dataset:

        subject = trial_data["Subject"]
        task = trial_data["Task"]
        trial = trial_data["Trial"]

        sample_path_types = trial_data["SamplePathTypes"]

        unique_paths = unique_path_types_safe(sample_path_types)

        numeric_paths = {
            path
            for path in unique_paths
            if path is not None
        }

        path_names = [
            PATH_TYPE_MAP.get(path, "unknown")
            for path in unique_paths
        ]

        if task in static_only_tasks:

            # Static-only trials should not contain numeric path types.
            # They can either store SamplePathTypes as None, or as an array
            # containing only None values.
            if len(numeric_paths) == 0:
                print(
                    f"[OK] Subject {subject} | {task} | {trial}: "
                    f"path type is not applicable, found {path_names}"
                )
            else:
                print(
                    f"[ERROR] Subject {subject} | {task} | {trial}: "
                    f"static task should contain only None path types, found {path_names}"
                )

        elif task in straight_only_tasks:

            # Stairs may contain None for non-walking/static samples and 0 for
            # stair-walking samples. Curved path type is not expected here.
            if len(numeric_paths) == 0:
                print(
                    f"[ERROR] Subject {subject} | {task} | {trial}: "
                    f"no numeric path type found, found {path_names}"
                )

            elif numeric_paths.issubset({0}):
                print(
                    f"[OK] Subject {subject} | {task} | {trial}: "
                    f"stairs path types = {path_names}"
                )

            else:
                print(
                    f"[ERROR] Subject {subject} | {task} | {trial}: "
                    f"stairs should contain only straight path type, found {path_names}"
                )

        elif task in straight_curved_tasks:

            # Walking tasks may contain:
            # - None for static/non-walking samples;
            # - 0 for straight walking;
            # - 1 for curved walking.
            if len(numeric_paths) == 0:
                print(
                    f"[ERROR] Subject {subject} | {task} | {trial}: "
                    f"no numeric path type found, found {path_names}"
                )

            elif numeric_paths.issubset({0, 1}):
                print(
                    f"[OK] Subject {subject} | {task} | {trial}: "
                    f"path types = {path_names}"
                )

            else:
                print(
                    f"[ERROR] Subject {subject} | {task} | {trial}: "
                    f"unexpected path-type values found: {path_names}"
                )

        else:
            print(
                f"[WARNING] Subject {subject} | {task} | {trial}: "
                f"task not explicitly handled in path-type check, found {path_names}"
            )


# -----------------------------------------------------------------------------
# Function role: expected-label consistency check
# -----------------------------------------------------------------------------
# This function verifies that each task contains only the expected binary labels.
# For example, static tasks should contain only static labels, while walking
# tasks should contain static and walking samples. This is useful for detecting
# incorrect annotation parsing or task-specific labeling mistakes.

def check_indivi_expected_labels(dataset):
    """
    Check that each INDIVI task contains only the expected activity labels.

    Expected labels:
    - Test2, StandingBalance: static
    - SixMinutesWalkingTest, UTurn, TUG: static + walking
    - Stairs: static + walking

    The input dataset is a flat list of trial dictionaries.
    """

    expected_by_task = {
        "Test2": {LABEL_MAP["static"]},
        "StandingBalance": {LABEL_MAP["static"]},
        "SixMinutesWalkingTest": {LABEL_MAP["static"], LABEL_MAP["walking"]},
        "UTurn": {LABEL_MAP["static"], LABEL_MAP["walking"]},
        "TUG": {LABEL_MAP["static"], LABEL_MAP["walking"]},
        # Stairs are merged into generic walking for GSD.
        # Therefore only static and walking labels are expected.
        "Stairs": {
            LABEL_MAP["static"],
            LABEL_MAP["walking"]
        },
    }

    print("\n----- INDIVI LABEL CONSISTENCY CHECK -----")

    for trial_data in dataset:

        subject = trial_data["Subject"]
        task = trial_data["Task"]
        trial = trial_data["Trial"]

        expected_labels = expected_by_task.get(task, None)

        if expected_labels is None:
            print(f"[WARNING] Subject {subject} | Task {task}: task not expected.")
            continue

        unique_labels = set(
            np.unique(trial_data["SampleLabels"]).astype(int).tolist()
        )

        unexpected_labels = unique_labels - expected_labels

        label_names = [
            LABEL_NAME_MAP.get(lbl, "unknown")
            for lbl in sorted(unique_labels)
        ]

        if len(unexpected_labels) == 0:
            print(f"[OK] Subject {subject} | {task} | {trial}: {label_names}")
        else:
            unexpected_names = [
                LABEL_NAME_MAP.get(lbl, "unknown")
                for lbl in sorted(unexpected_labels)
            ]

            print(
                f"[ERROR] Subject {subject} | {task} | {trial}: "
                f"unexpected labels {unexpected_names}. "
                f"All labels: {label_names}"
            )

# =====================================================
# MAIN PROGRAM
# =====================================================


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
# The main block runs the complete INDIVI construction pipeline.
# The processing order is:
# 1. resolve the raw-data path from the project root;
# 2. select subjects and tasks;
# 3. build the trial-level dataset;
# 4. summarize the trial-level output;
# 5. apply gravity alignment;
# 6. run label/path consistency checks;
# 7. save the trial-level dataset;
# 8. build and save 1s, 2s and 10s window datasets;
# 9. print final diagnostic information.
# =============================================================================

if __name__ == "__main__":

    # INDIVI raw-data path
    # The dataset path is resolved automatically from the project root.
    # This avoids user-specific absolute Windows paths.

    root_path = (
            PROJECT_ROOT
            / "data"
            / "INDIVI"
            / "ALL data"
    )

    selected_subjects = [
        f"{i:03d}"
        for i in range(1, 22)
    ]

    selected_tasks = [
        "Test2",
        "SixMinutesWalkingTest",
        "StandingBalance",
        "UTurn",
        "TUG",
        "Stairs",
    ]

    dataset = build_dataset(
        root_path=root_path,
        selected_subjects=selected_subjects,
        selected_tasks=selected_tasks
    )

    summarize_dataset(dataset)

    dataset = apply_gravity_alignment_to_dataset(
        dataset=dataset,
        static_duration_s=1.0
    )

    check_indivi_expected_labels(dataset)
    check_indivi_path_types(dataset)

    # Results directory
    # All generated datasets are stored inside the common project
    # results folder.
    LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"

    results_dir = LOCAL_RESULTS_ROOT / "INDIVI_dataset"
    results_dir.mkdir(parents=True, exist_ok=True)

    trial_save_path = results_dir / "indivi_trial_dataset.pkl"
    save_dataset(dataset, trial_save_path)

    WINDOW_CONFIGS = {
        "1s_50p_overlap": {"window_size": 100, "step_size": 50},
        "2s_50p_overlap": {"window_size": 200, "step_size": 100},
        "5s_50p_overlap": {"window_size": 500, "step_size": 250},
        "10s_50p_overlap": {"window_size": 1000, "step_size": 500},
    }

    for name, cfg in WINDOW_CONFIGS.items():

        print(f"\n===== BUILDING WINDOW DATASET: {name} =====")

        window_dataset = build_window_dataset(
            dataset=dataset,
            window_size=cfg["window_size"],
            step_size=cfg["step_size"]
        )

        X, y, metadata = convert_windows_to_arrays(window_dataset)

        # Save metadata CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_save_path = results_dir / f"INDIVI_metadata_{name}.csv"
        # ------------------------------------------------------------
        # Semantic consistency checks for window-level metadata
        # ------------------------------------------------------------
        # These checks verify that the final window-level metadata remain
        # coherent with the final binary GSD label.
        #
        # A walking window must not have a final static_type different from none,
        # because static_type describes the final static condition of the window.
        #
        # A static window must not have a final path_type different from none,
        # because path_type describes locomotor geometry and is meaningful only
        # for final walking windows.
        #
        # Static-context metadata are allowed in both cases because they describe
        # the internal composition of the window, not the final class.
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
        print(
            metadata_df
            .groupby(["label_name", "path_type_name"])
            .size()
            .reset_index(name="n_windows")
        )

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        unique_labels, counts = np.unique(y, return_counts=True)

        print("\nWindow-label distribution:")
        for lbl, cnt in zip(unique_labels, counts):
            label_name = LABEL_NAME_MAP.get(int(lbl), "unknown")
            print(f"  Label {lbl} ({label_name}): {cnt} windows")

        valid_path_types = [
            m["path_type"]
            for m in metadata
            if m["path_type"] is not None
        ]

        print("\nWindow path-type distribution:")

        if len(valid_path_types) > 0:
            unique_paths, path_counts = np.unique(
                valid_path_types,
                return_counts=True
            )

            for path, cnt in zip(unique_paths, path_counts):
                path_name = PATH_TYPE_MAP.get(int(path), "unknown")
                print(f"  PathType {path} ({path_name}): {cnt} windows")
        else:
            print("  No valid path-type labels available.")

        robust_values = [
            m["is_robust_window"]
            for m in metadata
            if m["is_robust_window"] is not None
        ]

        print("\nRobust-window distribution:")

        if len(robust_values) > 0:
            unique_robust, robust_counts = np.unique(
                robust_values,
                return_counts=True
            )

            for robust_value, cnt in zip(unique_robust, robust_counts):
                print(f"  is_robust_window={robust_value}: {cnt} windows")
        else:
            print("  Robust-window metadata not available.")

        window_bundle = {
            "DatasetName": "INDIVI",
            "WindowConfig": name,
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

        window_save_path = results_dir / f"INDIVI_window_dataset_{name}.pkl"
        save_dataset(window_bundle, window_save_path)

    subjects = sorted({
        trial_data["Subject"]
        for trial_data in dataset
    })

    print("\nDataset subjects:")
    print(subjects)

    print("\nFirst 5 trial summaries:")

    for trial_data in dataset[:5]:
        print(f"\nSubject {trial_data['Subject']}")
        print(f"  Task: {trial_data['Task']}")
        print(f"  TestType: {trial_data.get('TestType', 'unknown')}")
        print(f"  Trial: {trial_data['Trial']}")
        print(f"  Acc shape: {trial_data['Acc'].shape}")
        print(f"  Gyr shape: {trial_data['Gyr'].shape}")
        print(f"  Mag shape: {trial_data['Mag'].shape}")
        print(f"  Time shape: {trial_data['Time'].shape}")
        print(f"  Fs: {trial_data['Fs']}")
        print(f"  SampleLabels shape: {trial_data['SampleLabels'].shape}")
        print(f"  Unique labels: {np.unique(trial_data['SampleLabels'])}")
        print(f"  Standard X shape: {trial_data['X'].shape}")
        print(f"  labels shape: {trial_data['labels'].shape}")
        print(f"  labels_har shape: {trial_data['labels_har'].shape}")
        print(f"  test_type: {trial_data['test_type']}")