"""
This module handles TOWalk dataset construction starting from the raw MATLAB files.

Its goal is to load the original subject-level acquisitions, extract the
relevant signals and metadata, and organize them into a consistent Python
representation ready for the following pipeline blocks.

Typical operations include:
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
from utils.quality_check import check_trial_quality, summarize_quality_report
from utils.rotations import align_imu_to_gravity_with_external_file
from utils.rotations import align_imu_to_gravity
from pathlib import Path

# -----------------------------------------------------------------------------
# Function role: automatic project-root discovery
# -----------------------------------------------------------------------------
# This function avoids hard-coded absolute paths. It starts from the current
# Python file and walks upward until it finds the project folder containing both
# src/ and data/. This is useful when the same repository is used on different
# Windows PCs with different user folders.

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

ROOT_PATH = DATA_ROOT / "TOWALK"

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
# but still contain 20% sitting samples. In that case this helper preserves
# sitting_static_type_fraction and has_sitting_context for error analysis.

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
# Function role: subject-folder discovery
# -----------------------------------------------------------------------------
# This function scans a TOWalk data directory and returns only valid subject
# folders. A valid subject folder must be a 4-digit numeric folder, so the
# function ignores unrelated files or folders automatically.

def get_subject_list(root_path):
    """
    Return the sorted list of valid TOWalk subject folders.
    A valid subject folder has a 4-digit numeric name, such as 0001.
    """
    # TOWalk Real-World data are organized in one folder per subject.
    # Each valid folder contains a data.mat file.
    subjects = sorted([
        folder for folder in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, folder))
        and folder.isdigit()
        and len(folder) == 4
    ])

    return subjects


# -----------------------------------------------------------------------------
# Function role: MATLAB file loading for one subject
# -----------------------------------------------------------------------------
# This function centralizes the loading of one subject data.mat file. Keeping
# this operation isolated makes the rest of the code independent from scipy
# loading options and from the exact local path of the dataset.

def load_subject_mat(root_path, subject):
    """
    Load the TOWalk data.mat file for one subject.

    The expected structure is:
    root_path / subject / data.mat
    """
    # TOWalk stores each subject in a dedicated folder.
    # The MATLAB data structure is loaded from that folder.
    file_path = os.path.join(root_path, subject, "data.mat")

    mat = scipy.io.loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False
    )

    return mat["data"]

# -----------------------------------------------------------------------------
# Function role: conversion from MATLAB time to relative seconds
# -----------------------------------------------------------------------------
# MATLAB datenum values are absolute timestamps expressed in days. For signal
# processing, annotation matching and windowing, the code needs time in seconds
# starting from zero. This function performs that conversion consistently.

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
# Function role: head-worn IMU signal extraction
# -----------------------------------------------------------------------------
# This function extracts the sensor signals used as model input from one TOWalk
# recording. The selected sensor is SU_INDIP.Head, which matches the head-worn
# IMU goal of the project. The output is a standardized trial dictionary trial_data.

def extract_head_signals(trial_struct, subject, recording_name, task_name, test_type):
    """
    Extract Head IMU signals from one TOWalk recording.

    The Head sensor is the head-worn IMU used as model input.
    """

    # Data structure:
    # data
    # └── TimeMeasure1
    #     └── Recording4
    #         ├── StartDateTime
    #         ├── TimeZone
    #         ├── SU_INDIP
    #         │   ├── LowerBack
    #         │   ├── LeftFoot
    #         │   ├── RightFoot
    #         │   ├── LeftWrist
    #         │   ├── RightWrist
    #         │   └── Head
    #         │       ├── Acc
    #         │       ├── Gyr
    #         │       ├── Mag
    #         │       ├── Timestamp
    #         │       └── Fs
    #         └── Standards
    #             ├── PressureInsoles_raw
    #             ├── DistanceModule_raw
    #             └── INDIP
    #                 ├── ContinuousWalkingPeriod
    #                 │   ├── [0]
    #                 │   │   ├── Start
    #                 │   │   ├── End
    #                 │   │   ├── Incline_Start = []
    #                 │   │   ├── Incline_End = []
    #                 │   │   └── ...
    #                 │   ├── [3]
    #                 │   │   ├── Start = 113.43
    #                 │   │   ├── End = 167.95
    #                 │   │   ├── Incline_Start = [124.59, 135.48, 144.86, 157.36]
    #                 │   │   ├── Incline_End = [132.79, 142.42, 154.17, 167.95]
    #                 │   │   ├── Incline_PositiveElevation = [...]
    #                 │   │   └── Incline_NegativeElevation = [...]
    #                 │   └── [...]
    #                 ├── MicroWB
    #                 └── Fs

    # TOWalk stores the head-worn sensor inside SU_INDIP.Head.
    # Acc and Gyr are the 6 channels used for the current pipeline.
    imu = trial_struct.SU_INDIP.Head

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
        "Dataset": "TOWalk",
        "Subject": subject,
        "Task": task_name,
        "TestType": test_type,
        "Trial": recording_name,
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
# This function reads the INDIP branch of the MATLAB structure and converts
# ContinuousWalkingPeriod information into Python dictionaries. This is needed
# because MATLAB may store one CWP as a single struct or multiple CWPs as an
# array of structs; the rest of the pipeline should always see one format.

def extract_indip_annotations(trial_struct, task_name):
    """
    Extract INDIP reference annotations from one trial.

    This function is responsible only for reference information stored in
    Standards.INDIP. These annotations are later used to define labels
    and task-specific metadata for the dataset.
    """
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
# Function role: technical consistency check of INDIP annotations
# -----------------------------------------------------------------------------
# This function checks whether the extracted annotation intervals are usable by
# the code. It does not validate clinical correctness; it detects technical
# problems such as missing starts/ends, inverted intervals, intervals outside
# signal duration and inconsistent turn/incline arrays.

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
    static_tasks = ["Balance"]

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

            # Empty arrays mean that no inclined interval was annotated
            # for the current walking segment.
            if incline_start_arr.size == 0 or incline_end_arr.size == 0:
                pass
            else:
                if incline_start_arr.size != incline_end_arr.size:
                    report["is_consistent"] = False
                    report["issues"].append(
                        f"Segment {segment_id}: incline_start and incline_end have different lengths."
                    )
                else:
                    # Each inclined interval is checked separately because
                    # TOWalk can store multiple incline bouts inside one CWP.
                    for istart, iend in zip(incline_start_arr, incline_end_arr):

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
    return report

# -----------------------------------------------------------------------------
# Function role: binary sample-level GSD label construction
# -----------------------------------------------------------------------------
# This function creates the main binary target for Gait Sequence Detection.
# All samples start as static. Samples inside INDIP ContinuousWalkingPeriod
# intervals are labelled as walking.
#
# Important:
# ascending and descending stair intervals are NOT kept in SampleLabels.
# They are merged into walking because, for binary GSD, stairs are still
# locomotor activity.
#
# Stair direction is preserved separately in SampleActivityDetail.

def build_sample_labels(trial_data):
    """
    Create binary GSD labels for each sample.

    Labels:
    0 = static
    1 = walking

    All ContinuousWalkingPeriod samples are labelled as walking.
    Incline intervals are not separated at the binary GSD level:
    ascending and descending are preserved later in SampleActivityDetail.
    """

    time = trial_data["Time"]
    annotations = trial_data["Annotations"]

    labels = np.full(
        len(time),
        GSD_LABEL_MAP["static"],
        dtype=np.int32
    )

    if annotations is None:
        return labels

    for seg in annotations["ContinuousWalkingSegments"]:

        start = seg["start"]
        end = seg["end"]

        if start is None or end is None:
            continue

        idx_walking = (time >= start) & (time <= end)

        # For GSD, every CWP sample is considered walking.
        # This includes level walking, stairs ascent and stairs descent.
        labels[idx_walking] = GSD_LABEL_MAP["walking"]

    return labels

# -----------------------------------------------------------------------------
# Function role: detailed activity-detail metadata construction
# -----------------------------------------------------------------------------
# This function preserves locomotor details that are intentionally not kept in
# the binary GSD label.
#
# SampleLabels remains binary:
# 0 = static
# 1 = walking
#
# SampleActivityDetail preserves:
# 0 = static
# 1 = walking
# 2 = ascending
# 3 = descending
#
# This allows the same dataset to be used for binary GSD while still keeping
# information about stairs up/down for later analysis.

def build_sample_activity_detail(trial_data):
    """
    Create HAR-level activity-detail labels.

    Labels:
    0 = static
    1 = walking
    2 = ascending
    3 = descending

    This field preserves stair information without changing binary GSD labels.
    """

    time = trial_data["Time"]
    annotations = trial_data["Annotations"]

    activity_detail = np.full(
        len(time),
        ACTIVITY_DETAIL_MAP["static"],
        dtype=np.int32
    )

    if annotations is None:
        return activity_detail

    for seg in annotations["ContinuousWalkingSegments"]:

        start = seg["start"]
        end = seg["end"]

        if start is None or end is None:
            continue

        idx_walking = (time >= start) & (time <= end)
        activity_detail[idx_walking] = ACTIVITY_DETAIL_MAP["walking"]

        incline_start = np.atleast_1d(seg.get("incline_start", []))
        incline_end = np.atleast_1d(seg.get("incline_end", []))
        pos_elev = np.atleast_1d(seg.get("incline_positive_elevation", []))
        neg_elev = np.atleast_1d(seg.get("incline_negative_elevation", []))

        if incline_start.size == 0 or incline_end.size == 0:
            continue

        for i, (istart, iend) in enumerate(zip(incline_start, incline_end)):

            idx_incline = (time >= istart) & (time <= iend)

            positive_value = pos_elev[i] if i < pos_elev.size else 0
            negative_value = neg_elev[i] if i < neg_elev.size else 0

            if positive_value > abs(negative_value):
                activity_detail[idx_incline] = ACTIVITY_DETAIL_MAP["ascending"]

            elif abs(negative_value) > positive_value:
                activity_detail[idx_incline] = ACTIVITY_DETAIL_MAP["descending"]

    return activity_detail

# -----------------------------------------------------------------------------
# Function role: static-type metadata construction
# -----------------------------------------------------------------------------
# This function assigns a static subtype to each sample.
# The goal is to preserve a more informative description of static samples
# without changing the main GSD label.
# GSD labels remain:
# - 0 = static
# - 1 = walking
#
# StaticType is only metadata:
# - none     -> used for walking samples, where static type is not meaningful
# - standing -> used for controlled in-lab Test2(Balance) static samples
# - static   -> used for Real-World and the others walking tasks containing
#               static samples, where posture is not controlled and
#               cannot be safely interpreted as standing.
#
# In TOWalk Standardized, Test2 is a controlled balance task and static
# samples can be interpreted as standing. For the other standardized walking
# tests, the static samples before/after walking trials are also acquired
# in a controlled in-lab context, so they are marked as standing.
#
# In TOWalk Real-World, static samples are marked as generic static because
# the recording is ecological and does not provide a reliable sitting/standing
# subdivision.

def build_sample_static_types(trial_data):
    """
    Build sample-wise static-type metadata.

    This function assigns a static subtype only to samples labelled as
    static. Walking samples remain set to "none", because a static subtype
    is not meaningful during locomotion.

    The distinction is task-dependent:

    - Test2 / Balance is a controlled standardized standing task.
      Therefore, static samples are marked here as "standing".

    - Standardized walking trials, such as slow walking, self-pace walking and
      hurried pace walking, may contain static samples before, after or
      between walking bouts. However, these samples are not controlled balance
      acquisitions. Therefore, they are marked as generic "static".

    - Real-World recordings may also contain static samples.
      They are generically marked as generic "static".

    Static-type encoding:
    - none     = static subtype not applicable, mainly walking samples
    - static   = generic static sample without controlled posture subtype
    - standing = controlled standing/balance sample
    - sitting  = controlled sitting condition

    """

    labels = trial_data["SampleLabels"]
    test_type = trial_data["TestType"]

    # Initialize every sample as "none".
    # This is the correct default for walking samples, because static subtype
    # is not meaningful during locomotion.
    static_type = np.full(
        len(labels),
        STATIC_TYPE_MAP["none"],
        dtype=np.int32
    )

    # Identify samples that are not labelled as walking.
    # Only these samples can receive a static subtype.
    static_idx = labels == GSD_LABEL_MAP["static"]

    # Test2 is mapped to test_type == "standing" in TOWALK_TEST_CONFIG.
    # In this specific case, static samples correspond to a controlled
    # standing/balance acquisition.
    if test_type == "standing" or trial_data["Task"] == "Balance":
        static_type[static_idx] = STATIC_TYPE_MAP["standing"]

    # In all other TOWalk test types, static samples are treated as
    # generic static samples. This includes:
    # - standardized walking tests: slow, self, fast
    # - Real-World recording: real_world
    else:
        static_type[static_idx] = STATIC_TYPE_MAP["static"]

    return static_type

# -----------------------------------------------------------------------------
# Function role: gait-phase metadata placeholder
# -----------------------------------------------------------------------------
# This function keeps the dataset compatible with other datasets where gait
# initiation, steady state and termination may be explicitly represented. In this
# MOVEWISE OutOfLab version, gait phases are not estimated because INDIP CWPs are
# already trimmed and may not include reliable transition phases.

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
#
# This function has a different purpose: it marks static samples
# immediately before and after each INDIP ContinuousWalkingPeriod boundary.
# These samples are not relabelled. They remain static in SampleLabels.
#
# The output is only metadata used later at window level to populate:
# - WindowIsTransition
# - WindowTransitionType
#
# Possible transition types:
# - Static2Walking
# - Walking2Static
# - None
def build_sample_transition_types(trial_data, margin_s=1.0):
    """
    Mark static samples immediately before and after each CWP boundary.

    Labels are NOT modified.
    TransitionType is only metadata.

    Possible values:
    - Static2Walking
    - Walking2Static
    - None
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
            pre_idx & (labels == GSD_LABEL_MAP["static"])
        ] = "Static2Walking"

        transition_types[
            post_idx & (labels == GSD_LABEL_MAP["static"])
        ] = "Walking2Static"

    return transition_types

# -----------------------------------------------------------------------------
# Function role: sample-to-walking-bout association
# -----------------------------------------------------------------------------
# This function assigns each sample to its ContinuousWalkingPeriod segment.
# Samples outside walking bouts receive -1. This metadata is useful for tracing
# each window back to a specific walking bout.

def build_sample_walking_bouts(trial_data):
    """
    Assign one walking-bout ID to each sample.

    Samples outside continuous walking periods are marked as -1.
    Samples inside a walking bout receive the corresponding segment_id.
    """
    time = trial_data["Time"]
    annotations = trial_data["Annotations"]

    walking_bouts = np.full(len(time), -1, dtype=int)

    if annotations is None:
        return walking_bouts

    segments = annotations["ContinuousWalkingSegments"]

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        segment_id = seg["segment_id"]

        idx = (time >= start) & (time <= end)
        walking_bouts[idx] = segment_id

    return walking_bouts

# -----------------------------------------------------------------------------
# Function role: straight/curved path metadata
# -----------------------------------------------------------------------------
# This function creates a sample-wise path-type vector. Samples are initialized
# as straight and then samples inside INDIP Turn_Start/Turn_End intervals are
# marked as curved. Balance/static recordings return None because path geometry
# is not meaningful there.

def build_sample_path_types(trial_data):
    """
    Create one path-type value for each sample of the trial.

    Path types:
    None = path type not applicable, used for static / non-walking samples
    0    = straight walking
    1    = curved walking

    Balance trials return None because path geometry is not meaningful there.
    For walking trials, only samples labelled as walking receive a path type.
    Static samples before, after or between walking bouts remain None.
    """

    test_type = trial_data["TestType"]

    # Path type is not meaningful for pure balance/static trials.
    if test_type == "standing" or trial_data["Task"] == "Balance":
        return None

    time = trial_data["Time"]
    labels = trial_data["SampleLabels"]
    annotations = trial_data["Annotations"]

    # Initialize every sample as None.
    # This avoids assigning straight path type to static samples.
    path_types = np.full(
        len(time),
        None,
        dtype=object
    )

    # Path geometry is meaningful only during walking.
    walking_idx = labels == GSD_LABEL_MAP["walking"]

    # Walking samples are straight by default.
    # Turn intervals will be overwritten as curved.
    path_types[walking_idx] = 0

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

        turn_start_arr = np.atleast_1d(turn_start)
        turn_end_arr = np.atleast_1d(turn_end)

        if turn_start_arr.size == 0 or turn_end_arr.size == 0:
            continue

        for ts, te in zip(turn_start_arr, turn_end_arr):

            # Only walking samples inside turn intervals are marked as curved.
            # Static samples remain None even if they are temporally close to turns.
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
# Function role: harmonization of trial-level field names
# -----------------------------------------------------------------------------
# This function adds standardized lowercase keys that match the format used in
# the other construction scripts.
#
# Original TOWalk-specific fields are preserved. Additional aliases are created
# so that downstream scripts such as feature computation, ML training and
# dataset inspection can read all datasets through the same field names.

def update_standard_trial_fields(trial_data):
    """
    Add WearGaitPD-like lowercase fields to the TOWalk trial.

    Original dataset-specific fields are preserved.
    Standardized fields are added for compatibility with the generic pipeline.
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
    # TOWalk does not currently encode a specific experimental challenge,
    # so all trials are marked as "none".
    trial_data["challenge_type"] = trial_data.get("ChallengeType", "none")

    trial_data["file_name"] = trial_data.get("FileName", "data.mat")
    trial_data["file_path"] = trial_data.get("FilePath", None)

    # Store a project-relative path for portable metadata.
    # Standardized tests and Real-World recordings are stored in different folders,
    # so the folder branch is inferred from the trial name.
    if trial_data["Trial"] == "Recording4":
        relative_branch = "Real-World"
    else:
        relative_branch = "Standardized"

    trial_data["relative_file_path"] = trial_data.get(
        "RelativeFilePath",
        str(
            Path("data")
            / "TOWALK"
            / relative_branch
            / trial_data["Subject"]
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

    # Walking-bout metadata aligned with TOWalk and MOVEWISE.
    # It is useful for bout-level analyses and for tracing each sample back to the
    # corresponding INDIP ContinuousWalkingPeriod.
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

def build_windows_from_trial(trial_data, window_size=200, step_size=100):
    """
    Build fixed-length windows from one trial.

    The final window label remains binary for GSD:
    - static
    - walking

    In addition to the final label, this function stores detailed window
    composition metadata:
    - static/walking/none sample counts and fractions;
    - robust-window flag based on final-label fraction;
    - robust path-type composition;
    - static-context composition;
    - transition metadata.
    """

    acc = trial_data["Acc"]
    gyr = trial_data["Gyr"]

    sample_labels = trial_data["SampleLabels"]
    sample_activity_detail = trial_data["SampleActivityDetail"]
    sample_path_types = trial_data["SamplePathTypes"]
    sample_walking_bouts = trial_data["SampleWalkingBouts"]
    sample_static_types = trial_data["SampleStaticTypes"]
    sample_gait_phases = trial_data["SampleGaitPhases"]
    sample_transition_types = trial_data["SampleTransitionTypes"]

    time = trial_data["Time"]

    signal_6ch = np.hstack([acc, gyr])

    windows = []

    for start_idx in range(0, len(signal_6ch) - window_size + 1, step_size):

        end_idx = start_idx + window_size

        window_signal = signal_6ch[start_idx:end_idx]
        window_labels = sample_labels[start_idx:end_idx]
        window_activity_detail = sample_activity_detail[start_idx:end_idx]
        window_walking_bouts = sample_walking_bouts[start_idx:end_idx]
        window_static_types = sample_static_types[start_idx:end_idx]
        window_gait_phases = sample_gait_phases[start_idx:end_idx]
        window_transition_types = sample_transition_types[start_idx:end_idx]
        window_time = time[start_idx:end_idx]

        if sample_path_types is not None:
            window_path_types = sample_path_types[start_idx:end_idx]
        else:
            window_path_types = None

        # ------------------------------------------------------------
        # Final binary GSD label by majority voting
        # ------------------------------------------------------------
        # The final label is still assigned by majority voting over the
        # sample-wise binary labels.
        # TOWalk currently has no -1/none GSD labels, but the none counter is
        # still computed for cross-dataset metadata consistency.
        unique_labels, counts = np.unique(
            window_labels,
            return_counts=True
        )

        majority_label = int(
            unique_labels[np.argmax(counts)]
        )

        n_static_samples = int(
            np.sum(window_labels == GSD_LABEL_MAP["static"])
        )

        n_walking_samples = int(
            np.sum(window_labels == GSD_LABEL_MAP["walking"])
        )

        # In TOWalk, labels are expected to be only static/walking.
        # This value is therefore usually zero, but it is kept to align the
        # metadata with WearGaitPD and future datasets that may contain none.
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

        # Backward-compatible diagnostic ratio.
        # It is no longer used to decide robustness.
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
        # Example:
        # label_name = walking, static_type_name = none,
        # standing_static_type_fraction = 0.25.
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
        # The detailed fractions are always saved, even when final path_type
        # remains none.
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
        # The final metadata must remain consistent with the final GSD label.
        # Therefore:
        # - walking windows can have path_type and gait_phase;
        # - walking windows must have static_type = none;
        # - static windows can have static_type;
        # - static windows must have path_type = none and gait_phase = none.
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
            "ChallengeType": trial_data.get("challenge_type", "none"),
            "Trial": trial_data["Trial"],

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
            "WindowLabel": majority_label,

            "SampleActivityDetail": window_activity_detail,
            "WindowActivityDetail": final_activity_detail,

            "NStaticSamples": n_static_samples,
            "NWalkingSamples": n_walking_samples,
            "NNoneLabelSamples": n_none_label_samples,
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
            "NStraightPathSamples": int(path_summary["n_straight_path_samples"]),
            "NCurvedPathSamples": int(path_summary["n_curved_path_samples"]),
            "NNonePathSamples": int(path_summary["n_none_path_samples"]),
            "NOtherPathSamples": int(path_summary["n_other_path_samples"]),
            "StraightPathFraction": float(path_summary["straight_path_fraction"]),
            "CurvedPathFraction": float(path_summary["curved_path_fraction"]),
            "NonePathFraction": float(path_summary["none_path_fraction"]),
            "PathTypeConfidence": float(path_summary["path_type_confidence"]),
            "IsPathTypeRobust": bool(path_summary["is_path_type_robust"]),

            "SampleWalkingBouts": window_walking_bouts,
            "WindowWalkingBout": final_walking_bout,

            "SampleTransitionTypes": window_transition_types,
            "WindowIsTransition": window_is_transition,
            "WindowTransitionType": window_transition_type,

            "WindowStaticType": final_static_type,
            "WindowGaitPhase": final_gait_phase,

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

            "Group": trial_data.get("Group", "UNKNOWN"),
        }

        windows.append(window_dict)

    return windows


# -----------------------------------------------------------------------------
# Function role: window generation for the full dataset
# -----------------------------------------------------------------------------
# This function loops through the trial dataset and applies build_windows_from_trial
# to every trial. The output is a flat list of windows across all subjects, tasks
# and trials.

def build_window_dataset(dataset, window_size=200, step_size=100):
    """
    Build the full window-based dataset from a flat trial-level dataset.

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
# Function role: conversion to model-ready arrays and metadata
# -----------------------------------------------------------------------------
# This function converts window dictionaries into X, y and metadata. X contains
# numerical signals, y contains one label per window, and metadata preserves
# traceability for subject, task, trial, timing, path type and transition status.

def convert_windows_to_arrays(window_dataset):
    """
    Convert the list-based window dataset into arrays ready for model training.

    X has shape (num_windows, window_size, num_channels).
    y contains one binary GSD label for each window.
    Metadata preserves traceability and activity-detail information.
    """

    X = np.stack([w["Signal"] for w in window_dataset], axis=0)
    y = np.array([w["WindowLabel"] for w in window_dataset], dtype=int)

    metadata = []

    for w in window_dataset:

        metadata.append({
            "dataset": w.get("Dataset", "TOWalk"),
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

            "walking_bout": w["WindowWalkingBout"],

            "is_transition": bool(w["WindowIsTransition"]),
            "transition_type": w.get("WindowTransitionType", "None"),
            "transition_definition": w.get(
                "TransitionDefinition",
                TRANSITION_DEFINITION
            ),

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

            "gait_phase": w["WindowGaitPhase"],
            "gait_phase_name": GAIT_PHASE_NAME_MAP.get(
                w["WindowGaitPhase"],
                "none"
            ),

            "group": w.get("Group", "UNKNOWN"),
        })

    return X, y, metadata

# -----------------------------------------------------------------------------
# Dataset-specific recording configuration
# -----------------------------------------------------------------------------
# This dictionary maps the original TOWalk MATLAB field names to standardized
# task and test_type metadata. It determines which recordings are processed and
# how they will appear in downstream metadata.
TOWALK_TEST_CONFIG = {
    "Test2": {
        "task_name": "Balance",
        "test_type": "standing",
        "recording_group": "Standardized"
    },
    "Test4": {
        "task_name": "Slow walking",
        "test_type": "slow",
        "recording_group": "Standardized"
    },
    "Test5": {
        "task_name": "Self-pace walking",
        "test_type": "self",
        "recording_group": "Standardized"
    },
    "Test6": {
        "task_name": "Hurried pace walking",
        "test_type": "fast",
        "recording_group": "Standardized"
    },
    "Recording4": {
        "task_name": "Real-World",
        "test_type": "real_world",
        "recording_group": "Real-World"
    },
}

# -----------------------------------------------------------------------------
# Function role: MATLAB structure inspection utility
# -----------------------------------------------------------------------------
# This helper recursively prints MATLAB struct fields. It is useful for debugging
# unfamiliar .mat structures but it is not part of the actual dataset-building
# pipeline unless called manually.

def print_mat_fields(obj, name, level=0, max_level=2):
    """
    Print MATLAB struct fields to understand the internal organization.
    """
    indent = "  " * level

    if not hasattr(obj, "_fieldnames"):
        print(f"{indent}{name}: no _fieldnames")
        return

    print(f"{indent}{name} fields: {obj._fieldnames}")

    if level >= max_level:
        return

    for field in obj._fieldnames:
        try:
            child = getattr(obj, field)
            print_mat_fields(child, f"{name}.{field}", level + 1, max_level)
        except Exception as e:
            print(f"{indent}  {field}: error reading field ({e})")

# -----------------------------------------------------------------------------
# Function role: subject-level TOWalk dataset construction
# -----------------------------------------------------------------------------
# This function processes one subject for either Standardized or Real-World data.
# It selects the configured recordings, extracts signals and annotations, builds
# labels and metadata, applies gravity alignment and stores valid trials.

def build_subject_dataset(data_struct, subject, recording_group):
    """
    Build the TOWalk trial-level dataset for one subject.

    The dataset includes:
    - standardized tests: Test2, Test4, Test5, Test6
    - Real-World recording: Recording4
    """
    subject_dataset = []

    time_measure = data_struct.TimeMeasure1

    for recording_name, config in TOWALK_TEST_CONFIG.items():

        if recording_group == "Standardized" and config["recording_group"] != "Standardized":
            continue

        if recording_group == "Real-World" and config["recording_group"] != "Real-World":
            continue

        if not hasattr(time_measure, recording_name):
            print(f"[WARNING] Subject {subject}: {recording_name} not found.")
            continue

        trial_struct = getattr(time_measure, recording_name)

        # ------------------------------------------------------------
        # Real-World structure
        # ------------------------------------------------------------
        if recording_name == "Recording4":

            trial_list = [(recording_name, trial_struct)]

        # ------------------------------------------------------------
        # Standardized structure
        # ------------------------------------------------------------
        else:

            trial_list = []

            test_struct = trial_struct

            for field in test_struct._fieldnames:

                if field.startswith("Trial"):
                    current_trial_struct = getattr(test_struct, field)

                    trial_list.append((field, current_trial_struct))

        # ------------------------------------------------------------
        # Loop over extracted trials
        # ------------------------------------------------------------
        for trial_name, trial_struct in trial_list:

            if recording_name == "Recording4":
                standardized_trial_name = recording_name
            else:
                standardized_trial_name = f"{recording_name}_{trial_name}"

            try:
                signal_data = extract_head_signals(
                    trial_struct=trial_struct,
                    subject=subject,
                    recording_name=standardized_trial_name,
                    task_name=config["task_name"],
                    test_type=config["test_type"]
                )

            except Exception as e:
                # If signal extraction fails, the current trial cannot be safely used.
                # The loop continues with the next trial instead of using an undefined
                # or stale signal_data variable.
                print(f"[WARNING] Failed to extract subject {subject} {recording_name}: {e}")
                continue

            annotation_data = extract_indip_annotations(
                trial_struct=trial_struct,
                task_name=config["task_name"]
            )

            trial_data = {
                **signal_data,
                "Annotations": annotation_data,
            }

            annotation_qc = check_annotation_consistency(trial_data)
            trial_data["annotation_check"] = annotation_qc

            trial_data["SampleLabels"] = build_sample_labels(trial_data)

            # SampleActivityDetail preserves the original locomotor subtype.
            # It keeps stairs ascent/descent separated from generic walking,
            # while SampleLabels remains binary for GSD.
            trial_data["SampleActivityDetail"] = build_sample_activity_detail(trial_data)

            trial_data["SampleWalkingBouts"] = build_sample_walking_bouts(trial_data)
            trial_data["SamplePathTypes"] = build_sample_path_types(trial_data)
            trial_data["SampleStaticTypes"] = build_sample_static_types(trial_data)
            trial_data["SampleGaitPhases"] = build_sample_gait_phases(trial_data)

            trial_data["SampleTransitionTypes"] = build_sample_transition_types(
                trial_data,
                margin_s=1.0
            )

            # ------------------------------------------------------------
            # Gravity alignment
            # ------------------------------------------------------------
            acc = trial_data["Acc"]
            gyr = trial_data["Gyr"]
            sampling_rate_hz = trial_data["Fs"]["Acc"]

            static_duration_s = 1.0
            alignment_reference = "first_1s_of_current_recording"

            acc_aligned, gyr_aligned, R = align_imu_to_gravity(
                acc=acc,
                gyr=gyr,
                sampling_rate_hz=sampling_rate_hz,
                static_duration_s=static_duration_s,
                gravity_ideal=np.array([1.0, 0.0, 0.0])
            )

            trial_data["RawAcc"] = acc
            trial_data["RawGyr"] = gyr
            trial_data["Acc"] = acc_aligned
            trial_data["Gyr"] = gyr_aligned

            trial_data = update_standard_trial_fields(trial_data)

            # ------------------------------------------------------------
            # Gravity alignment check
            # ------------------------------------------------------------
            n_static = int(static_duration_s * sampling_rate_hz)

            static_check_aligned = acc_aligned[:n_static]
            mean_static_acc = np.mean(static_check_aligned, axis=0)
            mean_static_acc_norm = mean_static_acc / np.linalg.norm(mean_static_acc)

            print(f"\n[ALIGNMENT CHECK] Subject {subject} | {recording_name}_{trial_name}")
            print("Alignment reference:", alignment_reference)
            print("Static duration [s]:", static_duration_s)
            print("Mean aligned static acceleration:", mean_static_acc)
            print("Normalized mean aligned static acceleration:", mean_static_acc_norm)

            trial_data["RotationMatrix"] = R

            trial_data["GravityAlignment"] = {
                "reference": alignment_reference,
                "static_duration_s": static_duration_s,
                "gravity_ideal": [1.0, 0.0, 0.0],
            }

            qc = check_trial_quality(
                trial_data,
                expected_fs=100,
                min_duration_sec=3.0
            )

            if qc["is_valid"]:
                trial_data["quality_check"] = qc

                subject_dataset.append(trial_data)

            else:
                print(f"[WARNING] {subject} - {recording_name} | {summarize_quality_report(qc)}")

    return subject_dataset

# -----------------------------------------------------------------------------
# Function role: full TOWalk dataset construction
# -----------------------------------------------------------------------------
# This function combines Standardized and Real-World folders. It discovers the
# subjects available in both branches, processes each subject and aggregates
# the resulting subject-level datasets.

def build_dataset(root_path, selected_subjects=None, selected_tasks=None):
    """
    Build the TOWalk dataset from both Standardized and Real-World folders.

    Expected structure:
    root_path / Standardized / subject / data.mat
    root_path / Real-World / subject / data.mat
    """
    dataset = []

    standardized_root = Path(root_path) / "Standardized"
    realworld_root = Path(root_path) / "Real-World"

    subjects_standardized = get_subject_list(standardized_root)
    subjects_realworld = get_subject_list(realworld_root)

    subjects = sorted(set(subjects_standardized + subjects_realworld))

    if selected_subjects is not None:
        subjects = [s for s in subjects if s in selected_subjects]

    for subject in subjects:
        print(f"Processing subject {subject}...")

        subject_dataset = []

        # ------------------------------------------------------------
        # Standardized data
        # ------------------------------------------------------------
        if subject in subjects_standardized:
            try:
                data_struct = load_subject_mat(standardized_root, subject)
                standardized_dataset = build_subject_dataset(
                    data_struct=data_struct,
                    subject=subject,
                    recording_group="Standardized"
                )
                subject_dataset.extend(standardized_dataset)
            except Exception as e:
                print(f"[WARNING] Failed to process standardized subject {subject}: {e}")

        # ------------------------------------------------------------
        # Real-World data
        # ------------------------------------------------------------
        if subject in subjects_realworld:
            try:
                data_struct = load_subject_mat(realworld_root, subject)
                realworld_dataset = build_subject_dataset(
                    data_struct=data_struct,
                    subject=subject,
                    recording_group="Real-World"
                )
                subject_dataset.extend(realworld_dataset)
            except Exception as e:
                print(f"[WARNING] Failed to process Real-World subject {subject}: {e}")

        dataset.extend(subject_dataset)

    return dataset


# -----------------------------------------------------------------------------
# Function role: dataset summary printing
# -----------------------------------------------------------------------------
# This function prints the number of subjects, task groups and valid trials. It
# is a quick sanity check after construction and before saving.
# It handles dataset provided in flat list form.
def summarize_dataset(dataset):
    """
    Print a summary of the flat trial-level dataset.
    """

    subjects = sorted({
        trial_data["Subject"]
        for trial_data in dataset
    })

    tasks = sorted({
        trial_data["Task"]
        for trial_data in dataset
    })

    test_types = sorted({
        trial_data["TestType"]
        for trial_data in dataset
    })

    print("\n----- TOWALK TRIAL DATASET SUMMARY -----")
    print("Number of subjects:", len(subjects))
    print("Number of tasks:", len(tasks))
    print("Number of test types:", len(test_types))
    print("Number of valid trials:", len(dataset))

    print("\nTasks:")
    for task in tasks:
        n_task_trials = sum(
            trial_data["Task"] == task
            for trial_data in dataset
        )
        print(f"  {task}: {n_task_trials} trials")

    print("\nTest types:")
    for test_type in test_types:
        n_test_type_trials = sum(
            trial_data["TestType"] == test_type
            for trial_data in dataset
        )
        print(f"  {test_type}: {n_test_type_trials} trials")

        print("\n----- TASK / TEST_TYPE CHECK -----")

        for trial_data in dataset:
            print(
                trial_data["Subject"],
                "| Task:",
                trial_data["Task"],
                "| TestType:",
                trial_data["TestType"],
                "| Trial:",
                trial_data["Trial"]
            )

# -----------------------------------------------------------------------------
# Function role: pickle serialization
# -----------------------------------------------------------------------------
# This function saves Python objects to disk. It is used both for the trial-level
# dataset and for window-level bundles.

def save_dataset(dataset, save_path):
    """
    Save the constructed dataset as a pickle file.
    """
    # pickle format allows easier access to the dataset.
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to: {save_path}")

PROJECT_ROOT = find_project_root(Path(__file__))
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

ROOT_PATH = DATA_ROOT / "TOWalk"

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
# This block runs only when the script is executed directly. It selects subjects,
# builds the TOWalk dataset, saves the trial-level pickle, creates windowed
# datasets at different durations, exports metadata CSV files and prints debug
# summaries.
# =============================================================================
# main program.
if __name__ == "__main__":
    # f"{i:04d} creates a string using index i with 4 digits format.
    selected_subjects = [f"{i:04d}" for i in range(1, 16)]
    print("DATA_ROOT:", DATA_ROOT)
    print("Folders inside DATA_ROOT:", os.listdir(DATA_ROOT))

    dataset = build_dataset(
        root_path=ROOT_PATH,
        selected_subjects=selected_subjects
    )

    # Print debug dataset summary.
    summarize_dataset(dataset)

    # Define the path where results are expected to be stored.
    results_dir = RESULTS_ROOT / "TOWalk_dataset"

    # If not existing, make directory; otherwise (if exist_ok is
    # already True), skip this step.
    os.makedirs(results_dir, exist_ok=True)

    # Save dataset.
    save_path = results_dir / "towalk_trial_dataset.pkl"
    save_dataset(dataset, save_path)

    WINDOW_CONFIGS = {
        "1s_50p_overlap": {"window_size": 100, "step_size": 50},
        "2s_50p_overlap": {"window_size": 200, "step_size": 100},
        "5s_50p_overlap": {"window_size": 500, "step_size": 250},
        "10s_50p_overlap": {"window_size": 1000, "step_size": 500},
    }

    for name, cfg in WINDOW_CONFIGS.items():
        print(f"\n[INFO] Building {name} window dataset...")

        window_dataset = build_window_dataset(
            dataset,
            window_size=cfg["window_size"],
            step_size=cfg["step_size"]
        )

        if len(window_dataset) == 0:
            print(f"[WARNING] No windows created for {name}. Skipping this window configuration.")
            continue

        X, y, metadata = convert_windows_to_arrays(window_dataset)

        # Save metadata CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_save_path = results_dir / f"TOWalk_metadata_{name}.csv"
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

        window_bundle = {
            "DatasetName": "TOWalk",
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

        save_path = results_dir / f"TOWalk_window_dataset_{name}.pkl"
        save_dataset(window_bundle, save_path)

        print(f"[SUCCESS] {name} dataset saved")
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        unique_labels, counts = np.unique(y, return_counts=True)

        print("\nWindow-label distribution:")
        for lbl, cnt in zip(unique_labels, counts):
            print(f"  Label {lbl}: {cnt} windows")

        transition_counts = pd.Series(
            [m["is_transition"] for m in metadata]
        ).value_counts().sort_index()

        print("\nWindow transition distribution:")
        for transition_flag, cnt in transition_counts.items():
            transition_name = "pure" if transition_flag == 0 else "transition"
            print(f"  {transition_flag} ({transition_name}): {cnt} windows")

        subjects = sorted({
            trial_data["Subject"]
            for trial_data in dataset
        })

        print("\nDataset subjects:")
        print(subjects)

        for trial_data in dataset:
            print(f"\nSubject {trial_data['Subject']}")
            print(f"  Task: {trial_data['Task']}")
            print(f"  TestType: {trial_data['TestType']}")
            print(f"  Trial: {trial_data['Trial']}")
            print(f"  Acc shape: {trial_data['Acc'].shape}")
            print(f"  Gyr shape: {trial_data['Gyr'].shape}")
            print(f"  Mag shape: {trial_data['Mag'].shape}")
            print(f"  Time shape: {trial_data['Time'].shape}")
            print(f"  Fs: {trial_data['Fs']}")
            print(f"  SampleLabels shape: {trial_data['SampleLabels'].shape}")
            print(f"  Unique labels: {np.unique(trial_data['SampleLabels'])}")