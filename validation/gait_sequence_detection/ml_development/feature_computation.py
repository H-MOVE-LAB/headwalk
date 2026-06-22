"""
Generic feature computation script for GSD window datasets.

This script is dataset-agnostic:
- it loads a saved trial/window dataset produced by a dataset-specific construction script;
- it computes the same feature set for each dataset;
- it preserves standardized metadata columns;
- it saves feature CSV files in the dataset-specific results folder.

Only DATASET_NAME and DATASET_CONFIGS should be changed when running a new dataset.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.feature_extraction import compute_window_features, LABEL_NAME_MAP


# =====================================================
# DATASET SELECTION
# =====================================================

# DATASET_NAME = "INDIVI"
DATASET_NAME = "WearGaitPD"
# DATASET_NAME = "TOWalk"
# DATASET_NAME = "MOVEWISE_InLab"
# DATASET_NAME = "MOVEWISE_OutOfLab"

# Window configurations to process.
# These names must match the suffixes used by the construction scripts.
WINDOW_CONFIG_NAMES = [
    "1s_50p_overlap",
    "2s_50p_overlap",
    "5s_50p_overlap",
    "10s_50p_overlap",
]

# Trial-level features are usually less useful for GSD because a full trial may
# contain both static and walking periods. They can be enabled only for inspection.
COMPUTE_TRIAL_FEATURES = False

# Trial-level features are usually less useful for GSD because a full trial may
# contain both static and walking periods. They can be enabled only for inspection.
COMPUTE_TRIAL_FEATURES = False

DATASET_CONFIGS = {
    "WearGaitPD": {
        "results_folder": "WearGaitPD_dataset",
        "dataset_slug": "weargaitpd",
        "trial_file": "WearGaitPD_trial_dataset.pkl",
        "window_file_template": "WearGaitPD_window_dataset_{window_config}.pkl",
        "sampling_frequency_hz": 100,
        "channel_names": ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"],
    },

    "INDIVI": {
        "results_folder": "INDIVI_dataset",
        "dataset_slug": "indivi",
        "trial_file": "indivi_trial_dataset.pkl",
        "window_file_template": "INDIVI_window_dataset_{window_config}.pkl",
        "sampling_frequency_hz": 100,
        "channel_names": ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"],
    },

    "TOWalk": {
        "results_folder": "TOWalk_dataset",
        "dataset_slug": "towalk",
        "trial_file": "towalk_trial_dataset.pkl",
        "window_file_template": "TOWalk_window_dataset_{window_config}.pkl",
        "sampling_frequency_hz": 100,
        "channel_names": ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"],
    },

    "MOVEWISE_InLab": {
        "results_folder": "MOVEWISE_InLab_dataset",
        "dataset_slug": "movewise_inlab",
        "trial_file": "movewise_inlab_trial_dataset.pkl",
        "window_file_template": "movewise_inlab_window_dataset_{window_config}.pkl",
        "sampling_frequency_hz": 100,
        "channel_names": ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"],
    },

    "MOVEWISE_OutOfLab": {
        "results_folder": "MOVEWISE_OutOfLab_dataset",
        "dataset_slug": "movewise_outoflab",
        "trial_file": "movewise_outoflab_trial_dataset.pkl",
        "window_file_template": "movewise_outoflab_window_dataset_{window_config}.pkl",
        "sampling_frequency_hz": 100,
        "channel_names": ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"],
    },
}


# =====================================================
# PATH MANAGEMENT
# =====================================================

def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by moving upward from the current script.

    This avoids user-specific paths such as:
    C:/Users/cesar/...
    C:/Utenti/mirko/...
    """

    current = start_path.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "data").exists():
            return parent

    raise FileNotFoundError(
        "Project root not found. Expected folders: src/ and data/."
    )


PROJECT_ROOT = find_project_root(Path(__file__))

# Results are resolved from the project root, exactly as in the construction
# scripts. This avoids hard-coded user-specific paths and keeps the script
# portable across different Windows PCs.
LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"

CONFIG = DATASET_CONFIGS[DATASET_NAME]

RESULTS_DIR = LOCAL_RESULTS_ROOT / CONFIG["results_folder"]
FEATURES_DIR = RESULTS_DIR / "features"

FS = CONFIG["sampling_frequency_hz"]
CHANNEL_NAMES = CONFIG["channel_names"]
DATASET_SLUG = CONFIG["dataset_slug"]


def load_pickle(file_path):
    """
    Load a pickle file from disk.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def compute_trial_features(trial_dataset, output_csv):
    """
    Compute one feature row for each full trial.

    This function is compatible with two possible trial formats:

    1. Generic/WearGait-like format:
       trial["X"] contains the full 6-channel signal.
       trial["labels"] contains sample-wise labels.

    2. INDIVI flat format:
       trial["Acc"] and trial["Gyr"] contain raw/aligned IMU signals.
       trial["SampleLabels"] contains sample-wise labels.

    The output is a trial-level feature table with the same metadata naming
    used by the window-level pipeline.
    """

    rows = []

    # ------------------------------------------------------------
    # Normalize trial container format
    # ------------------------------------------------------------
    # INDIVI now saves a flat list of trial dictionaries.
    # If a legacy nested dictionary is accidentally loaded, this block
    # flattens it into a list to avoid iteration over subject IDs.

    if isinstance(trial_dataset, dict):

        flat_trials = []

        for subject_data in trial_dataset.values():
            for task_data in subject_data.values():
                for trial_data in task_data.values():
                    flat_trials.append(trial_data)

        trial_dataset = flat_trials

    if not isinstance(trial_dataset, list):
        raise TypeError(
            "trial_dataset must be a list of trial dictionaries or a legacy nested dict."
        )

    for trial in tqdm(trial_dataset, desc="Computing trial features"):

        # ------------------------------------------------------------
        # Read full-trial signal
        # ------------------------------------------------------------
        # Case 1: generic/WearGait-like format.
        # Case 2: INDIVI format with Acc and Gyr stored separately.

        if "X" in trial:
            signal = np.asarray(trial["X"])

        elif "Acc" in trial and "Gyr" in trial:
            signal = np.hstack([
                np.asarray(trial["Acc"]),
                np.asarray(trial["Gyr"])
            ])

        else:
            raise KeyError(
                "Trial does not contain either 'X' or both 'Acc' and 'Gyr'."
            )

        # ------------------------------------------------------------
        # Read sample-wise labels
        # ------------------------------------------------------------

        if "labels" in trial:
            labels = np.asarray(trial["labels"])

        elif "SampleLabels" in trial:
            labels = np.asarray(trial["SampleLabels"])

        else:
            raise KeyError(
                "Trial does not contain either 'labels' or 'SampleLabels'."
            )

        if len(signal) != len(labels):
            raise ValueError(
                "Signal and label vectors have different lengths. "
                f"Signal length={len(signal)}, labels length={len(labels)}"
            )

        # ------------------------------------------------------------
        # Majority voting for trial-level label
        # ------------------------------------------------------------

        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_label = int(unique_labels[np.argmax(counts)])

        row = compute_window_features(
            window=signal,
            fs=FS,
            channel_names=CHANNEL_NAMES
        )

        row["label"] = majority_label
        row["label_name"] = LABEL_NAME_MAP.get(majority_label, "unknown")

        # ------------------------------------------------------------
        # Standardized metadata
        # ------------------------------------------------------------

        row["subject_id"] = trial.get("subject_id", trial.get("Subject", None))
        row["task"] = trial.get("task", trial.get("Task", None))
        row["test_type"] = trial.get("test_type", trial.get("TestType", None))
        row["challenge_type"] = trial.get("challenge_type", trial.get("ChallengeType", "none"))
        row["trial"] = trial.get("trial", trial.get("Trial", None))
        row["group"] = trial.get("group", trial.get("Group", None))
        row["dataset"] = trial.get("dataset", trial.get("Dataset", DATASET_NAME))

        row["n_samples"] = int(len(signal))
        row["duration_s"] = float(len(signal) / FS)

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(output_csv, index=False)

    print(f"[SUCCESS] Saved trial features to: {output_csv}")
    print(f"[SUCCESS] Shape: {df.shape}")

    return df

def compute_window_features_from_bundle(bundle, output_csv, window_config_name):
    """
    Compute feature rows from one selected window bundle.

    The same function is used for all window configurations:
    1 s, 2 s, 5 s and 10 s with 50% overlap.
    """

    X = bundle["X"]
    Y = bundle["Y"]
    metadata = bundle["metadata"]

    rows = []

    for i in tqdm(
            range(len(X)),
            desc=f"Computing {window_config_name} window features"
    ):
        meta = metadata[i]

        row = compute_window_features(
            window=X[i],
            fs=FS,
            channel_names=CHANNEL_NAMES
        )

        # ------------------------------------------------------------
        # Label handling
        # ------------------------------------------------------------
        # Y is kept as the numerical label saved in the window bundle.
        # However, the most reliable source for standardized metadata is the
        # metadata dictionary created during dataset construction.
        #
        # The main target for the GSD classifier is binary:
        # 0 = static
        # 1 = walking
        #
        # Activity detail is preserved separately and may contain:
        # 0 = static
        # 1 = walking
        # 2 = ascending
        # 3 = descending

        bundle_label = int(Y[i])

        metadata_label = meta.get("label", bundle_label)

        try:
            metadata_label = int(metadata_label)
        except (TypeError, ValueError):
            metadata_label = bundle_label

        if metadata_label in [1, 2, 3]:
            gsd_label = 1
            gsd_label_name = "walking"
        elif metadata_label == 0:
            gsd_label = 0
            gsd_label_name = "static"
        else:
            gsd_label = -1
            gsd_label_name = "none"

        row["label"] = gsd_label
        row["label_name"] = meta.get("label_name", gsd_label_name)

        row["original_window_label"] = bundle_label

        activity_detail = meta.get("activity_detail", bundle_label)
        row["activity_detail"] = activity_detail

        row["activity_detail_name"] = meta.get(
            "activity_detail_name",
            LABEL_NAME_MAP.get(activity_detail, "unknown")
        )

        row["dataset"] = meta.get("dataset", DATASET_NAME)
        row["subject_id"] = meta.get("subject_id", None)
        row["task"] = meta.get("task", None)
        row["test_type"] = meta.get("test_type", None)
        row["challenge_type"] = meta.get("challenge_type", "none")
        row["file_name"] = meta.get("file_name", None)
        row["file_path"] = meta.get("file_path", None)
        row["relative_file_path"] = meta.get("relative_file_path", None)

        row["window_start_sample"] = meta.get("window_start_sample", None)
        row["window_end_sample"] = meta.get("window_end_sample", None)
        row["window_start_time"] = meta.get("window_start_time", None)
        row["window_end_time"] = meta.get("window_end_time", None)
        row["window_duration_s"] = meta.get("window_duration_s", None)

        # Robust-window metadata computed during dataset construction.
        # These fields are used to distinguish clean training windows from
        # ambiguous transition windows.
        #
        # They do not change the label itself:
        # - label is still assigned by majority voting;
        # - is_robust_window only says whether the label is reliable enough
        #   to be used during training.

        row["n_static_samples"] = meta.get("n_static_samples", None)
        row["n_walking_samples"] = meta.get("n_walking_samples", None)
        row["n_none_label_samples"] = meta.get("n_none_label_samples", None)

        row["static_fraction"] = meta.get("static_fraction", None)
        row["walking_fraction"] = meta.get("walking_fraction", None)
        row["none_label_fraction"] = meta.get("none_label_fraction", None)
        row["valid_label_fraction"] = meta.get("valid_label_fraction", None)
        row["final_label_fraction"] = meta.get("final_label_fraction", None)
        row["label_min_fraction"] = meta.get("label_min_fraction", None)

        row["walking_static_ratio"] = meta.get("walking_static_ratio", None)
        row["is_robust_window"] = meta.get("is_robust_window", None)

        # ------------------------------------------------------------
        # Path-type metadata
        # ------------------------------------------------------------
        # These fields describe the straight/curved composition of the window.
        # They are metadata, not IMU-derived features.
        # Downstream scripts must therefore exclude them from feature columns
        # used for clustering, ranking and model training.

        row["path_type"] = meta.get("path_type", None)
        row["path_type_name"] = meta.get("path_type_name", None)
        row["path_type_min_fraction"] = meta.get("path_type_min_fraction", None)

        row["n_straight_path_samples"] = meta.get("n_straight_path_samples", None)
        row["n_curved_path_samples"] = meta.get("n_curved_path_samples", None)
        row["n_none_path_samples"] = meta.get("n_none_path_samples", None)
        row["n_other_path_samples"] = meta.get("n_other_path_samples", None)

        row["straight_path_fraction"] = meta.get("straight_path_fraction", None)
        row["curved_path_fraction"] = meta.get("curved_path_fraction", None)
        row["none_path_fraction"] = meta.get("none_path_fraction", None)

        row["path_type_confidence"] = meta.get("path_type_confidence", None)
        row["is_path_type_robust"] = meta.get("is_path_type_robust", None)

        # Static-type metadata.
        row["static_type"] = meta.get("static_type", None)
        row["static_type_name"] = meta.get("static_type_name", None)

        row["static_context_min_fraction"] = meta.get("static_context_min_fraction", None)

        row["n_none_static_type_samples"] = meta.get("n_none_static_type_samples", None)
        row["n_generic_static_type_samples"] = meta.get("n_generic_static_type_samples", None)
        row["n_standing_static_type_samples"] = meta.get("n_standing_static_type_samples", None)
        row["n_sitting_static_type_samples"] = meta.get("n_sitting_static_type_samples", None)

        row["none_static_type_fraction"] = meta.get("none_static_type_fraction", None)
        row["generic_static_type_fraction"] = meta.get("generic_static_type_fraction", None)
        row["standing_static_type_fraction"] = meta.get("standing_static_type_fraction", None)
        row["sitting_static_type_fraction"] = meta.get("sitting_static_type_fraction", None)

        row["static_context_type"] = meta.get("static_context_type", None)
        row["static_context_type_name"] = meta.get("static_context_type_name", None)
        row["static_context_fraction"] = meta.get("static_context_fraction", None)

        row["has_static_context"] = meta.get("has_static_context", None)
        row["has_standing_context"] = meta.get("has_standing_context", None)
        row["has_sitting_context"] = meta.get("has_sitting_context", None)

        # Transition metadata.
        transition_type = meta.get("transition_type", None)
        row["transition_type"] = transition_type

        if "is_transition" in meta:
            row["is_transition"] = meta.get("is_transition", None)
        else:
            row["is_transition"] = transition_type not in [None, "None", "none", ""]

        row["walking_bout"] = meta.get("walking_bout", -1)

        row["gait_phase"] = meta.get("gait_phase", None)
        row["gait_phase_name"] = meta.get("gait_phase_name", None)

        row["group"] = meta.get("group", None)

        row["window_index"] = i

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"[SUCCESS] Saved {window_config_name} window features to: {output_csv}")


if __name__ == "__main__":

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    trial_path = RESULTS_DIR / CONFIG["trial_file"]

    trial_output_csv = FEATURES_DIR / f"{DATASET_SLUG}_features_trial.csv"

    if COMPUTE_TRIAL_FEATURES:
        print(f"\n===== FEATURE COMPUTATION: {DATASET_NAME} TRIAL DATASET =====")

        trial_dataset = load_pickle(trial_path)

        compute_trial_features(
            trial_dataset=trial_dataset,
            output_csv=trial_output_csv
        )
    else:
        print(f"\n[INFO] Trial-level feature computation skipped for {DATASET_NAME}.")


    '''if COMPUTE_TRIAL_FEATURES:
        print(f"\n===== FEATURE COMPUTATION: {DATASET_NAME} TRIAL DATASET =====")
        trial_dataset = load_pickle(trial_path)
        compute_trial_features(trial_dataset, trial_output_csv)
    else:
        print(f"\n[INFO] Trial-level feature computation skipped for {DATASET_NAME}.")'''

    for window_config_name in WINDOW_CONFIG_NAMES:

        print(
            f"\n===== FEATURE COMPUTATION: "
            f"{DATASET_NAME} {window_config_name} WINDOW DATASET ====="
        )

        window_path = RESULTS_DIR / CONFIG["window_file_template"].format(
            window_config=window_config_name
        )

        window_output_csv = (
                FEATURES_DIR /
                f"{DATASET_SLUG}_features_{window_config_name}.csv"
        )

        if not window_path.exists():
            print(f"[WARNING] Window bundle not found. Skipping: {window_path}")
            continue

        window_bundle = load_pickle(window_path)

        compute_window_features_from_bundle(
            bundle=window_bundle,
            output_csv=window_output_csv,
            window_config_name=window_config_name
        )