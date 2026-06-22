"""
Cluster-based subject split for GSD feature CSV files.

Goal:
- Avoid purely random subject split.
- Represent subjects using class-specific mean feature profiles.
- Compute the clustered subject split only once on a reference window length.
- Reuse the same subject-level train/internal-test assignment for all window lengths.

This script does NOT resample windows.
It only creates train/internal-test CSV files.
CBM resampling, if used, must be applied later only on the training set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =====================================================
# PATH MANAGEMENT
# =====================================================

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

# Results are resolved from the project root.
# This keeps the script portable across different Windows PCs.
LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"


# =====================================================
# DATASET SELECTION
# =====================================================

DATASET_NAME = "WearGaitPD"

DATASET_CONFIGS = {
    "WearGaitPD": {
        "results_folder": "WearGaitPD_dataset",
        "dataset_slug": "weargaitpd",
        "train_subjects_per_group": {
            "PD": 60,
            "CONTROL": 60,
        },
    },
    "INDIVI": {
        "results_folder": "INDIVI_dataset",
        "dataset_slug": "indivi",
        "train_fraction_per_group": 0.8,
    },
    "TOWalk": {
        "results_folder": "TOWalk_dataset",
        "dataset_slug": "towalk",
        "train_fraction_per_group": 0.8,
    },
    "MOVEWISE_InLab": {
        "results_folder": "MOVEWISE_InLab_dataset",
        "dataset_slug": "movewise_inlab",
        "train_fraction_per_group": 0.8,
    },
    "MOVEWISE_OutOfLab": {
        "results_folder": "MOVEWISE_OutOfLab_dataset",
        "dataset_slug": "movewise_outoflab",
        "train_fraction_per_group": 0.8,
    },
}

CONFIG = DATASET_CONFIGS[DATASET_NAME]
DATASET_SLUG = CONFIG["dataset_slug"]


# =====================================================
# PATHS
# =====================================================

RESULTS_DIR = LOCAL_RESULTS_ROOT / CONFIG["results_folder"]
FEATURES_DIR = RESULTS_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_CONFIG_NAMES = [
    "1s_50p_overlap",
    "2s_50p_overlap",
    "5s_50p_overlap",
    "10s_50p_overlap",
]


# =====================================================
# FIXED SUBJECT SPLIT ACROSS WINDOW CONFIGURATIONS
# =====================================================
# When True, the clustered split is computed only once on the reference
# window configuration. The same subject-level train/internal-test assignment
# is then reused for all other window lengths.
#
# This is essential when comparing 1s, 2s, 5s and 10s windows: performance
# differences should depend on window length and features, not on different
# subjects entering the internal validation set.
FIXED_SUBJECT_SPLIT_ACROSS_WINDOW_CONFIGS = True

# Reference configuration used to compute the clustered subject split.
# 2s is a reasonable default because it is a central configuration and usually
# provides enough windows per subject for stable subject-level profiles.
REFERENCE_WINDOW_CONFIG_NAME = "2s_50p_overlap"


def get_split_paths(window_config_name: str):
    """
    Build input and output paths for one window configuration.
    """

    input_csv = FEATURES_DIR / f"{DATASET_SLUG}_features_{window_config_name}.csv"

    output_train_csv = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_features_{window_config_name}_clustered_train.csv"
    )

    output_test_csv = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_features_{window_config_name}_clustered_internal_test.csv"
    )

    output_subject_split_csv = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_subject_clustered_split_{window_config_name}.csv"
    )

    return input_csv, output_train_csv, output_test_csv, output_subject_split_csv


# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_STATE = 42
N_CLUSTERS_PER_CLASS = 4

CLASS_NAMES = [
    "static",
    "walking",
]

METADATA_COLS = [
    "label",
    "label_name",
    "subject_id",
    "task",
    "test_type",
    "file_name",
    "file_path",
    "challenge_type",
    "relative_file_path",
    "original_window_label",
    "is_transition",
    "walking_bout",
    "trial",
    "trial_index",
    "n_samples",
    "duration_s",
    "window_start_sample",
    "window_end_sample",
    "window_start_time",
    "window_end_time",
    "window_duration_s",
    "n_static_samples",
    "n_walking_samples",
    "n_none_label_samples",
    "static_fraction",
    "walking_fraction",
    "none_label_fraction",
    "valid_label_fraction",
    "final_label_fraction",
    "label_min_fraction",
    "walking_static_ratio",
    "is_robust_window",

    "path_type",
    "path_type_name",
    "path_type_min_fraction",
    "n_straight_path_samples",
    "n_curved_path_samples",
    "n_none_path_samples",
    "n_other_path_samples",
    "straight_path_fraction",
    "curved_path_fraction",
    "none_path_fraction",
    "path_type_confidence",
    "is_path_type_robust",

    "static_type",
    "static_type_name",
    "static_context_min_fraction",
    "n_none_static_type_samples",
    "n_generic_static_type_samples",
    "n_standing_static_type_samples",
    "n_sitting_static_type_samples",
    "none_static_type_fraction",
    "generic_static_type_fraction",
    "standing_static_type_fraction",
    "sitting_static_type_fraction",
    "static_context_type",
    "static_context_type_name",
    "static_context_fraction",
    "has_static_context",
    "has_standing_context",
    "has_sitting_context",

    "transition_type",
    "gait_phase",
    "gait_phase_name",
    "group",

    "window_index",
    "dataset",
    "activity_detail",
    "activity_detail_name",
]


def get_n_train_for_group(group_subjects: pd.DataFrame, group_name: str) -> int:
    """
    Decide how many subjects should be assigned to training for one group.

    WearGaitPD uses fixed numbers because PD and CONTROL are large groups.
    Other datasets use a fraction-based split, which is more robust when the
    number of subjects is different.
    """

    if "train_subjects_per_group" in CONFIG:
        if group_name in CONFIG["train_subjects_per_group"]:
            return CONFIG["train_subjects_per_group"][group_name]

    train_fraction = CONFIG.get("train_fraction_per_group", 0.8)
    n_subjects = len(group_subjects)

    n_train = int(round(n_subjects * train_fraction))

    # Keep at least one subject in the test set whenever possible.
    n_train = min(n_train, n_subjects - 1)

    # Keep at least one subject in the training set.
    n_train = max(n_train, 1)

    return n_train


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Select numerical feature columns only.

    Metadata columns are excluded because they describe labels, subjects,
    tasks or acquisition information and must not influence clustering.
    """

    candidate_cols = [
        col for col in df.columns
        if col not in METADATA_COLS
    ]

    feature_cols = [
        col for col in candidate_cols
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_cols


def convert_har_labels_to_gsd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the feature table from HAR labels to binary GSD labels.

    Original labels:
    - 0 = static
    - 1 = walking
    - 2 = ascending
    - 3 = descending

    GSD labels:
    - 0 = static
    - 1 = walking

    The original HAR condition is preserved in activity_detail/activity_detail_name
    when these columns are not already present.
    """

    df = df.copy()

    if "activity_detail" not in df.columns:
        df["activity_detail"] = df["label"]

    if "activity_detail_name" not in df.columns:
        df["activity_detail_name"] = df["label_name"]

    df["label"] = df["label"].replace({
        2: 1,
        3: 1,
    })

    df["label_name"] = df["label"].map({
        0: "static",
        1: "walking",
    })

    return df


def compute_subject_class_profiles(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Compute one mean feature vector for each subject and each class.

    The output table has one row per subject. For each class, feature means are
    stored with a class-specific prefix.
    """

    subjects = (
        df[["subject_id", "group"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    subject_profiles = subjects.copy()

    for class_name in CLASS_NAMES:
        df_class = df[df["label_name"] == class_name].copy()

        class_means = (
            df_class
            .groupby("subject_id")[feature_cols]
            .mean()
            .reset_index()
        )

        renamed_cols = {
            col: f"{class_name}__{col}"
            for col in feature_cols
        }

        class_means = class_means.rename(columns=renamed_cols)

        subject_profiles = subject_profiles.merge(
            class_means,
            on="subject_id",
            how="left"
        )

        subject_profiles[f"has_{class_name}"] = (
            subject_profiles[f"{class_name}__{feature_cols[0]}"]
            .notna()
            .astype(int)
        )

    return subject_profiles


def cluster_subjects_per_class(
    subject_profiles: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Cluster subjects separately for each class.

    For each activity class:
    - use the subject-level mean feature vector for that class;
    - fill missing class-specific profiles with the population mean;
    - standardize features;
    - run KMeans;
    - save the resulting cluster ID as a subject-level descriptor.
    """

    clustered_subjects = subject_profiles[["subject_id", "group"]].copy()

    for class_name in CLASS_NAMES:
        class_feature_cols = [
            f"{class_name}__{col}"
            for col in feature_cols
        ]

        X_class = subject_profiles[class_feature_cols].copy()

        # Missing values occur when a subject has no windows for a given class.
        # They are replaced with the population average profile of that class.
        X_class = X_class.fillna(X_class.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_class)

        n_available_subjects = len(subject_profiles)
        n_clusters = min(N_CLUSTERS_PER_CLASS, n_available_subjects)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=20
        )

        cluster_labels = kmeans.fit_predict(X_scaled)

        clustered_subjects[f"cluster_{class_name}"] = cluster_labels
        clustered_subjects[f"has_{class_name}"] = subject_profiles[f"has_{class_name}"]

    return clustered_subjects


def build_combined_cluster_profile(clustered_subjects: pd.DataFrame) -> pd.DataFrame:
    """
    Combine class-specific clusters into one global subject profile.

    Current task:
    - binary GSD;
    - static vs walking.
    """

    clustered_subjects = clustered_subjects.copy()

    clustered_subjects["cluster_profile"] = (
        "S" + clustered_subjects["cluster_static"].astype(str) +
        "_W" + clustered_subjects["cluster_walking"].astype(str)
    )

    clustered_subjects["strata"] = (
        clustered_subjects["group"].astype(str) +
        "__" +
        clustered_subjects["cluster_profile"].astype(str)
    )

    return clustered_subjects


def merge_rare_strata(subject_table: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """
    Merge rare strata to avoid split errors.

    Stratified splitting requires at least two subjects per stratum. Very rare
    combined cluster profiles are merged into a group-specific OTHER stratum.
    """

    subject_table = subject_table.copy()

    counts = subject_table["strata"].value_counts()
    rare_strata = counts[counts < min_count].index.tolist()

    subject_table["strata_merged"] = subject_table["strata"]

    rare_mask = subject_table["strata"].isin(rare_strata)

    subject_table.loc[rare_mask, "strata_merged"] = (
        subject_table.loc[rare_mask, "group"].astype(str) + "__OTHER"
    )

    return subject_table


def clustered_group_split(
    subject_table: pd.DataFrame,
    group_name: str,
    n_train: int
):
    """
    Select training subjects within one diagnostic group.

    The split is stratified using the merged cluster profiles. This keeps the
    train/internal-test split representative of the subject clusters.
    """

    group_subjects = subject_table[
        subject_table["group"].astype(str) == str(group_name)
    ].copy()

    if len(group_subjects) <= n_train:
        raise ValueError(
            f"Requested {n_train} training subjects for group {group_name}, "
            f"but only {len(group_subjects)} are available."
        )

    strata_counts = group_subjects["strata_merged"].value_counts()
    use_stratification = (strata_counts.min() >= 2)

    if use_stratification:
        print(f"[INFO] Using stratified split for group {group_name}")
        stratify_labels = group_subjects["strata_merged"]
    else:
        print(
            f"[WARNING] Stratification disabled for group {group_name} "
            f"because at least one stratum has fewer than 2 subjects."
        )
        stratify_labels = None

    train_subjects, test_subjects = train_test_split(
        group_subjects,
        train_size=n_train,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_labels
    )

    return train_subjects, test_subjects


# =====================================================
# FIXED SPLIT UTILITIES
# =====================================================

def create_clustered_subject_split_table(
    df: pd.DataFrame,
    window_config_name: str
) -> pd.DataFrame:
    """
    Create the clustered subject-level split table for the reference window.

    This function contains the actual clustering logic. It is intentionally
    separated from CSV saving so that the resulting subject split can be reused
    across all window configurations.
    """

    feature_cols = get_feature_columns(df)

    print("\n[INFO] Number of feature columns used for clustering:", len(feature_cols))

    subject_profiles = compute_subject_class_profiles(
        df=df,
        feature_cols=feature_cols
    )

    clustered_subjects = cluster_subjects_per_class(
        subject_profiles=subject_profiles,
        feature_cols=feature_cols
    )

    clustered_subjects = build_combined_cluster_profile(clustered_subjects)

    clustered_subjects = merge_rare_strata(
        subject_table=clustered_subjects,
        min_count=2
    )

    print("\n[DEBUG] Overall strata distribution:")
    print(clustered_subjects["strata_merged"].value_counts())

    print("\n[DEBUG] Strata distribution by group:")
    print(
        clustered_subjects
        .groupby(["group", "strata_merged"])
        .size()
        .sort_values(ascending=False)
    )

    print("\n[DEBUG] Rare strata by group:")
    rare_debug = (
        clustered_subjects
        .groupby(["group", "strata_merged"])
        .size()
        .reset_index(name="n_subjects")
    )
    print(rare_debug[rare_debug["n_subjects"] < 2])

    train_subject_list = []
    test_subject_list = []

    available_groups = sorted(
        clustered_subjects["group"]
        .dropna()
        .astype(str)
        .unique()
    )

    for group_name in available_groups:
        group_subjects = clustered_subjects[
            clustered_subjects["group"].astype(str) == group_name
        ].copy()

        n_train = get_n_train_for_group(
            group_subjects=group_subjects,
            group_name=group_name
        )

        current_train, current_test = clustered_group_split(
            subject_table=clustered_subjects,
            group_name=group_name,
            n_train=n_train
        )

        train_subject_list.append(current_train)
        test_subject_list.append(current_test)

    train_subjects = pd.concat(train_subject_list, axis=0)
    test_subjects = pd.concat(test_subject_list, axis=0)

    train_ids = set(train_subjects["subject_id"])
    test_ids = set(test_subjects["subject_id"])

    clustered_subjects["split"] = "unused"
    clustered_subjects.loc[
        clustered_subjects["subject_id"].isin(train_ids),
        "split"
    ] = "train"
    clustered_subjects.loc[
        clustered_subjects["subject_id"].isin(test_ids),
        "split"
    ] = "test"

    clustered_subjects["split_reference_window_config"] = window_config_name
    clustered_subjects["split_application_mode"] = "reference_clustered_split"

    return clustered_subjects


def apply_fixed_subject_split(
    df: pd.DataFrame,
    fixed_subject_split_table: pd.DataFrame,
    window_config_name: str
) -> pd.DataFrame:
    """
    Apply a precomputed subject-level split to the current feature table.

    This is the core fix for cross-window consistency. It prevents the same
    subject from being assigned to train for one window length and to internal
    validation for another one.
    """

    required_columns = ["subject_id", "group", "split"]

    missing_columns = [
        col for col in required_columns
        if col not in fixed_subject_split_table.columns
    ]

    if len(missing_columns) > 0:
        raise ValueError(
            "The fixed subject split table is missing required columns: "
            f"{missing_columns}"
        )

    current_subjects = (
        df[["subject_id", "group"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    fixed_subjects = fixed_subject_split_table.copy()

    duplicated_fixed_subjects = fixed_subjects[
        fixed_subjects.duplicated(subset=["subject_id"], keep=False)
    ]

    if len(duplicated_fixed_subjects) > 0:
        raise ValueError(
            "Duplicated subject IDs found in the fixed split table. "
            "Each subject must have exactly one split assignment."
        )

    # Merge on subject_id only, then explicitly check whether the group is
    # consistent. This makes group mismatches visible.
    applied_subjects = current_subjects.merge(
        fixed_subjects,
        on="subject_id",
        how="left",
        suffixes=("", "_reference")
    )

    missing_split_mask = applied_subjects["split"].isna()

    if missing_split_mask.any():
        missing_subjects = applied_subjects.loc[
            missing_split_mask,
            ["subject_id", "group"]
        ].copy()

        debug_missing_path = (
            FEATURES_DIR /
            f"{DATASET_SLUG}_missing_fixed_split_subjects_{window_config_name}.csv"
        )

        missing_subjects.to_csv(debug_missing_path, index=False)

        raise RuntimeError(
            "Some subjects in the current window configuration are not present "
            "in the fixed reference split. A debug CSV was saved to: "
            f"{debug_missing_path}"
        )

    if "group_reference" in applied_subjects.columns:
        group_mismatch_mask = (
            applied_subjects["group"].astype(str) !=
            applied_subjects["group_reference"].astype(str)
        )

        if group_mismatch_mask.any():
            debug_group_path = (
                FEATURES_DIR /
                f"{DATASET_SLUG}_fixed_split_group_mismatch_{window_config_name}.csv"
            )

            applied_subjects.loc[group_mismatch_mask].to_csv(
                debug_group_path,
                index=False
            )

            raise RuntimeError(
                "Group mismatch detected between the current feature table and "
                "the fixed reference split. A debug CSV was saved to: "
                f"{debug_group_path}"
            )

    valid_split_values = {"train", "test"}
    observed_split_values = set(applied_subjects["split"].dropna().astype(str))
    unexpected_values = observed_split_values.difference(valid_split_values)

    if len(unexpected_values) > 0:
        raise ValueError(
            "Unexpected split values found in the fixed subject split table: "
            f"{sorted(unexpected_values)}"
        )

    # Keep the current group column as the authoritative group for this feature
    # table, while preserving reference clustering columns for debug.
    if "group_reference" in applied_subjects.columns:
        applied_subjects = applied_subjects.drop(columns=["group_reference"])

    applied_subjects["split_reference_window_config"] = (
        fixed_subject_split_table["split_reference_window_config"].iloc[0]
        if "split_reference_window_config" in fixed_subject_split_table.columns
        else REFERENCE_WINDOW_CONFIG_NAME
    )
    applied_subjects["split_application_mode"] = "fixed_reference_split_applied"
    applied_subjects["current_window_config"] = window_config_name

    return applied_subjects


# =====================================================
# REPORTING / SAVING UTILITIES
# =====================================================

def print_split_debug_report(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    subject_split_table: pd.DataFrame
) -> None:
    """
    Print standard diagnostics for one train/internal-test split.
    """

    print("\n[DEBUG] ====================================")
    print("[DEBUG] SUBJECT LEAKAGE CHECK")
    print("[DEBUG] ====================================")

    train_subjects_set = set(df_train["subject_id"].unique())
    test_subjects_set = set(df_test["subject_id"].unique())
    shared_subjects = train_subjects_set.intersection(test_subjects_set)

    print("[DEBUG] Number of train subjects:", len(train_subjects_set))
    print("[DEBUG] Number of test subjects:", len(test_subjects_set))
    print("[DEBUG] Shared subjects between train and test:", len(shared_subjects))

    if len(shared_subjects) > 0:
        print("[WARNING] LEAKAGE DETECTED!")
        print("[WARNING] Shared subjects:")
        print(sorted(shared_subjects))
    else:
        print("[SUCCESS] No subject leakage detected.")

    print("\n[DEBUG] ====================================")
    print("[DEBUG] TASK DISTRIBUTION")
    print("[DEBUG] ====================================")

    print("\n[DEBUG] TRAIN TASK DISTRIBUTION:")
    print(df_train["task"].value_counts(normalize=True).sort_index())

    print("\n[DEBUG] TEST TASK DISTRIBUTION:")
    print(df_test["task"].value_counts(normalize=True).sort_index())

    print("\n[DEBUG] ====================================")
    print("[DEBUG] LABEL DISTRIBUTION")
    print("[DEBUG] ====================================")

    print("\n[DEBUG] TRAIN LABEL DISTRIBUTION:")
    print(df_train["label_name"].value_counts(normalize=True))

    print("\n[DEBUG] TEST LABEL DISTRIBUTION:")
    print(df_test["label_name"].value_counts(normalize=True))

    print("\n[DEBUG] ====================================")
    print("[DEBUG] GROUP DISTRIBUTION")
    print("[DEBUG] ====================================")

    print("\n[DEBUG] TRAIN GROUP DISTRIBUTION:")
    print(df_train["group"].value_counts())

    print("\n[DEBUG] TEST GROUP DISTRIBUTION:")
    print(df_test["group"].value_counts())

    print("\n[DEBUG] ====================================")
    print("[DEBUG] DUPLICATE WINDOW CHECK")
    print("[DEBUG] ====================================")

    duplicate_cols = [
        "subject_id",
        "task",
        "file_name",
        "window_start_sample",
        "window_end_sample",
    ]

    train_duplicates = df_train.duplicated(subset=duplicate_cols).sum()
    test_duplicates = df_test.duplicated(subset=duplicate_cols).sum()

    print("[DEBUG] Train duplicated windows:", train_duplicates)
    print("[DEBUG] Test duplicated windows:", test_duplicates)

    print("\n[DEBUG] ====================================")
    print("[DEBUG] TRAIN/TEST SUBJECT-SPLIT COMPARISON")
    print("[DEBUG] ====================================")

    split_debug = (
        subject_split_table
        .groupby(["split", "group"])
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["group", "split"])
    )

    print(split_debug.to_string(index=False))

    if "strata_merged" in subject_split_table.columns:
        print("\n[DEBUG] ====================================")
        print("[DEBUG] TRAIN/TEST STRATA COMPARISON")
        print("[DEBUG] ====================================")

        split_strata_debug = (
            subject_split_table
            .groupby(["split", "group", "strata_merged"])
            .size()
            .reset_index(name="n_subjects")
            .sort_values(["group", "strata_merged", "split"])
        )

        print(split_strata_debug.to_string(index=False))


def save_split_outputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    subject_split_table: pd.DataFrame,
    output_train_csv: Path,
    output_test_csv: Path,
    output_subject_split_csv: Path,
) -> None:
    """
    Save train/internal-test CSVs and the subject-level split table.
    """

    df_train.to_csv(output_train_csv, index=False)
    df_test.to_csv(output_test_csv, index=False)
    subject_split_table.to_csv(output_subject_split_csv, index=False)

    print("\n===== SUBJECT SPLIT SUMMARY =====")

    print("\nTrain subjects:")
    print(
        subject_split_table[subject_split_table["split"] == "train"]["group"]
        .value_counts()
    )

    print("\nTest subjects:")
    print(
        subject_split_table[subject_split_table["split"] == "test"]["group"]
        .value_counts()
    )

    print("\nTrain label distribution:")
    print(df_train["label_name"].value_counts())

    print("\nTest label distribution:")
    print(df_test["label_name"].value_counts())

    if "strata_merged" in subject_split_table.columns:
        print("\nTrain cluster strata distribution:")
        print(
            subject_split_table[subject_split_table["split"] == "train"]
            ["strata_merged"]
            .value_counts()
        )

        print("\nTest cluster strata distribution:")
        print(
            subject_split_table[subject_split_table["split"] == "test"]
            ["strata_merged"]
            .value_counts()
        )

    print("\n[SUCCESS] Saved train CSV:", output_train_csv)
    print("[SUCCESS] Saved test CSV:", output_test_csv)
    print("[SUCCESS] Saved subject split table:", output_subject_split_csv)


# =====================================================
# MAIN SPLIT FUNCTIONS
# =====================================================

def run_clustered_split_for_window_config(
    window_config_name: str,
    fixed_subject_split_table: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Run subject-level split for one window configuration.

    If fixed_subject_split_table is None, the clustered split is computed from
    the current window configuration. This should happen only for the reference
    window configuration when fixed cross-window splitting is enabled.

    If fixed_subject_split_table is provided, no new clustering/splitting is
    performed: the precomputed subject assignment is applied directly. This is
    what keeps subject membership coherent across 1s, 2s, 5s and 10s.
    """

    input_csv, output_train_csv, output_test_csv, output_subject_split_csv = get_split_paths(
        window_config_name
    )

    print("\n============================================================")
    print(f"SUBJECT SPLIT | {window_config_name}")
    print("============================================================")

    if not input_csv.exists():
        print(f"[WARNING] Feature CSV not found. Skipping: {input_csv}")
        return None

    df = pd.read_csv(input_csv, low_memory=False)

    # Convert the input feature table to binary GSD before subject splitting.
    # This ensures that the saved train/test CSVs are coherent with the current
    # task: static vs walking.
    df = convert_har_labels_to_gsd_dataframe(df)

    print("[INFO] Loaded feature table:", df.shape)

    print("\n[INFO] Window-level label distribution:")
    print(df["label_name"].value_counts())

    print("\n[INFO] Subject distribution:")
    print(df[["subject_id", "group"]].drop_duplicates()["group"].value_counts())

    if fixed_subject_split_table is None:
        print("\n[INFO] Computing clustered subject split from this window configuration.")
        print("[INFO] This configuration is the split reference.")

        subject_split_table = create_clustered_subject_split_table(
            df=df,
            window_config_name=window_config_name
        )

    else:
        print("\n[INFO] Applying fixed reference subject split.")
        print("[INFO] No clustering is recomputed for this window configuration.")

        subject_split_table = apply_fixed_subject_split(
            df=df,
            fixed_subject_split_table=fixed_subject_split_table,
            window_config_name=window_config_name
        )

    train_ids = set(
        subject_split_table.loc[
            subject_split_table["split"] == "train",
            "subject_id"
        ]
    )

    test_ids = set(
        subject_split_table.loc[
            subject_split_table["split"] == "test",
            "subject_id"
        ]
    )

    df_train = df[df["subject_id"].isin(train_ids)].copy()
    df_test = df[df["subject_id"].isin(test_ids)].copy()

    # Safety check: every row in the current feature CSV must belong to exactly
    # one of the two subject-level splits.
    n_unassigned_rows = len(df) - len(df_train) - len(df_test)

    if n_unassigned_rows != 0:
        raise RuntimeError(
            "Some feature rows were not assigned to train or internal test. "
            f"Unassigned rows: {n_unassigned_rows}"
        )

    print_split_debug_report(
        df_train=df_train,
        df_test=df_test,
        subject_split_table=subject_split_table
    )

    save_split_outputs(
        df_train=df_train,
        df_test=df_test,
        subject_split_table=subject_split_table,
        output_train_csv=output_train_csv,
        output_test_csv=output_test_csv,
        output_subject_split_csv=output_subject_split_csv
    )

    return subject_split_table


def check_cross_window_subject_split_consistency() -> None:
    """
    Verify that every subject has the same split across all window configurations.

    The output CSV is a compact audit table. It is useful for confirming that
    subjects such as nls208 do not move from train to internal validation when
    changing the window length.
    """

    split_tables = []

    for window_config_name in WINDOW_CONFIG_NAMES:
        _, _, _, output_subject_split_csv = get_split_paths(window_config_name)

        if not output_subject_split_csv.exists():
            continue

        split_df = pd.read_csv(output_subject_split_csv, low_memory=False)

        split_df = split_df[["subject_id", "group", "split"]].copy()
        split_df = split_df.rename(
            columns={"split": f"split_{window_config_name}"}
        )

        split_tables.append(split_df)

    if len(split_tables) == 0:
        print("[WARNING] No subject split tables found for consistency check.")
        return

    consistency_df = split_tables[0]

    for split_df in split_tables[1:]:
        consistency_df = consistency_df.merge(
            split_df,
            on=["subject_id", "group"],
            how="outer"
        )

    split_cols = [
        col for col in consistency_df.columns
        if col.startswith("split_")
    ]

    consistency_df["n_unique_splits"] = consistency_df[split_cols].nunique(
        axis=1,
        dropna=True
    )

    consistency_df["is_consistent_across_windows"] = (
        consistency_df["n_unique_splits"] <= 1
    )

    output_path = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_subject_split_consistency_across_windows.csv"
    )

    consistency_df.to_csv(output_path, index=False)

    inconsistent_subjects = consistency_df[
        ~consistency_df["is_consistent_across_windows"]
    ].copy()

    print("\n[DEBUG] ====================================")
    print("[DEBUG] CROSS-WINDOW SUBJECT SPLIT CONSISTENCY")
    print("[DEBUG] ====================================")
    print("[DEBUG] Output:", output_path)
    print("[DEBUG] Subjects with inconsistent split across windows:", len(inconsistent_subjects))

    if len(inconsistent_subjects) > 0:
        print(inconsistent_subjects.to_string(index=False))
        raise RuntimeError(
            "Cross-window split inconsistency detected. "
            "Check the consistency CSV before proceeding."
        )

    print("[SUCCESS] All subjects keep the same split across window configurations.")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print("\n[INFO] Features directory:")
    print(FEATURES_DIR)

    if FIXED_SUBJECT_SPLIT_ACROSS_WINDOW_CONFIGS:
        if REFERENCE_WINDOW_CONFIG_NAME not in WINDOW_CONFIG_NAMES:
            raise ValueError(
                "REFERENCE_WINDOW_CONFIG_NAME must be one of WINDOW_CONFIG_NAMES."
            )

        print("\n[INFO] Fixed subject split across window configurations: ENABLED")
        print("[INFO] Reference window configuration:", REFERENCE_WINDOW_CONFIG_NAME)

        reference_subject_split_table = run_clustered_split_for_window_config(
            window_config_name=REFERENCE_WINDOW_CONFIG_NAME,
            fixed_subject_split_table=None
        )

        for window_config_name in WINDOW_CONFIG_NAMES:
            if window_config_name == REFERENCE_WINDOW_CONFIG_NAME:
                continue

            run_clustered_split_for_window_config(
                window_config_name=window_config_name,
                fixed_subject_split_table=reference_subject_split_table
            )

        check_cross_window_subject_split_consistency()

    else:
        print("\n[INFO] Fixed subject split across window configurations: DISABLED")
        print("[INFO] Each window configuration will be split independently.")

        for window_config_name in WINDOW_CONFIG_NAMES:
            run_clustered_split_for_window_config(
                window_config_name=window_config_name,
                fixed_subject_split_table=None
            )
