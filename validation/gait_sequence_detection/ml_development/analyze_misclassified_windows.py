# =====================================================
# DATASET SELECTION
# =====================================================
"""
Generic misclassification analysis for GSD wrapper predictions.

This script reads window-level predictions produced by bottom_up_wrapper_feature_selection.py
and summarizes errors by metadata.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# DATASET SELECTION
# =====================================================

DATASET_NAME = "WearGaitPD"

# Window configurations produced by the bottom-up wrapper.
# Each configuration has its own output folder and its own best model summary.
WINDOW_CONFIG_NAMES = [
    "1s_50p_overlap",
    "2s_50p_overlap",
    "5s_50p_overlap",
    "10s_50p_overlap",
]

# Models evaluated by the bottom-up wrapper.
# For each model, this script will automatically read the best number of
# selected features from bottomup_wrapper_BEST_by_model.csv.
MODELS_TO_ANALYZE = [
    "knn",
    "rf",
    "svm",
    "lr",
    "gnb",
]

DATASET_CONFIGS = {
    "WearGaitPD": {
        "results_folder": "WearGaitPD_dataset",
        "dataset_slug": "weargaitpd",
    },
    "INDIVI": {
        "results_folder": "INDIVI_dataset",
        "dataset_slug": "indivi",
    },
    "TOWalk": {
        "results_folder": "TOWalk_dataset",
        "dataset_slug": "towalk",
    },
    "MOVEWISE_InLab": {
        "results_folder": "MOVEWISE_InLab_dataset",
        "dataset_slug": "movewise_inlab",
    },
    "MOVEWISE_OutOfLab": {
        "results_folder": "MOVEWISE_OutOfLab_dataset",
        "dataset_slug": "movewise_outoflab",
    },
}


# =====================================================
# PATH MANAGEMENT
# =====================================================

def find_project_root(start_path: Path) -> Path:
    """
    Find the project root without depending on user-specific Windows folders.
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
# This avoids hard-coded user-specific paths and keeps the script portable
# across different Windows PCs.
LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"

CONFIG = DATASET_CONFIGS[DATASET_NAME]

RESULTS_DIR = LOCAL_RESULTS_ROOT / CONFIG["results_folder"]
FEATURES_DIR = RESULTS_DIR / "features"
DATASET_SLUG = CONFIG["dataset_slug"]

# Main bottom-up wrapper folder.
# The expected structure is:
# bottomup_wrapper_feature_selection/<window_config>/<model>/<model>_top<n>_window_predictions.csv
BOTTOMUP_DIR = FEATURES_DIR / "bottomup_wrapper_feature_selection"

# Global folder for the new misclassification analysis.
# Results from all window configurations and models are stored here.
# Global folder for the focused analysis of the best model selected for each
# window configuration. This avoids mixing all classifiers when the goal is to
# understand where the best model for each window length fails.
GLOBAL_OUTPUT_DIR = FEATURES_DIR / "misclassification_analysis_best_models_by_window"
GLOBAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_window_output_dir(window_config_name: str) -> Path:
    """
    Return the bottom-up wrapper output folder for one window configuration.
    """

    return BOTTOMUP_DIR / window_config_name


def get_test_csv_path(window_config_name: str) -> Path:
    """
    Return the internal validation CSV used by the bottom-up wrapper.

    This file is used only for consistency checks because the prediction files
    should already contain the metadata needed for error analysis.
    """

    return (
        FEATURES_DIR /
        f"{DATASET_SLUG}_features_{window_config_name}_clustered_internal_test.csv"
    )


def get_best_by_model_path(window_config_name: str) -> Path:
    """
    Return the table containing the best feature subset for each model.
    """

    return (
        get_window_output_dir(window_config_name) /
        "bottomup_wrapper_BEST_by_model.csv"
    )


def get_prediction_csv_path(
    window_config_name: str,
    model_name: str,
    n_features: int
) -> Path:
    """
    Return the window-level prediction CSV for one model and one feature subset.
    """

    return (
        get_window_output_dir(window_config_name) /
        model_name /
        f"{model_name}_top{n_features}_window_predictions.csv"
    )


def get_model_analysis_output_dir(
    window_config_name: str,
    model_name: str,
    n_features: int
) -> Path:
    """
    Return the output folder for the misclassification analysis of one model.
    """

    output_dir = (
        GLOBAL_OUTPUT_DIR /
        window_config_name /
        model_name /
        f"{model_name}_top{n_features}_misclassification_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

# =====================================================
# CONFIGURATION
# =====================================================

LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}


# Raw metadata columns directly available in the prediction CSV.
# Missing columns are skipped automatically.
METADATA_COLUMNS_TO_ANALYZE = [
    "group",
    "subject_id",
    "task",
    "test_type",
    "challenge_type",
    "activity_detail_name",
    "path_type_name",
    "static_type_name",
    "transition_type",
    "gait_phase_name",
    "is_robust_window",
    "walking_static_ratio",
    "window_duration_s",
    "file_name",
]

# Derived boolean window-type columns created by this script.
# These columns are useful because they convert heterogeneous metadata into
# explicit questions such as:
# - is this window curved?
# - is this window a transition?
# - is this window ascending/descending stairs?
# - is this window standing/static?
DERIVED_FLAG_COLUMNS = [
    "is_curved_window",
    "is_straight_window",
    "is_transition_window",
    "is_robust_window_bool",
    "is_non_robust_window",
    "is_static_true",
    "is_walking_true",
    "is_standing_window",
    "is_sitting_window",
    "is_stairs_window",
    "is_ascending_window",
    "is_descending_window",
    "is_walking_with_path_type_none",
    "is_walking_with_standing_static_type",
    "is_walking_with_static_type_standing_or_sitting",

    # Focused transition-analysis flags.
    "is_static2walking_transition",
    "is_walking2static_transition",
    "has_curved_path_involvement",
    "has_sitting_context",
    "is_tug_context",
    "has_excessive_none_fraction",
    "is_clean_context_for_transition_focus",
    "is_valid_for_initiation_termination_focus",
]

# Columns used to compute condition-specific performance.
# These columns answer the main debugging question:
# "In which type of window does the best model fail more?"
#
# Raw metadata columns preserve the original dataset information.
# Derived flag columns provide explicit yes/no conditions.
CONDITION_COLUMNS_TO_ANALYZE = [
    "path_type_name",
    "static_type_name",
    "transition_type",
    "transition_phase_interpretation",
    "gait_phase_name",
    "activity_detail_name",
    "task",
    "test_type",
    "is_curved_window",
    "is_straight_window",
    "is_transition_window",
    "is_robust_window_bool",
    "is_non_robust_window",
    "is_standing_window",
    "is_sitting_window",
    "is_stairs_window",
    "is_ascending_window",
    "is_descending_window",
    "is_walking_with_path_type_none",
    "is_walking_with_standing_static_type",
    "is_walking_with_static_type_standing_or_sitting",
    "is_static2walking_transition",
    "is_walking2static_transition",
    "has_curved_path_involvement",
    "has_sitting_context",
    "is_tug_context",
    "has_excessive_none_fraction",
    "is_clean_context_for_transition_focus",
    "is_valid_for_initiation_termination_focus",
    "final_label_fraction_bin",
    "valid_label_fraction_bin",
    "true_class_and_final_label_fraction_bin",
    "true_class_and_valid_label_fraction_bin",
]

# Conditions with very few windows are still saved, but this threshold can be
# used later to filter unreliable rows during interpretation.
MIN_WINDOWS_FOR_CONDITION_REPORT = 1

# Row-wise consistency check keys.
# These columns should identify each validation window uniquely and in the same
# order in both the official validation CSV and the prediction CSV.
ROWWISE_KEY_COLUMNS = [
    "subject_id",
    "file_name",
    "window_start_sample",
    "window_end_sample",
]

# Maximum number of suspicious/ambiguous windows saved in compact sample files.
# Full files are saved separately, while sample files are useful for quick manual
# inspection.
AMBIGUOUS_WINDOW_SAMPLE_SIZE = 100

# Window configuration where gait initiation and gait termination are analyzed.
# This analysis is intentionally restricted to 1-second windows because longer
# windows may contain a large amount of steady-state walking or static data,
# making the whole-window interpretation as initiation/termination misleading.
TRANSITION_FOCUS_WINDOW_CONFIG = "1s_50p_overlap"

# Minimum valid static/walking content required for the focused transition
# analysis. This threshold is applied to valid_label_fraction, not to
# final_label_fraction. This is important because real transition windows can be
# approximately 50% static and 50% walking, but they should not be dominated by
# none/unlabeled samples.
MIN_VALID_LABEL_FRACTION_FOR_TRANSITION_FOCUS = 0.80

# Minimum curved-path fraction used to mark a window as potentially affected by
# turning/curved walking. A value of 0.01 corresponds to roughly one sample in a
# 100-sample, 1-second window.
MIN_CURVED_PATH_FRACTION_FOR_CONFOUND = 0.01

# If True, stairs are excluded from the focused initiation/termination analysis.
# This keeps the analysis closer to level straight walking.
EXCLUDE_STAIRS_FROM_TRANSITION_FOCUS_ANALYSIS = True

# Optional conservative fallback. If no explicit sitting sample-fraction
# metadata is available, the TUG task can be treated as a potential sitting
# context. The default is False to avoid excluding the whole TUG task only
# because the task may contain sit-to-stand and stand-to-sit phases.
EXCLUDE_TUG_FROM_TRANSITION_FOCUS_ANALYSIS = False

# =====================================================
# ANALYSIS FUNCTIONS
# =====================================================

def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add readable error columns to the prediction dataframe.

    Error types:
    - correct_static
    - correct_walking
    - false_positive_static_as_walking
    - false_negative_walking_as_static
    """

    df = df.copy()

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)

    df["y_true_name"] = df["y_true"].map(LABEL_NAME_MAP)
    df["y_pred_name"] = df["y_pred"].map(LABEL_NAME_MAP)

    df["is_correct"] = df["y_true"] == df["y_pred"]

    df["error_type"] = "correct"

    df.loc[
        (df["y_true"] == 0) & (df["y_pred"] == 0),
        "error_type"
    ] = "correct_static"

    df.loc[
        (df["y_true"] == 1) & (df["y_pred"] == 1),
        "error_type"
    ] = "correct_walking"

    df.loc[
        (df["y_true"] == 0) & (df["y_pred"] == 1),
        "error_type"
    ] = "false_positive_static_as_walking"

    df.loc[
        (df["y_true"] == 1) & (df["y_pred"] == 0),
        "error_type"
    ] = "false_negative_walking_as_static"

    return df

def normalize_text_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return a normalized text version of a metadata column.

    The function is intentionally robust:
    - if the column exists, values are converted to lowercase strings;
    - if the column does not exist, a synthetic 'missing' column is returned.

    This avoids errors when different prediction files contain slightly
    different metadata columns.
    """

    if column not in df.columns:
        return pd.Series(
            ["missing"] * len(df),
            index=df.index,
            dtype="string"
        )

    return (
        df[column]
        .astype("string")
        .fillna("missing")
        .str.lower()
        .str.strip()
    )


def parse_boolean_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Convert a metadata column to a boolean Series.

    This is needed because CSV files may store boolean values in different ways:
    True / False, true / false, 1 / 0, yes / no.
    Missing columns are interpreted as False.
    """

    if column not in df.columns:
        return pd.Series(
            [False] * len(df),
            index=df.index,
            dtype=bool
        )

    return (
        df[column]
        .astype("string")
        .fillna("false")
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes"])
    )


def get_original_multiclass_label(df: pd.DataFrame) -> pd.Series:
    """
    Recover the original multiclass label when available.

    For binary GSD, y_true is usually:
    - 0 = static
    - 1 = walking

    However, some files may still contain an original multiclass label where:
    - 2 = ascending
    - 3 = descending

    If no suitable column exists, a NaN Series is returned.
    """

    candidate_columns = [
        "original_label",
        "original_window_label",
        "window_label",
        "label",
    ]

    for column in candidate_columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")

    return pd.Series(
        [np.nan] * len(df),
        index=df.index
    )


def add_window_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit boolean flags describing each validation window.

    These flags make the misclassification analysis more interpretable.
    Instead of relying only on raw metadata columns, the script creates direct
    yes/no variables such as:
    - is this window curved?
    - is this window a transition?
    - is this window robust?
    - is this window standing?
    - is this window related to stairs?
    """

    df = df.copy()

    path_type = normalize_text_column(df, "path_type_name")
    activity_detail = normalize_text_column(df, "activity_detail_name")
    static_type = normalize_text_column(df, "static_type_name")
    transition_type = normalize_text_column(df, "transition_type")
    gait_phase = normalize_text_column(df, "gait_phase_name")
    task = normalize_text_column(df, "task")
    test_type = normalize_text_column(df, "test_type")
    label_name = normalize_text_column(df, "label_name")

    # Combine several metadata fields because different datasets may encode the
    # same information in different columns.
    combined_context = (
        activity_detail + " " +
        static_type + " " +
        transition_type + " " +
        gait_phase + " " +
        task + " " +
        test_type + " " +
        label_name
    )

    original_label = get_original_multiclass_label(df)

    # ------------------------------------------------------------
    # Path-type flags
    # ------------------------------------------------------------

    df["is_curved_window"] = (
        path_type.str.contains("curved", na=False) |
        path_type.str.contains("turn", na=False)
    )

    df["is_straight_window"] = (
        path_type.str.contains("straight", na=False)
    )

    # ------------------------------------------------------------
    # Transition flags
    # ------------------------------------------------------------

    # A window is considered transitional if transition_type is meaningful and
    # not equal to neutral/missing values.
    df["is_transition_window"] = ~transition_type.isin(
        ["none", "nan", "missing", "", "no_transition"]
    )

    # ------------------------------------------------------------
    # Robustness flags
    # ------------------------------------------------------------

    df["is_robust_window_bool"] = parse_boolean_column(
        df,
        "is_robust_window"
    )

    df["is_non_robust_window"] = ~df["is_robust_window_bool"]

    # ------------------------------------------------------------
    # True binary class flags
    # ------------------------------------------------------------

    df["is_static_true"] = df["y_true"].astype(int) == 0
    df["is_walking_true"] = df["y_true"].astype(int) == 1

    # ------------------------------------------------------------
    # Static subtype flags
    # ------------------------------------------------------------

    df["is_standing_window"] = (
        static_type.str.contains("standing", na=False) |
        static_type.str.contains("stand", na=False) |
        combined_context.str.contains("standing", na=False) |
        combined_context.str.contains("balance", na=False)
    )

    df["is_sitting_window"] = (
        static_type.str.contains("sitting", na=False) |
        static_type.str.contains("sit", na=False) |
        combined_context.str.contains("sitting", na=False)
    )

    # ------------------------------------------------------------
    # Stairs-related flags
    # ------------------------------------------------------------

    # Ascending/descending are detected from both text metadata and original
    # multiclass labels when available.
    # Convention, if preserved:
    # 2 = ascending
    # 3 = descending
    df["is_ascending_window"] = (
        combined_context.str.contains("ascending", na=False) |
        combined_context.str.contains("ascent", na=False) |
        combined_context.str.contains("stairs up", na=False) |
        combined_context.str.contains("upstairs", na=False) |
        (original_label == 2)
    )

    df["is_descending_window"] = (
        combined_context.str.contains("descending", na=False) |
        combined_context.str.contains("descent", na=False) |
        combined_context.str.contains("stairs down", na=False) |
        combined_context.str.contains("downstairs", na=False) |
        (original_label == 3)
    )

    df["is_stairs_window"] = (
        df["is_ascending_window"] |
        df["is_descending_window"] |
        combined_context.str.contains("stairs", na=False) |
        combined_context.str.contains("stair", na=False)
    )

    # ------------------------------------------------------------
    # Ambiguous metadata flags
    # ------------------------------------------------------------
    # These flags are not necessarily construction bugs.
    # They identify windows that deserve manual inspection because the binary
    # label and the contextual metadata appear potentially inconsistent.
    #
    # Example:
    # - y_true = walking but path_type_name = none
    # - y_true = walking but static_type_name = standing

    df["is_walking_with_path_type_none"] = (
        df["is_walking_true"] &
        path_type.isin(["none", "nan", "missing", ""])
    )

    df["is_walking_with_standing_static_type"] = (
        df["is_walking_true"] &
        (
            static_type.str.contains("standing", na=False) |
            static_type.str.contains("stand", na=False)
        )
    )

    # This broader flag is used only for debugging ambiguous metadata cases.
    # It identifies true walking windows where static_type_name still reports
    # a static subtype. This is not automatically a bug, because it may happen
    # in mixed windows, transition windows or metadata assigned by majority vote.
    df["is_walking_with_static_type_standing_or_sitting"] = (
        df["is_walking_true"] &
        (
            static_type.str.contains("standing", na=False) |
            static_type.str.contains("stand", na=False) |
            static_type.str.contains("sitting", na=False) |
            static_type.str.contains("sit", na=False)
        )
    )

    return df

def add_transition_focus_columns(
    df: pd.DataFrame,
    window_config_name: str
) -> pd.DataFrame:
    """
    Add focused transition-analysis columns.

    The goal is to separate:
    - robust static windows;
    - robust walking windows;
    - gait initiation windows, approximated by Static2Walking;
    - gait termination windows, approximated by Walking2Static.

    Important methodological detail:
    transition_type is expected to describe only the binary GSD transition
    between static and walking. Therefore, confounding situations such as
    turning or sitting are not expected to appear as separate transition_type
    values. They are derived here from path, static-context and fraction
    metadata when those fields are available.
    """

    df = df.copy()

    transition_type = normalize_text_column(df, "transition_type")
    path_type = normalize_text_column(df, "path_type_name")
    static_type = normalize_text_column(df, "static_type_name")
    activity_detail = normalize_text_column(df, "activity_detail_name")
    gait_phase = normalize_text_column(df, "gait_phase_name")
    task = normalize_text_column(df, "task")
    test_type = normalize_text_column(df, "test_type")
    challenge_type = normalize_text_column(df, "challenge_type")

    combined_context = (
        activity_detail + " " +
        gait_phase + " " +
        task + " " +
        test_type + " " +
        challenge_type + " " +
        static_type
    )

    def get_numeric_column(
        column: str,
        default_value: float = np.nan
    ) -> pd.Series:
        """
        Return a numeric metadata column when available.

        Missing columns are filled with default_value so that the analysis can
        still run on older prediction files generated before the new metadata
        propagation was added.
        """

        if column not in df.columns:
            return pd.Series(
                [default_value] * len(df),
                index=df.index,
                dtype=float
            )

        return pd.to_numeric(df[column], errors="coerce")

    # Normalize transition labels by removing separators. This makes the logic
    # robust to formats such as Static2Walking, static_2_walking or
    # static-to-walking.
    transition_type_norm = (
        transition_type
        .str.replace("_", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    df["transition_type_normalized"] = transition_type_norm

    # Static2Walking is interpreted as gait initiation.
    # Walking2Static is interpreted as gait termination.
    # These are analysis labels only: the model still predicts static/walking.
    df["is_static2walking_transition"] = (
        transition_type_norm == "static2walking"
    )

    df["is_walking2static_transition"] = (
        transition_type_norm == "walking2static"
    )

    df["transition_phase_interpretation"] = "non_transition"

    df.loc[
        df["is_static2walking_transition"],
        "transition_phase_interpretation"
    ] = "gait_initiation"

    df.loc[
        df["is_walking2static_transition"],
        "transition_phase_interpretation"
    ] = "gait_termination"

    other_transition_mask = (
        df["is_transition_window"]
        &
        ~df["is_static2walking_transition"]
        &
        ~df["is_walking2static_transition"]
    )

    df.loc[
        other_transition_mask,
        "transition_phase_interpretation"
    ] = "other_transition"

    # ------------------------------------------------------------
    # Fraction metadata
    # ------------------------------------------------------------
    # final_label_fraction describes how dominant the final assigned label is.
    # valid_label_fraction describes how much of the window is covered by either
    # static or walking, independently from which one is dominant.
    #
    # For transition analysis, valid_label_fraction is the safer quantity:
    # a true initiation/termination window can be mixed static/walking, but it
    # should not be dominated by none/unlabeled samples.

    final_label_fraction = get_numeric_column(
        "final_label_fraction",
        default_value=np.nan
    )

    valid_label_fraction = get_numeric_column(
        "valid_label_fraction",
        default_value=np.nan
    )

    none_label_fraction = get_numeric_column(
        "none_label_fraction",
        default_value=np.nan
    )

    n_static_samples = get_numeric_column(
        "n_static_samples",
        default_value=np.nan
    )

    n_walking_samples = get_numeric_column(
        "n_walking_samples",
        default_value=np.nan
    )

    n_none_label_samples = get_numeric_column(
        "n_none_label_samples",
        default_value=np.nan
    )

    # Fallback for older files:
    # if valid_label_fraction is missing but sample counters exist, reconstruct
    # the fraction from static/walking/none counters.
    reconstructed_denominator = (
        n_static_samples + n_walking_samples + n_none_label_samples
    )

    reconstructed_valid_fraction = np.where(
        reconstructed_denominator > 0,
        (n_static_samples + n_walking_samples) / reconstructed_denominator,
        np.nan
    )

    valid_label_fraction = valid_label_fraction.fillna(
        pd.Series(
            reconstructed_valid_fraction,
            index=df.index,
            dtype=float
        )
    )

    # Last fallback for very old files:
    # if no fraction information exists, keep the row instead of excluding it.
    # This avoids silently losing all transition-focused rows simply because the
    # metadata was not available in an older run.
    valid_label_fraction = valid_label_fraction.fillna(1.0)

    none_label_fraction = none_label_fraction.fillna(
        1.0 - valid_label_fraction
    )

    df["final_label_fraction_numeric"] = final_label_fraction
    df["valid_label_fraction_numeric"] = valid_label_fraction
    df["none_label_fraction_numeric"] = none_label_fraction

    df["has_excessive_none_fraction"] = (
        valid_label_fraction < MIN_VALID_LABEL_FRACTION_FOR_TRANSITION_FOCUS
    )

    # ------------------------------------------------------------
    # Curved/turning confound
    # ------------------------------------------------------------
    # path_type_name may be none for final static windows by design. Therefore,
    # curved involvement is derived not only from path_type_name, but also from
    # curved_path_fraction when it is available. This preserves information about
    # minority curved samples inside mixed windows.

    curved_path_fraction = get_numeric_column(
        "curved_path_fraction",
        default_value=0.0
    ).fillna(0.0)

    straight_path_fraction = get_numeric_column(
        "straight_path_fraction",
        default_value=0.0
    ).fillna(0.0)

    df["curved_path_fraction_numeric"] = curved_path_fraction
    df["straight_path_fraction_numeric"] = straight_path_fraction

    df["has_curved_path_involvement"] = (
        path_type.str.contains("curved", na=False)
        |
        path_type.str.contains("turn", na=False)
        |
        (curved_path_fraction >= MIN_CURVED_PATH_FRACTION_FOR_CONFOUND)
    )

    df["has_straight_path_involvement"] = (
        path_type.str.contains("straight", na=False)
        |
        (straight_path_fraction > 0.0)
    )

    # ------------------------------------------------------------
    # Sitting confound
    # ------------------------------------------------------------
    # static_type_name may be none for final walking windows by design. Therefore,
    # this flag is conservative but not perfect unless sample-wise sitting
    # counters are added during dataset construction.
    #
    # If future construction metadata includes sitting_fraction, this code will
    # automatically use it.

    sitting_fraction = get_numeric_column(
        "sitting_fraction",
        default_value=0.0
    ).fillna(0.0)

    df["sitting_fraction_numeric"] = sitting_fraction

    explicit_sitting_text = (
        static_type.str.contains(r"\bsitting\b", regex=True, na=False)
        |
        static_type.str.contains(r"\bsit\b", regex=True, na=False)
        |
        combined_context.str.contains(r"\bsitting\b", regex=True, na=False)
        |
        combined_context.str.contains(r"\bsit\b", regex=True, na=False)
        |
        combined_context.str.contains(r"\bseated\b", regex=True, na=False)
        |
        combined_context.str.contains(r"\bchair\b", regex=True, na=False)
    )

    df["is_tug_context"] = (
        task.str.contains("tug", na=False)
        |
        test_type.str.contains("tug", na=False)
        |
        activity_detail.str.contains("tug", na=False)
    )

    df["has_sitting_context"] = (
        explicit_sitting_text
        |
        (sitting_fraction > 0.0)
    )

    if EXCLUDE_TUG_FROM_TRANSITION_FOCUS_ANALYSIS:
        df["has_sitting_context"] = (
            df["has_sitting_context"]
            |
            df["is_tug_context"]
        )

    # ------------------------------------------------------------
    # Clean context for transition-focused analysis
    # ------------------------------------------------------------
    # A window is considered clean for initiation/termination analysis only if:
    # - it is not affected by curved/turning path involvement;
    # - it is not affected by sitting context;
    # - it is not related to stairs, if stairs are excluded;
    # - it is not dominated by none/unlabeled samples.

    if EXCLUDE_STAIRS_FROM_TRANSITION_FOCUS_ANALYSIS:
        has_stairs_confound = (
            df["is_stairs_window"]
            .fillna(False)
            .astype(bool)
        )
    else:
        has_stairs_confound = pd.Series(
            [False] * len(df),
            index=df.index,
            dtype=bool
        )

    df["has_stairs_confound_for_transition_focus"] = has_stairs_confound

    df["is_clean_context_for_transition_focus"] = (
        ~df["has_curved_path_involvement"].fillna(False).astype(bool)
        &
        ~df["has_sitting_context"].fillna(False).astype(bool)
        &
        ~df["has_stairs_confound_for_transition_focus"].fillna(False).astype(bool)
        &
        ~df["has_excessive_none_fraction"].fillna(False).astype(bool)
    )

    # The focused initiation/termination analysis is enabled only for the
    # configured short-window setting.
    df["is_transition_focus_window_config"] = (
        window_config_name == TRANSITION_FOCUS_WINDOW_CONFIG
    )

    df["is_valid_for_initiation_termination_focus"] = (
        df["is_transition_focus_window_config"]
        &
        df["is_clean_context_for_transition_focus"]
        &
        (
            df["is_static2walking_transition"]
            |
            df["is_walking2static_transition"]
        )
    )

    # ------------------------------------------------------------
    # Fraction bins for interpretability
    # ------------------------------------------------------------

    df["final_label_fraction_bin"] = pd.cut(
        final_label_fraction,
        bins=[0.0, 0.50, 0.60, 0.70, 0.80, 1.000001],
        labels=[
            "0-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-100%",
        ],
        include_lowest=True,
        right=False
    )

    df["valid_label_fraction_bin"] = pd.cut(
        valid_label_fraction,
        bins=[0.0, 0.50, 0.60, 0.70, 0.80, 1.000001],
        labels=[
            "0-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-100%",
        ],
        include_lowest=True,
        right=False
    )

    df["true_class_and_final_label_fraction_bin"] = (
        df["y_true_name"].astype(str)
        +
        "__"
        +
        df["final_label_fraction_bin"].astype(str)
    )

    df["true_class_and_valid_label_fraction_bin"] = (
        df["y_true_name"].astype(str)
        +
        "__"
        +
        df["valid_label_fraction_bin"].astype(str)
    )

    return df


def summarize_robustness_class_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance across robust and non-robust static/walking windows.

    This table answers:
    - how well does the model classify robust static windows?
    - how well does it classify non-robust static windows?
    - how well does the model classify robust walking windows?
    - how well does it classify non-robust walking windows?
    """

    df = df.copy()

    rows = []

    conditions = {
        "static_robust": (
            (df["y_true"].astype(int) == 0)
            &
            df["is_robust_window_bool"].fillna(False).astype(bool)
        ),
        "static_non_robust": (
            (df["y_true"].astype(int) == 0)
            &
            ~df["is_robust_window_bool"].fillna(False).astype(bool)
        ),
        "walking_robust": (
            (df["y_true"].astype(int) == 1)
            &
            df["is_robust_window_bool"].fillna(False).astype(bool)
        ),
        "walking_non_robust": (
            (df["y_true"].astype(int) == 1)
            &
            ~df["is_robust_window_bool"].fillna(False).astype(bool)
        ),
    }

    for condition_name, mask in conditions.items():

        subset = df[mask].copy()
        metrics = compute_binary_performance_metrics(subset)

        metrics["analysis_name"] = "robust_vs_non_robust_by_true_class"
        metrics["condition_name"] = condition_name

        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    ordered_columns = [
        "analysis_name",
        "condition_name",
        "n_windows",
        "n_true_static",
        "n_true_walking",
        "n_errors",
        "error_rate_percent",
        "accuracy",
        "recall_static",
        "recall_walking",
        "balanced_accuracy_available_classes",
        "n_false_positive_static_as_walking",
        "n_false_negative_walking_as_static",
    ]

    existing_columns = [
        col for col in ordered_columns
        if col in summary.columns
    ]

    remaining_columns = [
        col for col in summary.columns
        if col not in existing_columns
    ]

    return summary[existing_columns + remaining_columns]


def summarize_transition_type_focus_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize performance by transition type.

    This table keeps Static2Walking and Walking2Static separated.
    It is useful to check whether errors are more frequent during gait
    initiation or gait termination.
    """

    transition_subset = df[
        df["is_transition_window"].fillna(False).astype(bool)
    ].copy()

    if transition_subset.empty:
        return pd.DataFrame()

    rows = []

    for transition_value, subset in transition_subset.groupby(
        "transition_phase_interpretation",
        dropna=False
    ):

        metrics = compute_binary_performance_metrics(subset)

        metrics["analysis_name"] = "all_transition_types"
        metrics["transition_phase_interpretation"] = transition_value

        if "transition_type" in subset.columns:
            metrics["raw_transition_types"] = ";".join(
                sorted(
                    subset["transition_type"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )

        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary = summary.sort_values(
        by=["n_errors", "error_rate_percent"],
        ascending=False
    ).reset_index(drop=True)

    return summary


def summarize_clean_initiation_termination_focus(
    df: pd.DataFrame,
    window_config_name: str
) -> pd.DataFrame:
    """
    Compare robust static, robust walking, gait initiation and gait termination.

    Important methodological choice:
    - gait initiation and gait termination are analyzed only for 1-second windows;
    - curved/turning, sitting, stairs and none-dominated windows are removed;
    - this isolates the transition effect as much as possible.
    """

    if window_config_name != TRANSITION_FOCUS_WINDOW_CONFIG:
        return pd.DataFrame()

    clean_context = (
        df["is_clean_context_for_transition_focus"]
        .fillna(False)
        .astype(bool)
    )

    robust_mask = (
        df["is_robust_window_bool"]
        .fillna(False)
        .astype(bool)
    )

    non_transition_mask = (
        ~df["is_transition_window"]
        .fillna(False)
        .astype(bool)
    )

    conditions = {
        "clean_static_robust_non_transition": (
            clean_context
            &
            non_transition_mask
            &
            robust_mask
            &
            (df["y_true"].astype(int) == 0)
        ),
        "clean_walking_robust_non_transition": (
            clean_context
            &
            non_transition_mask
            &
            robust_mask
            &
            (df["y_true"].astype(int) == 1)
        ),
        "clean_gait_initiation_static2walking": (
            clean_context
            &
            df["is_static2walking_transition"]
            .fillna(False)
            .astype(bool)
        ),
        "clean_gait_termination_walking2static": (
            clean_context
            &
            df["is_walking2static_transition"]
            .fillna(False)
            .astype(bool)
        ),
    }

    rows = []

    for condition_name, mask in conditions.items():

        subset = df[mask].copy()
        metrics = compute_binary_performance_metrics(subset)

        metrics["analysis_name"] = "clean_1s_transition_focus"
        metrics["condition_name"] = condition_name
        metrics["window_config_restriction"] = TRANSITION_FOCUS_WINDOW_CONFIG
        metrics["curved_sitting_stairs_none_excluded"] = True

        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    ordered_columns = [
        "analysis_name",
        "condition_name",
        "n_windows",
        "n_true_static",
        "n_true_walking",
        "n_errors",
        "error_rate_percent",
        "accuracy",
        "recall_static",
        "recall_walking",
        "balanced_accuracy_available_classes",
        "n_false_positive_static_as_walking",
        "n_false_negative_walking_as_static",
        "window_config_restriction",
        "curved_sitting_stairs_none_excluded",
    ]

    existing_columns = [
        col for col in ordered_columns
        if col in summary.columns
    ]

    remaining_columns = [
        col for col in summary.columns
        if col not in existing_columns
    ]

    return summary[existing_columns + remaining_columns]


def summarize_label_fraction_bin_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize performance by final-label fraction bins.

    This analysis checks whether the model performs worse when the final window
    label covers a smaller percentage of the window.
    """

    if "final_label_fraction_numeric" not in df.columns:
        return pd.DataFrame()

    if df["final_label_fraction_numeric"].isna().all():
        return pd.DataFrame()

    return summarize_performance_by_column(
        df=df,
        column="true_class_and_final_label_fraction_bin"
    )


def summarize_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Summarize model errors for one metadata column.

    For each metadata value, the table reports:
    - number of windows belonging to that value;
    - percentage of all validation windows belonging to that value;
    - number of errors within that value;
    - error rate within that value;
    - percentage of all errors explained by that value;
    - false positives and false negatives separately.

    This distinction is important:
    - error_rate_within_value tells whether that condition is difficult;
    - pct_all_errors tells how much that condition contributes to total errors.
    """

    if column not in df.columns:
        return pd.DataFrame()

    total_windows = len(df)
    total_errors = int((~df["is_correct"]).sum())

    grouped = (
        df
        .groupby(column, dropna=False)
        .agg(
            n_windows=("error_type", "size"),
            n_correct=("is_correct", "sum"),
            n_true_static=("y_true", lambda x: np.sum(x.astype(int) == 0)),
            n_true_walking=("y_true", lambda x: np.sum(x.astype(int) == 1)),
            n_false_positive=(
                "error_type",
                lambda x: np.sum(x == "false_positive_static_as_walking")
            ),
            n_false_negative=(
                "error_type",
                lambda x: np.sum(x == "false_negative_walking_as_static")
            ),
        )
        .reset_index()
    )

    grouped["n_errors"] = (
        grouped["n_false_positive"] +
        grouped["n_false_negative"]
    )

    grouped["pct_all_windows"] = grouped["n_windows"] / total_windows * 100.0

    grouped["error_rate_within_value"] = (
        grouped["n_errors"] / grouped["n_windows"] * 100.0
    )

    grouped["pct_all_errors"] = np.where(
        total_errors > 0,
        grouped["n_errors"] / total_errors * 100.0,
        0.0
    )

    grouped["false_positive_rate_within_true_static"] = np.where(
        grouped["n_true_static"] > 0,
        grouped["n_false_positive"] / grouped["n_true_static"] * 100.0,
        np.nan
    )

    grouped["false_negative_rate_within_true_walking"] = np.where(
        grouped["n_true_walking"] > 0,
        grouped["n_false_negative"] / grouped["n_true_walking"] * 100.0,
        np.nan
    )

    grouped = grouped.sort_values(
        by=["n_errors", "error_rate_within_value"],
        ascending=False
    )

    return grouped

def compute_binary_performance_metrics(subset: pd.DataFrame) -> dict:
    """
    Compute binary GSD performance metrics for one subset of windows.

    This function is used for condition-specific analysis, for example:
    - curved windows only;
    - straight windows only;
    - standing windows only;
    - transition windows only.

    Important note:
    some conditions may contain only one true class.
    For example, standing windows may contain only true static samples.
    In that case, static recall is meaningful, while walking recall is NaN.
    """

    n_windows = len(subset)

    if n_windows == 0:
        return {
            "n_windows": 0,
            "n_true_static": 0,
            "n_true_walking": 0,
            "n_pred_static": 0,
            "n_pred_walking": 0,
            "n_correct": 0,
            "n_errors": 0,
            "n_correct_static": 0,
            "n_correct_walking": 0,
            "n_false_positive_static_as_walking": 0,
            "n_false_negative_walking_as_static": 0,
            "accuracy": np.nan,
            "recall_static": np.nan,
            "recall_walking": np.nan,
            "balanced_accuracy_if_both_classes_present": np.nan,
            "balanced_accuracy_available_classes": np.nan,
            "error_rate_percent": np.nan,
        }

    y_true = subset["y_true"].astype(int)
    y_pred = subset["y_pred"].astype(int)

    true_static_mask = y_true == 0
    true_walking_mask = y_true == 1
    pred_static_mask = y_pred == 0
    pred_walking_mask = y_pred == 1

    correct_static = true_static_mask & pred_static_mask
    correct_walking = true_walking_mask & pred_walking_mask

    false_positive = true_static_mask & pred_walking_mask
    false_negative = true_walking_mask & pred_static_mask

    n_true_static = int(true_static_mask.sum())
    n_true_walking = int(true_walking_mask.sum())

    n_correct_static = int(correct_static.sum())
    n_correct_walking = int(correct_walking.sum())

    n_false_positive = int(false_positive.sum())
    n_false_negative = int(false_negative.sum())

    n_correct = n_correct_static + n_correct_walking
    n_errors = n_false_positive + n_false_negative

    recall_static = (
        n_correct_static / n_true_static
        if n_true_static > 0 else np.nan
    )

    recall_walking = (
        n_correct_walking / n_true_walking
        if n_true_walking > 0 else np.nan
    )

    # This is the strict balanced accuracy. It is defined only when both true
    # classes are present in the condition.
    if n_true_static > 0 and n_true_walking > 0:
        balanced_accuracy_if_both = (recall_static + recall_walking) / 2.0
    else:
        balanced_accuracy_if_both = np.nan

    # This alternative metric is useful for single-class conditions.
    # Example: for standing-only windows, it corresponds to static recall.
    available_recalls = [
        value for value in [recall_static, recall_walking]
        if not pd.isna(value)
    ]

    balanced_accuracy_available = (
        float(np.mean(available_recalls))
        if len(available_recalls) > 0 else np.nan
    )

    return {
        "n_windows": n_windows,
        "n_true_static": n_true_static,
        "n_true_walking": n_true_walking,
        "n_pred_static": int(pred_static_mask.sum()),
        "n_pred_walking": int(pred_walking_mask.sum()),
        "n_correct": n_correct,
        "n_errors": n_errors,
        "n_correct_static": n_correct_static,
        "n_correct_walking": n_correct_walking,
        "n_false_positive_static_as_walking": n_false_positive,
        "n_false_negative_walking_as_static": n_false_negative,
        "accuracy": n_correct / n_windows,
        "recall_static": recall_static,
        "recall_walking": recall_walking,
        "balanced_accuracy_if_both_classes_present": balanced_accuracy_if_both,
        "balanced_accuracy_available_classes": balanced_accuracy_available,
        "error_rate_percent": n_errors / n_windows * 100.0,
    }


def summarize_performance_by_column(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Compute performance metrics separately for each value of one condition column.

    Example outputs:
    - performance for path_type_name = straight;
    - performance for path_type_name = curved;
    - performance for static_type_name = standing;
    - performance for is_transition_window = True.

    This table is more informative than an error-count table because it reports:
    - class distribution inside each condition;
    - accuracy;
    - static recall;
    - walking recall;
    - balanced accuracy where meaningful;
    - false positives and false negatives.
    """

    if column not in df.columns:
        return pd.DataFrame()

    total_windows = len(df)
    total_errors = int((~df["is_correct"]).sum())

    rows = []

    for condition_value, subset in df.groupby(column, dropna=False):

        metrics = compute_binary_performance_metrics(subset)

        if metrics["n_windows"] < MIN_WINDOWS_FOR_CONDITION_REPORT:
            continue

        metrics[column] = condition_value
        metrics["pct_all_windows"] = (
            metrics["n_windows"] / total_windows * 100.0
            if total_windows > 0 else np.nan
        )
        metrics["pct_all_errors"] = (
            metrics["n_errors"] / total_errors * 100.0
            if total_errors > 0 else 0.0
        )

        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    # Move the condition value to the first column.
    ordered_columns = [column] + [
        col for col in summary.columns
        if col != column
    ]

    summary = summary[ordered_columns]

    summary = summary.sort_values(
        by=[
            "n_errors",
            "error_rate_percent",
            "n_windows",
        ],
        ascending=False
    ).reset_index(drop=True)

    return summary

def summarize_performance_by_column_with_filter(
    df: pd.DataFrame,
    column: str,
    applicability_mask: pd.Series,
    analysis_name: str
) -> pd.DataFrame:
    """
    Compute condition-specific performance only on windows where the metadata
    column is meaningful.

    This avoids misleading interpretations such as comparing:
    - path_type_name = none with straight/curved on static windows;
    - static_type_name = none with standing/sitting on walking windows.

    The function keeps the denominator explicit by reporting both:
    - percentage relative to the filtered applicable subset;
    - percentage relative to the full validation set.
    """

    if column not in df.columns:
        return pd.DataFrame()

    applicable_df = df[applicability_mask].copy()

    if applicable_df.empty:
        return pd.DataFrame()

    total_all_windows = len(df)
    total_applicable_windows = len(applicable_df)
    total_applicable_errors = int((~applicable_df["is_correct"]).sum())

    rows = []

    for condition_value, subset in applicable_df.groupby(column, dropna=False):

        metrics = compute_binary_performance_metrics(subset)

        metrics[column] = condition_value
        metrics["analysis_name"] = analysis_name

        metrics["n_applicable_windows"] = total_applicable_windows

        metrics["pct_of_applicable_windows"] = (
            metrics["n_windows"] / total_applicable_windows * 100.0
            if total_applicable_windows > 0 else np.nan
        )

        metrics["pct_of_all_validation_windows"] = (
            metrics["n_windows"] / total_all_windows * 100.0
            if total_all_windows > 0 else np.nan
        )

        metrics["pct_of_applicable_errors"] = (
            metrics["n_errors"] / total_applicable_errors * 100.0
            if total_applicable_errors > 0 else 0.0
        )

        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    ordered_columns = [
        "analysis_name",
        column,
        "n_windows",
        "n_applicable_windows",
        "pct_of_applicable_windows",
        "pct_of_all_validation_windows",
        "n_true_static",
        "n_true_walking",
        "n_errors",
        "error_rate_percent",
        "accuracy",
        "recall_static",
        "recall_walking",
        "balanced_accuracy_if_both_classes_present",
        "balanced_accuracy_available_classes",
        "n_false_positive_static_as_walking",
        "n_false_negative_walking_as_static",
        "pct_of_applicable_errors",
    ]

    existing_ordered_columns = [
        col for col in ordered_columns
        if col in summary.columns
    ]

    remaining_columns = [
        col for col in summary.columns
        if col not in existing_ordered_columns
    ]

    summary = summary[existing_ordered_columns + remaining_columns]

    summary = summary.sort_values(
        by=["n_errors", "error_rate_percent"],
        ascending=False
    ).reset_index(drop=True)

    return summary

def summarize_error_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count global error types.
    """

    summary = (
        df["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="n_windows")
    )

    summary["percentage"] = summary["n_windows"] / len(df) * 100.0

    return summary

def summarize_flag_error_composition(
    df: pd.DataFrame,
    flag_columns: list
) -> pd.DataFrame:
    """
    Summarize how much each window-type flag contributes to the errors.

    For each boolean flag, the table reports:
    - how many validation windows have that flag;
    - how many errors have that flag;
    - the error rate inside flagged windows;
    - the percentage of total errors explained by that flag;
    - false positives and false negatives within that flag.

    Example:
    If is_curved_window has pct_all_errors_with_flag = 40%,
    then 40% of all model errors occurred in curved windows.
    """

    rows = []

    total_windows = len(df)
    total_errors = int((~df["is_correct"]).sum())
    total_false_positive = int(
        np.sum(df["error_type"] == "false_positive_static_as_walking")
    )
    total_false_negative = int(
        np.sum(df["error_type"] == "false_negative_walking_as_static")
    )

    for flag in flag_columns:

        if flag not in df.columns:
            continue

        flag_mask = df[flag].fillna(False).astype(bool)
        error_mask = ~df["is_correct"]

        n_windows_with_flag = int(flag_mask.sum())
        n_errors_with_flag = int((flag_mask & error_mask).sum())

        n_false_positive_with_flag = int(
            (
                flag_mask &
                (df["error_type"] == "false_positive_static_as_walking")
            ).sum()
        )

        n_false_negative_with_flag = int(
            (
                flag_mask &
                (df["error_type"] == "false_negative_walking_as_static")
            ).sum()
        )

        rows.append({
            "flag": flag,
            "n_windows_with_flag": n_windows_with_flag,
            "pct_all_windows_with_flag": (
                n_windows_with_flag / total_windows * 100.0
                if total_windows > 0 else 0.0
            ),
            "n_errors_with_flag": n_errors_with_flag,
            "pct_all_errors_with_flag": (
                n_errors_with_flag / total_errors * 100.0
                if total_errors > 0 else 0.0
            ),
            "error_rate_within_flag": (
                n_errors_with_flag / n_windows_with_flag * 100.0
                if n_windows_with_flag > 0 else np.nan
            ),
            "n_false_positive_with_flag": n_false_positive_with_flag,
            "pct_all_false_positive_with_flag": (
                n_false_positive_with_flag / total_false_positive * 100.0
                if total_false_positive > 0 else 0.0
            ),
            "n_false_negative_with_flag": n_false_negative_with_flag,
            "pct_all_false_negative_with_flag": (
                n_false_negative_with_flag / total_false_negative * 100.0
                if total_false_negative > 0 else 0.0
            ),
        })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            by=["n_errors_with_flag", "error_rate_within_flag"],
            ascending=False
        )

    return summary


def summarize_error_composition_by_error_type(
    df: pd.DataFrame,
    flag_columns: list
) -> pd.DataFrame:
    """
    Summarize window-type flags separately for each error type.

    This helps answer more specific questions, for example:
    - are false positives mainly standing windows?
    - are false negatives mainly walking transition windows?
    - are curved windows more frequent among false negatives or false positives?
    """

    rows = []

    error_types_to_analyze = [
        "false_positive_static_as_walking",
        "false_negative_walking_as_static",
    ]

    for error_type in error_types_to_analyze:

        subset = df[df["error_type"] == error_type].copy()
        n_error_type = len(subset)

        for flag in flag_columns:

            if flag not in subset.columns:
                continue

            n_with_flag = int(subset[flag].fillna(False).astype(bool).sum())

            rows.append({
                "error_type": error_type,
                "flag": flag,
                "n_error_windows": n_error_type,
                "n_error_windows_with_flag": n_with_flag,
                "pct_error_type_with_flag": (
                    n_with_flag / n_error_type * 100.0
                    if n_error_type > 0 else 0.0
                ),
            })

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            by=["error_type", "n_error_windows_with_flag"],
            ascending=[True, False]
        )

    return summary

def summarize_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create subject-level error summary.

    This helps identify whether errors are concentrated in specific subjects.
    """

    required_cols = {"subject_id", "group", "error_type", "is_correct"}

    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    summary = (
        df
        .groupby(["group", "subject_id"], dropna=False)
        .agg(
            n_windows=("error_type", "size"),
            n_correct=("is_correct", "sum"),
            n_false_positive=(
                "error_type",
                lambda x: np.sum(x == "false_positive_static_as_walking")
            ),
            n_false_negative=(
                "error_type",
                lambda x: np.sum(x == "false_negative_walking_as_static")
            ),
        )
        .reset_index()
    )

    summary["n_errors"] = (
        summary["n_false_positive"] +
        summary["n_false_negative"]
    )

    summary["error_rate"] = summary["n_errors"] / summary["n_windows"]

    summary = summary.sort_values(
        by=["n_errors", "error_rate"],
        ascending=False
    )

    return summary


def log_message(message, file_handle=None):
    """
    Print a message to console and optionally save it to a text report.
    """

    print(message)

    if file_handle is not None:
        file_handle.write(str(message) + "\n")

def get_best_n_features_for_model(
    best_by_model_df: pd.DataFrame,
    model_name: str
) -> int:
    """
    Extract the best number of features selected by the bottom-up wrapper.

    The script analyzes the best configuration for each model, not an arbitrary
    top-N feature subset.
    """

    model_rows = best_by_model_df[
        best_by_model_df["model"].astype(str) == model_name
    ].copy()

    if model_rows.empty:
        raise ValueError(
            f"Model '{model_name}' not found in bottomup_wrapper_BEST_by_model.csv."
        )

    return int(model_rows.iloc[0]["n_features"])

def get_best_model_for_window_config(
    best_by_model_df: pd.DataFrame,
    selection_metric: str = "balanced_accuracy"
) -> dict:
    """
    Select the best-performing model for one window configuration.

    The selection is performed on the validation metric saved by the bottom-up
    wrapper. By default, balanced accuracy is used because it gives equal
    importance to static and walking classes.

    The returned dictionary contains:
    - model name;
    - best number of features;
    - validation balanced accuracy;
    - validation accuracy;
    - static recall;
    - walking recall.
    """

    required_columns = {
        "model",
        "n_features",
        "balanced_accuracy",
        "accuracy",
        "recall_static",
        "recall_walking",
    }

    missing_columns = required_columns.difference(best_by_model_df.columns)

    if missing_columns:
        raise ValueError(
            "The best-by-model table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if selection_metric not in best_by_model_df.columns:
        raise ValueError(
            f"Selection metric '{selection_metric}' not found in "
            "bottomup_wrapper_BEST_by_model.csv."
        )

    # Sort by the main metric first.
    # Accuracy and recalls are used only as secondary tie-breakers.
    sorted_df = best_by_model_df.sort_values(
        by=[
            selection_metric,
            "accuracy",
            "recall_static",
            "recall_walking",
        ],
        ascending=False
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]

    return {
        "model": str(best_row["model"]).strip().lower(),
        "n_features": int(best_row["n_features"]),
        "selected_by_metric": selection_metric,
        "validation_balanced_accuracy": float(best_row["balanced_accuracy"]),
        "validation_accuracy": float(best_row["accuracy"]),
        "validation_recall_static": float(best_row["recall_static"]),
        "validation_recall_walking": float(best_row["recall_walking"]),
    }

def normalize_key_dataframe(df: pd.DataFrame, key_columns: list) -> pd.DataFrame:
    """
    Normalize key columns before row-wise comparisons.

    All key values are converted to strings after filling missing values.
    This avoids false mismatches caused only by pandas dtype differences.
    """

    normalized = pd.DataFrame(index=df.index)

    for column in key_columns:
        normalized[column] = (
            df[column]
            .astype("string")
            .fillna("<MISSING>")
            .str.strip()
        )

    return normalized


def make_logical_key(df: pd.DataFrame, key_columns: list) -> pd.Series:
    """
    Build one compact logical key from the selected key columns.

    The key is used only for consistency checks and duplicate detection.
    """

    normalized = normalize_key_dataframe(df, key_columns)

    return normalized.astype(str).agg(" | ".join, axis=1)


def check_rowwise_alignment(
    prediction_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_columns: list,
    output_dir: Path,
    window_config_name: str,
    model_name: str,
    n_features: int
) -> dict:
    """
    Check whether prediction CSV and official validation CSV are aligned row by row.

    This goes beyond checking that the two files have the same number of rows.
    It verifies that each row refers to the same logical window in the same order.
    """

    missing_prediction_keys = [
        column for column in key_columns
        if column not in prediction_df.columns
    ]

    missing_test_keys = [
        column for column in key_columns
        if column not in test_df.columns
    ]

    summary = {
        "window_config": window_config_name,
        "model": model_name,
        "best_n_features": n_features,
        "n_prediction_rows": len(prediction_df),
        "n_test_rows": len(test_df),
        "required_key_columns": ";".join(key_columns),
        "missing_prediction_key_columns": ";".join(missing_prediction_keys),
        "missing_test_key_columns": ";".join(missing_test_keys),
        "rowwise_check_performed": False,
        "same_number_of_rows": len(prediction_df) == len(test_df),
        "n_rowwise_key_mismatches": np.nan,
        "pct_rowwise_key_mismatches": np.nan,
        "n_y_true_label_mismatches": np.nan,
        "same_logical_key_set": np.nan,
        "n_unique_prediction_keys": np.nan,
        "n_unique_test_keys": np.nan,
        "n_keys_only_in_predictions": np.nan,
        "n_keys_only_in_test": np.nan,
    }

    if missing_prediction_keys or missing_test_keys:
        return summary

    if len(prediction_df) != len(test_df):
        return summary

    pred_keys = normalize_key_dataframe(prediction_df, key_columns)
    test_keys = normalize_key_dataframe(test_df, key_columns)

    key_match_mask = (pred_keys == test_keys).all(axis=1)
    n_key_mismatches = int((~key_match_mask).sum())

    prediction_logical_keys = make_logical_key(prediction_df, key_columns)
    test_logical_keys = make_logical_key(test_df, key_columns)

    prediction_key_set = set(prediction_logical_keys)
    test_key_set = set(test_logical_keys)

    keys_only_in_predictions = prediction_key_set.difference(test_key_set)
    keys_only_in_test = test_key_set.difference(prediction_key_set)

    summary["rowwise_check_performed"] = True
    summary["n_rowwise_key_mismatches"] = n_key_mismatches
    summary["pct_rowwise_key_mismatches"] = (
        n_key_mismatches / len(prediction_df) * 100.0
        if len(prediction_df) > 0 else np.nan
    )
    summary["same_logical_key_set"] = prediction_key_set == test_key_set
    summary["n_unique_prediction_keys"] = len(prediction_key_set)
    summary["n_unique_test_keys"] = len(test_key_set)
    summary["n_keys_only_in_predictions"] = len(keys_only_in_predictions)
    summary["n_keys_only_in_test"] = len(keys_only_in_test)

    if "y_true" in prediction_df.columns and "label" in test_df.columns:
        pred_y = pd.to_numeric(prediction_df["y_true"], errors="coerce")
        test_y = pd.to_numeric(test_df["label"], errors="coerce")
        summary["n_y_true_label_mismatches"] = int((pred_y != test_y).sum())

    if n_key_mismatches > 0:
        mismatch_rows = pd.DataFrame({
            "row_index": prediction_df.index[~key_match_mask],
        })

        for column in key_columns:
            mismatch_rows[f"prediction_{column}"] = prediction_df.loc[
                ~key_match_mask,
                column
            ].to_numpy()

            mismatch_rows[f"test_{column}"] = test_df.loc[
                ~key_match_mask,
                column
            ].to_numpy()

        if "y_true" in prediction_df.columns:
            mismatch_rows["prediction_y_true"] = prediction_df.loc[
                ~key_match_mask,
                "y_true"
            ].to_numpy()

        if "label" in test_df.columns:
            mismatch_rows["test_label"] = test_df.loc[
                ~key_match_mask,
                "label"
            ].to_numpy()

        mismatch_rows.to_csv(
            output_dir / "rowwise_alignment_key_mismatches.csv",
            index=False
        )

    return summary


def check_logical_duplicates(
    df: pd.DataFrame,
    key_columns: list,
    table_name: str,
    output_dir: Path,
    window_config_name: str,
    model_name: str,
    n_features: int
) -> dict:
    """
    Check whether a table contains duplicated logical windows.

    A logical window is identified by:
    subject_id, file_name, window_start_sample, window_end_sample.
    """

    missing_keys = [
        column for column in key_columns
        if column not in df.columns
    ]

    summary = {
        "window_config": window_config_name,
        "model": model_name,
        "best_n_features": n_features,
        "table_name": table_name,
        "n_rows": len(df),
        "required_key_columns": ";".join(key_columns),
        "missing_key_columns": ";".join(missing_keys),
        "duplicate_check_performed": False,
        "n_unique_logical_windows": np.nan,
        "n_duplicated_rows": np.nan,
        "n_duplicate_groups": np.nan,
        "max_rows_per_duplicate_group": np.nan,
    }

    if missing_keys:
        return summary

    df_with_key = df.copy()
    df_with_key["logical_window_key"] = make_logical_key(
        df_with_key,
        key_columns
    )

    duplicate_mask = df_with_key.duplicated(
        subset=["logical_window_key"],
        keep=False
    )

    duplicate_rows = df_with_key[duplicate_mask].copy()

    duplicate_groups = (
        df_with_key
        .groupby("logical_window_key", dropna=False)
        .size()
        .reset_index(name="n_rows_with_same_key")
    )

    duplicate_groups = duplicate_groups[
        duplicate_groups["n_rows_with_same_key"] > 1
    ].copy()

    summary["duplicate_check_performed"] = True
    summary["n_unique_logical_windows"] = int(
        df_with_key["logical_window_key"].nunique(dropna=False)
    )
    summary["n_duplicated_rows"] = int(len(duplicate_rows))
    summary["n_duplicate_groups"] = int(len(duplicate_groups))

    if duplicate_groups.empty:
        summary["max_rows_per_duplicate_group"] = 1
    else:
        summary["max_rows_per_duplicate_group"] = int(
            duplicate_groups["n_rows_with_same_key"].max()
        )

    safe_table_name = table_name.lower().replace(" ", "_")

    duplicate_rows.to_csv(
        output_dir / f"duplicate_rows_{safe_table_name}.csv",
        index=False
    )

    duplicate_groups.to_csv(
        output_dir / f"duplicate_groups_{safe_table_name}.csv",
        index=False
    )

    return summary


def export_ambiguous_window_cases(
    df: pd.DataFrame,
    output_dir: Path,
    window_config_name: str,
    model_name: str,
    n_features: int
) -> pd.DataFrame:
    """
    Export and summarize potentially ambiguous metadata-window cases.

    These cases are not automatically considered construction bugs.
    They identify windows where the binary GSD label and contextual metadata
    may appear inconsistent or not directly applicable.

    The main cases inspected here are:
    - true walking windows with path_type_name = none;
    - true walking windows with static_type_name = standing or sitting;
    - true static windows with path_type_name = straight or curved;
    - true static windows with static_type_name = none.

    The goal is to understand whether these windows are:
    - rare and physiologically explainable;
    - mostly transition/non-robust windows;
    - concentrated in specific tasks, subjects or files;
    - associated with many classification errors.
    """

    df = df.copy()

    path_type = normalize_text_column(df, "path_type_name")
    static_type = normalize_text_column(df, "static_type_name")
    transition_type = normalize_text_column(df, "transition_type")

    y_true = df["y_true"].astype(int)

    # Robustness is read from the derived flag if available.
    # Otherwise, it is reconstructed from the original metadata column.
    if "is_robust_window_bool" in df.columns:
        robust_mask = df["is_robust_window_bool"].fillna(False).astype(bool)
    else:
        robust_mask = parse_boolean_column(df, "is_robust_window")

    transition_mask = ~transition_type.isin(
        ["none", "nan", "missing", "", "no_transition"]
    )

    # ------------------------------------------------------------
    # Definition of ambiguous metadata cases
    # ------------------------------------------------------------
    # The first two cases are the most important for the current question.
    # The last two are symmetric checks useful for metadata debugging.

    cases = {
        "true_walking_with_path_type_none": (
            (y_true == 1) &
            path_type.isin(["none", "nan", "missing", ""])
        ),
        "true_walking_with_static_type_standing_or_sitting": (
            (y_true == 1) &
            (
                static_type.str.contains("standing", na=False) |
                static_type.str.contains("stand", na=False) |
                static_type.str.contains("sitting", na=False) |
                static_type.str.contains("sit", na=False)
            )
        ),
        "true_static_with_valid_path_type": (
            (y_true == 0) &
            (
                path_type.str.contains("straight", na=False) |
                path_type.str.contains("curved", na=False) |
                path_type.str.contains("turn", na=False)
            )
        ),
        "true_static_with_static_type_none": (
            (y_true == 0) &
            static_type.isin(["none", "nan", "missing", ""])
        ),
    }

    # Columns saved in the full and sample inspection files.
    # Missing columns are skipped automatically to keep the function robust
    # across datasets with slightly different metadata.
    context_columns = [
        "subject_id",
        "group",
        "task",
        "test_type",
        "challenge_type",
        "file_name",
        "file_path",
        "path_type_name",
        "static_type_name",
        "transition_type",
        "gait_phase_name",
        "activity_detail_name",
        "is_robust_window",
        "walking_static_ratio",
        "window_start_sample",
        "window_end_sample",
        "window_start_time",
        "window_end_time",
        "y_true",
        "y_pred",
        "y_true_name",
        "y_pred_name",
        "error_type",
        "is_correct",
    ]

    # Columns used to summarize where ambiguous cases come from.
    grouping_columns = [
        "task",
        "test_type",
        "path_type_name",
        "static_type_name",
        "transition_type",
        "gait_phase_name",
        "activity_detail_name",
        "is_robust_window",
        "error_type",
    ]

    summary_rows = []

    for case_name, mask in cases.items():

        case_df = df[mask].copy()

        n_case_windows = len(case_df)
        n_case_errors = int((~case_df["is_correct"]).sum()) if n_case_windows > 0 else 0

        n_case_false_positive = int(
            np.sum(case_df["error_type"] == "false_positive_static_as_walking")
        ) if n_case_windows > 0 else 0

        n_case_false_negative = int(
            np.sum(case_df["error_type"] == "false_negative_walking_as_static")
        ) if n_case_windows > 0 else 0

        n_case_transition = int(
            transition_mask.loc[case_df.index].sum()
        ) if n_case_windows > 0 else 0

        n_case_robust = int(
            robust_mask.loc[case_df.index].sum()
        ) if n_case_windows > 0 else 0

        n_case_non_robust = (
            n_case_windows - n_case_robust
            if n_case_windows > 0 else 0
        )

        n_case_subjects = (
            int(case_df["subject_id"].nunique(dropna=True))
            if n_case_windows > 0 and "subject_id" in case_df.columns
            else 0
        )

        n_case_files = (
            int(case_df["file_name"].nunique(dropna=True))
            if n_case_windows > 0 and "file_name" in case_df.columns
            else 0
        )

        summary_rows.append({
            "window_config": window_config_name,
            "model": model_name,
            "best_n_features": n_features,
            "case_name": case_name,
            "n_windows": n_case_windows,
            "pct_all_validation_windows": (
                n_case_windows / len(df) * 100.0
                if len(df) > 0 else np.nan
            ),
            "n_errors": n_case_errors,
            "error_rate_percent": (
                n_case_errors / n_case_windows * 100.0
                if n_case_windows > 0 else np.nan
            ),
            "n_false_positive_static_as_walking": n_case_false_positive,
            "n_false_negative_walking_as_static": n_case_false_negative,
            "n_transition_windows": n_case_transition,
            "pct_case_windows_transition": (
                n_case_transition / n_case_windows * 100.0
                if n_case_windows > 0 else np.nan
            ),
            "n_robust_windows": n_case_robust,
            "pct_case_windows_robust": (
                n_case_robust / n_case_windows * 100.0
                if n_case_windows > 0 else np.nan
            ),
            "n_non_robust_windows": n_case_non_robust,
            "pct_case_windows_non_robust": (
                n_case_non_robust / n_case_windows * 100.0
                if n_case_windows > 0 else np.nan
            ),
            "n_subjects": n_case_subjects,
            "n_files": n_case_files,
        })

        available_context_columns = [
            column for column in context_columns
            if column in case_df.columns
        ]

        # Save the complete list of ambiguous windows.
        case_df[available_context_columns].to_csv(
            output_dir / f"ambiguous_{case_name}_FULL.csv",
            index=False
        )

        if n_case_windows > 0:

            # Save a compact sample, prioritizing misclassified windows first.
            # This makes manual inspection faster, because the most problematic
            # ambiguous cases appear at the top of the sample file.
            case_errors = case_df[~case_df["is_correct"]].copy()
            case_correct = case_df[case_df["is_correct"]].copy()

            sample_df = pd.concat(
                [case_errors, case_correct],
                axis=0,
                ignore_index=False
            ).head(AMBIGUOUS_WINDOW_SAMPLE_SIZE)

            sample_df[available_context_columns].to_csv(
                output_dir / f"ambiguous_{case_name}_SAMPLE.csv",
                index=False
            )

            available_grouping_columns = [
                column for column in grouping_columns
                if column in case_df.columns
            ]

            if len(available_grouping_columns) > 0:

                context_summary = (
                    case_df
                    .groupby(available_grouping_columns, dropna=False)
                    .size()
                    .reset_index(name="n_windows")
                    .sort_values("n_windows", ascending=False)
                )

                context_summary.to_csv(
                    output_dir / f"ambiguous_{case_name}_context_summary.csv",
                    index=False
                )

            if "subject_id" in case_df.columns:

                subject_summary = (
                    case_df
                    .groupby(["subject_id"], dropna=False)
                    .agg(
                        n_windows=("error_type", "size"),
                        n_errors=("is_correct", lambda x: int((~x).sum())),
                    )
                    .reset_index()
                    .sort_values(["n_errors", "n_windows"], ascending=False)
                )

                subject_summary.to_csv(
                    output_dir / f"ambiguous_{case_name}_by_subject.csv",
                    index=False
                )

            if "file_name" in case_df.columns:

                file_summary = (
                    case_df
                    .groupby(["file_name"], dropna=False)
                    .agg(
                        n_windows=("error_type", "size"),
                        n_errors=("is_correct", lambda x: int((~x).sum())),
                    )
                    .reset_index()
                    .sort_values(["n_errors", "n_windows"], ascending=False)
                )

                file_summary.to_csv(
                    output_dir / f"ambiguous_{case_name}_by_file.csv",
                    index=False
                )

    return pd.DataFrame(summary_rows)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    global_summary_rows = []
    global_flag_summary_rows = []
    global_error_type_flag_rows = []
    global_metadata_summary_rows = []
    global_condition_performance_rows = []
    best_model_selection_rows = []

    global_rowwise_alignment_rows = []
    global_duplicate_window_rows = []
    global_ambiguous_window_rows = []

    global_robustness_class_rows = []
    global_transition_type_focus_rows = []
    global_clean_initiation_termination_rows = []
    global_label_fraction_bin_rows = []

    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print("\n[INFO] Global output directory:")
    print(GLOBAL_OUTPUT_DIR)

    for window_config_name in WINDOW_CONFIG_NAMES:

        print("\n============================================================")
        print(f"MISCLASSIFICATION ANALYSIS | {window_config_name}")
        print("============================================================")

        test_csv = get_test_csv_path(window_config_name)
        best_by_model_path = get_best_by_model_path(window_config_name)

        if not test_csv.exists():
            print(f"[WARNING] Internal validation CSV not found. Skipping: {test_csv}")
            continue

        if not best_by_model_path.exists():
            print(f"[WARNING] Best-by-model CSV not found. Skipping: {best_by_model_path}")
            continue

        test_df = pd.read_csv(test_csv, low_memory=False)
        best_by_model_df = pd.read_csv(best_by_model_path, low_memory=False)

        # Select only the best-performing model for this window configuration.
        # This is the key change compared with the previous version, which
        # analyzed all models independently.
        try:
            best_model_info = get_best_model_for_window_config(
                best_by_model_df=best_by_model_df,
                selection_metric="balanced_accuracy"
            )
        except ValueError as exc:
            print(f"[WARNING] {exc}")
            continue

        best_model_info["window_config"] = window_config_name
        best_model_selection_rows.append(best_model_info.copy())

        print(
            "[INFO] Best model selected for this window configuration: "
            f"{best_model_info['model']} | "
            f"top {best_model_info['n_features']} features | "
            f"balanced_accuracy = "
            f"{best_model_info['validation_balanced_accuracy']:.6f}"
        )

        # A one-element loop is used so the remaining code structure can stay
        # almost unchanged.
        for selected_model in [best_model_info]:

            model_name = selected_model["model"]
            n_features = selected_model["n_features"]

            prediction_csv = get_prediction_csv_path(
                window_config_name=window_config_name,
                model_name=model_name,
                n_features=n_features
            )

            if not prediction_csv.exists():
                print(f"[WARNING] Prediction CSV not found. Skipping: {prediction_csv}")
                continue

            output_dir = get_model_analysis_output_dir(
                window_config_name=window_config_name,
                model_name=model_name,
                n_features=n_features
            )

            text_report_path = (
                output_dir /
                f"{model_name}_top{n_features}_misclassification_analysis.txt"
            )

            with open(text_report_path, "w", encoding="utf-8") as report_file:

                log_message("\n------------------------------------------------------------", report_file)
                log_message(
                    f"[INFO] Window config: {window_config_name}",
                    report_file
                )
                log_message(f"[INFO] Model: {model_name}", report_file)
                log_message(f"[INFO] Best n_features: {n_features}", report_file)

                log_message("\n[INFO] Loading predictions:", report_file)
                log_message(prediction_csv, report_file)

                df = pd.read_csv(prediction_csv, low_memory=False)

                log_message("\n[DEBUG] Prediction/test consistency check:", report_file)
                log_message(f"[DEBUG] Predictions shape: {df.shape}", report_file)
                log_message(f"[DEBUG] Official test shape: {test_df.shape}", report_file)

                log_message("\n[DEBUG] Prediction y_true distribution:", report_file)
                log_message(df["y_true"].value_counts().sort_index(), report_file)

                log_message("\n[DEBUG] Official test label distribution:", report_file)
                log_message(test_df["label"].value_counts().sort_index(), report_file)

                if len(df) != len(test_df):
                    log_message(
                        "[WARNING] Prediction file and official test CSV have different number of rows.",
                        report_file
                    )
                else:
                    log_message(
                        "[SUCCESS] Prediction file and official test CSV have the same number of rows.",
                        report_file
                    )

                if "subject_id" in df.columns and "subject_id" in test_df.columns:
                    pred_subjects = set(df["subject_id"].dropna().unique())
                    test_subjects = set(test_df["subject_id"].dropna().unique())

                    log_message(
                        f"[DEBUG] Prediction subjects: {len(pred_subjects)}",
                        report_file
                    )
                    log_message(
                        f"[DEBUG] Official test subjects: {len(test_subjects)}",
                        report_file
                    )
                    log_message(
                        f"[DEBUG] Same subject set: {pred_subjects == test_subjects}",
                        report_file
                    )

                if "file_name" in df.columns and "file_name" in test_df.columns:
                    pred_files = set(df["file_name"].dropna().unique())
                    test_files = set(test_df["file_name"].dropna().unique())

                    log_message(
                        f"[DEBUG] Prediction files: {len(pred_files)}",
                        report_file
                    )
                    log_message(
                        f"[DEBUG] Official test files: {len(test_files)}",
                        report_file
                    )
                    log_message(
                        f"[DEBUG] Same file set: {pred_files == test_files}",
                        report_file
                    )

                # ------------------------------------------------------------
                # Row-wise alignment check
                # ------------------------------------------------------------
                # This verifies that the prediction CSV and the official
                # validation CSV contain the same logical windows in the same
                # exact order.

                rowwise_summary = check_rowwise_alignment(
                    prediction_df=df,
                    test_df=test_df,
                    key_columns=ROWWISE_KEY_COLUMNS,
                    output_dir=output_dir,
                    window_config_name=window_config_name,
                    model_name=model_name,
                    n_features=n_features
                )

                global_rowwise_alignment_rows.append(rowwise_summary)

                log_message("\n[DEBUG] Row-wise alignment summary:", report_file)
                log_message(pd.Series(rowwise_summary).to_string(), report_file)

                # ------------------------------------------------------------
                # Logical duplicate checks
                # ------------------------------------------------------------
                # These checks do not replace the dedicated dataset-wide
                # duplicate script, but they verify whether duplicated logical
                # windows are present in the prediction and validation tables
                # used by this analysis.

                prediction_duplicate_summary = check_logical_duplicates(
                    df=df,
                    key_columns=ROWWISE_KEY_COLUMNS,
                    table_name="prediction_csv",
                    output_dir=output_dir,
                    window_config_name=window_config_name,
                    model_name=model_name,
                    n_features=n_features
                )

                test_duplicate_summary = check_logical_duplicates(
                    df=test_df,
                    key_columns=ROWWISE_KEY_COLUMNS,
                    table_name="official_validation_csv",
                    output_dir=output_dir,
                    window_config_name=window_config_name,
                    model_name=model_name,
                    n_features=n_features
                )

                global_duplicate_window_rows.append(prediction_duplicate_summary)
                global_duplicate_window_rows.append(test_duplicate_summary)

                log_message("\n[DEBUG] Prediction duplicate summary:", report_file)
                log_message(
                    pd.Series(prediction_duplicate_summary).to_string(),
                    report_file
                )

                log_message("\n[DEBUG] Official validation duplicate summary:", report_file)
                log_message(
                    pd.Series(test_duplicate_summary).to_string(),
                    report_file
                )

                # ------------------------------------------------------------
                # Add error labels and window-type flags.
                # ------------------------------------------------------------

                df = add_error_columns(df)
                df = add_window_type_flags(df)

                # Add focused transition-analysis columns.
                # This creates:
                # - Static2Walking / Walking2Static flags;
                # - gait initiation / gait termination interpretation;
                # - clean-context masks excluding curved, sitting, stairs and
                #   none-dominated windows;
                # - final-label and valid-label fraction bins, if available.
                df = add_transition_focus_columns(
                    df=df,
                    window_config_name=window_config_name
                )

                df["window_config"] = window_config_name
                df["model"] = model_name
                df["best_n_features"] = n_features

                # ------------------------------------------------------------
                # Robust vs non-robust performance by true class
                # ------------------------------------------------------------

                robustness_class_performance = summarize_robustness_class_performance(
                    df=df
                )

                if not robustness_class_performance.empty:

                    robustness_class_performance.insert(
                        0,
                        "best_n_features",
                        n_features
                    )
                    robustness_class_performance.insert(
                        0,
                        "model",
                        model_name
                    )
                    robustness_class_performance.insert(
                        0,
                        "window_config",
                        window_config_name
                    )

                    robustness_class_performance.to_csv(
                        output_dir / "performance_robust_vs_non_robust_by_true_class.csv",
                        index=False
                    )

                    global_robustness_class_rows.append(
                        robustness_class_performance
                    )

                    log_message(
                        "\n[INFO] Robust vs non-robust performance by true class:",
                        report_file
                    )
                    log_message(
                        robustness_class_performance.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Transition-type performance
                # ------------------------------------------------------------
                # This table separates Static2Walking and Walking2Static without
                # applying the strict clean-context restriction.

                transition_type_focus_performance = summarize_transition_type_focus_performance(
                    df=df
                )

                if not transition_type_focus_performance.empty:

                    transition_type_focus_performance.insert(
                        0,
                        "best_n_features",
                        n_features
                    )
                    transition_type_focus_performance.insert(
                        0,
                        "model",
                        model_name
                    )
                    transition_type_focus_performance.insert(
                        0,
                        "window_config",
                        window_config_name
                    )

                    transition_type_focus_performance.to_csv(
                        output_dir / "performance_by_transition_phase_interpretation.csv",
                        index=False
                    )

                    global_transition_type_focus_rows.append(
                        transition_type_focus_performance
                    )

                    log_message(
                        "\n[INFO] Performance by transition phase interpretation:",
                        report_file
                    )
                    log_message(
                        transition_type_focus_performance.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Clean 1-second initiation/termination focus
                # ------------------------------------------------------------
                # This is the focused table requested to compare:
                # - robust static;
                # - robust walking;
                # - gait initiation;
                # - gait termination.
                #
                # It is generated only for 1-second windows and excludes curved,
                # sitting, stairs and none-dominated cases.

                clean_initiation_termination_performance = summarize_clean_initiation_termination_focus(
                    df=df,
                    window_config_name=window_config_name
                )

                if not clean_initiation_termination_performance.empty:

                    clean_initiation_termination_performance.insert(
                        0,
                        "best_n_features",
                        n_features
                    )
                    clean_initiation_termination_performance.insert(
                        0,
                        "model",
                        model_name
                    )
                    clean_initiation_termination_performance.insert(
                        0,
                        "window_config",
                        window_config_name
                    )

                    clean_initiation_termination_performance.to_csv(
                        output_dir / "performance_clean_1s_static_walking_initiation_termination.csv",
                        index=False
                    )

                    global_clean_initiation_termination_rows.append(
                        clean_initiation_termination_performance
                    )

                    log_message(
                        "\n[INFO] Clean 1s static/walking/initiation/termination performance:",
                        report_file
                    )
                    log_message(
                        clean_initiation_termination_performance.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Final-label fraction bin performance
                # ------------------------------------------------------------

                label_fraction_bin_performance = summarize_label_fraction_bin_performance(
                    df=df
                )

                if not label_fraction_bin_performance.empty:

                    label_fraction_bin_performance.insert(
                        0,
                        "condition_column",
                        "true_class_and_final_label_fraction_bin"
                    )
                    label_fraction_bin_performance.insert(
                        0,
                        "best_n_features",
                        n_features
                    )
                    label_fraction_bin_performance.insert(
                        0,
                        "model",
                        model_name
                    )
                    label_fraction_bin_performance.insert(
                        0,
                        "window_config",
                        window_config_name
                    )

                    label_fraction_bin_performance.to_csv(
                        output_dir / "performance_by_true_class_and_final_label_fraction_bin.csv",
                        index=False
                    )

                    global_label_fraction_bin_rows.append(
                        label_fraction_bin_performance
                    )

                    log_message(
                        "\n[INFO] Performance by true class and final-label fraction bin:",
                        report_file
                    )
                    log_message(
                        label_fraction_bin_performance.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Export and summarize potentially ambiguous metadata windows
                # ------------------------------------------------------------
                # These are windows where the binary GSD label and contextual
                # metadata may appear inconsistent or not directly applicable.
                #
                # Main examples:
                # - true walking windows with path_type_name = none;
                # - true walking windows with static_type_name = standing/sitting;
                # - true static windows with path_type_name = straight/curved;
                # - true static windows with static_type_name = none.
                #
                # These cases are not automatically considered bugs.
                # They are exported to support manual inspection before drawing
                # final conclusions for Paolo/reporting.

                ambiguous_summary = export_ambiguous_window_cases(
                    df=df,
                    output_dir=output_dir,
                    window_config_name=window_config_name,
                    model_name=model_name,
                    n_features=n_features
                )

                if not ambiguous_summary.empty:
                    ambiguous_summary.to_csv(
                        output_dir / "ambiguous_window_cases_summary.csv",
                        index=False
                    )

                    global_ambiguous_window_rows.append(ambiguous_summary)

                    log_message(
                        "\n[DEBUG] Ambiguous window cases summary:",
                        report_file
                    )
                    log_message(
                        ambiguous_summary.to_string(index=False),
                        report_file
                    )

                log_message(
                    f"\n[INFO] Prediction table shape after adding error columns and flags: {df.shape}",
                    report_file
                )

                # ------------------------------------------------------------
                # Global error summary.
                # ------------------------------------------------------------

                error_summary = summarize_error_types(df)

                log_message("\n[INFO] Error type distribution:", report_file)
                log_message(error_summary.to_string(index=False), report_file)

                error_summary.to_csv(
                    output_dir / "error_type_summary.csv",
                    index=False
                )

                n_windows = len(df)
                n_errors = int((~df["is_correct"]).sum())
                n_false_positive = int(
                    np.sum(df["error_type"] == "false_positive_static_as_walking")
                )
                n_false_negative = int(
                    np.sum(df["error_type"] == "false_negative_walking_as_static")
                )

                global_summary_rows.append({
                    "window_config": window_config_name,
                    "model": model_name,
                    "best_n_features": n_features,
                    "n_windows": n_windows,
                    "n_errors": n_errors,
                    "error_rate": n_errors / n_windows * 100.0 if n_windows > 0 else np.nan,
                    "n_false_positive_static_as_walking": n_false_positive,
                    "n_false_negative_walking_as_static": n_false_negative,
                    "false_positive_pct_of_errors": (
                        n_false_positive / n_errors * 100.0
                        if n_errors > 0 else 0.0
                    ),
                    "false_negative_pct_of_errors": (
                        n_false_negative / n_errors * 100.0
                        if n_errors > 0 else 0.0
                    ),
                })

                # ------------------------------------------------------------
                # Save full prediction table with flags.
                # ------------------------------------------------------------

                df.to_csv(
                    output_dir / "window_predictions_with_error_type_and_flags.csv",
                    index=False
                )

                false_positives = df[
                    df["error_type"] == "false_positive_static_as_walking"
                ].copy()

                false_negatives = df[
                    df["error_type"] == "false_negative_walking_as_static"
                ].copy()

                false_positives.to_csv(
                    output_dir / "false_positives_static_as_walking.csv",
                    index=False
                )

                false_negatives.to_csv(
                    output_dir / "false_negatives_walking_as_static.csv",
                    index=False
                )

                # ------------------------------------------------------------
                # Error composition by derived window-type flags.
                # ------------------------------------------------------------

                flag_summary = summarize_flag_error_composition(
                    df=df,
                    flag_columns=DERIVED_FLAG_COLUMNS
                )

                if not flag_summary.empty:

                    flag_summary.insert(0, "best_n_features", n_features)
                    flag_summary.insert(0, "model", model_name)
                    flag_summary.insert(0, "window_config", window_config_name)

                    flag_summary.to_csv(
                        output_dir / "error_composition_by_window_type_flags.csv",
                        index=False
                    )

                    global_flag_summary_rows.append(flag_summary)

                    log_message(
                        "\n[INFO] Error composition by window-type flags:",
                        report_file
                    )
                    log_message(
                        flag_summary.to_string(index=False),
                        report_file
                    )

                error_type_flag_summary = summarize_error_composition_by_error_type(
                    df=df,
                    flag_columns=DERIVED_FLAG_COLUMNS
                )

                if not error_type_flag_summary.empty:

                    error_type_flag_summary.insert(0, "best_n_features", n_features)
                    error_type_flag_summary.insert(0, "model", model_name)
                    error_type_flag_summary.insert(0, "window_config", window_config_name)

                    error_type_flag_summary.to_csv(
                        output_dir / "error_composition_by_error_type_and_flags.csv",
                        index=False
                    )

                    global_error_type_flag_rows.append(error_type_flag_summary)

                    log_message(
                        "\n[INFO] Error composition by error type and flags:",
                        report_file
                    )
                    log_message(
                        error_type_flag_summary.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Condition-specific performance summaries.
                # ------------------------------------------------------------
                # These tables answer the main interpretation question:
                # "For the best model of this window length, which window types
                # are more difficult to classify?"

                for col in CONDITION_COLUMNS_TO_ANALYZE:

                    performance_col = summarize_performance_by_column(
                        df=df,
                        column=col
                    )

                    if performance_col.empty:
                        log_message(
                            f"[WARNING] Performance column not found or empty: {col}",
                            report_file
                        )
                        continue

                    performance_col.insert(0, "condition_column", col)
                    performance_col.insert(0, "best_n_features", n_features)
                    performance_col.insert(0, "model", model_name)
                    performance_col.insert(0, "window_config", window_config_name)

                    performance_output_path = (
                        output_dir /
                        f"performance_by_{col}.csv"
                    )

                    performance_col.to_csv(
                        performance_output_path,
                        index=False
                    )

                    global_condition_performance_rows.append(performance_col)

                    log_message(
                        f"\n[INFO] Condition-specific performance by {col}:",
                        report_file
                    )

                    log_message(
                        performance_col.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Applicability-aware condition-specific performance
                # ------------------------------------------------------------
                # These analyses avoid interpreting "none" as a real condition
                # when the metadata field is simply not applicable.

                path_type_text = normalize_text_column(df, "path_type_name")
                static_type_text = normalize_text_column(df, "static_type_name")

                valid_path_mask = (
                    (df["y_true"].astype(int) == 1) &
                    (~path_type_text.isin(["none", "nan", "missing", ""]))
                )

                valid_static_type_mask = (
                    (df["y_true"].astype(int) == 0) &
                    (~static_type_text.isin(["none", "nan", "missing", ""]))
                )

                path_valid_performance = summarize_performance_by_column_with_filter(
                    df=df,
                    column="path_type_name",
                    applicability_mask=valid_path_mask,
                    analysis_name="path_type_on_true_walking_valid_path_only"
                )

                if not path_valid_performance.empty:
                    path_valid_performance.insert(0, "best_n_features", n_features)
                    path_valid_performance.insert(0, "model", model_name)
                    path_valid_performance.insert(0, "window_config", window_config_name)

                    path_valid_performance.to_csv(
                        output_dir / "performance_by_path_type_on_true_walking_valid_path_only.csv",
                        index=False
                    )

                    global_condition_performance_rows.append(path_valid_performance)

                    log_message(
                        "\n[INFO] Applicability-aware performance by path type "
                        "(true walking + valid path only):",
                        report_file
                    )
                    log_message(
                        path_valid_performance.to_string(index=False),
                        report_file
                    )

                static_valid_performance = summarize_performance_by_column_with_filter(
                    df=df,
                    column="static_type_name",
                    applicability_mask=valid_static_type_mask,
                    analysis_name="static_type_on_true_static_valid_static_type_only"
                )

                if not static_valid_performance.empty:
                    static_valid_performance.insert(0, "best_n_features", n_features)
                    static_valid_performance.insert(0, "model", model_name)
                    static_valid_performance.insert(0, "window_config", window_config_name)

                    static_valid_performance.to_csv(
                        output_dir / "performance_by_static_type_on_true_static_valid_static_type_only.csv",
                        index=False
                    )

                    global_condition_performance_rows.append(static_valid_performance)

                    log_message(
                        "\n[INFO] Applicability-aware performance by static type "
                        "(true static + valid static type only):",
                        report_file
                    )
                    log_message(
                        static_valid_performance.to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Metadata-based error summaries.
                # ------------------------------------------------------------

                columns_to_analyze = (
                    METADATA_COLUMNS_TO_ANALYZE +
                    DERIVED_FLAG_COLUMNS
                )

                for col in columns_to_analyze:

                    summary_col = summarize_by_column(df, col)

                    if summary_col.empty:

                        log_message(
                            f"[WARNING] Column not found or empty: {col}",
                            report_file
                        )

                        continue

                    summary_col.insert(0, "metadata_column", col)
                    summary_col.insert(0, "best_n_features", n_features)
                    summary_col.insert(0, "model", model_name)
                    summary_col.insert(0, "window_config", window_config_name)

                    output_path = output_dir / f"errors_by_{col}.csv"

                    summary_col.to_csv(output_path, index=False)

                    global_metadata_summary_rows.append(summary_col)

                    log_message(
                        f"\n[INFO] Top error patterns by {col}:",
                        report_file
                    )

                    log_message(
                        summary_col.head(10).to_string(index=False),
                        report_file
                    )

                # ------------------------------------------------------------
                # Subject-level summary.
                # ------------------------------------------------------------

                subject_summary = summarize_subjects(df)

                if not subject_summary.empty:

                    subject_summary.insert(0, "best_n_features", n_features)
                    subject_summary.insert(0, "model", model_name)
                    subject_summary.insert(0, "window_config", window_config_name)

                    subject_summary.to_csv(
                        output_dir / "errors_by_subject.csv",
                        index=False
                    )

                    global_metadata_summary_rows.append(
                        subject_summary.assign(metadata_column="subject_summary")
                    )

                    log_message(
                        "\n[INFO] Top subjects by number of errors:",
                        report_file
                    )

                    log_message(
                        subject_summary.head(20).to_string(index=False),
                        report_file
                    )

                log_message(
                    "\n[SUCCESS] Misclassification analysis completed for this model/configuration.",
                    report_file
                )

                log_message("[SUCCESS] Results saved to:", report_file)
                log_message(output_dir, report_file)

    # =====================================================
    # SAVE GLOBAL TABLES ACROSS ALL MODELS AND WINDOWS
    # =====================================================

    global_summary_df = pd.DataFrame(global_summary_rows)

    global_summary_path = (
        GLOBAL_OUTPUT_DIR /
        "BEST_MODELS_BY_WINDOW_global_error_summary.csv"
    )

    global_summary_df.to_csv(global_summary_path, index=False)

    if len(global_flag_summary_rows) > 0:
        global_flag_summary_df = pd.concat(
            global_flag_summary_rows,
            axis=0,
            ignore_index=True
        )

        global_flag_summary_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_error_composition_by_window_type_flags.csv",
            index=False
        )

    if len(global_error_type_flag_rows) > 0:
        global_error_type_flag_df = pd.concat(
            global_error_type_flag_rows,
            axis=0,
            ignore_index=True
        )

        global_error_type_flag_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_error_composition_by_error_type_and_flags.csv",
            index=False
        )

    if len(global_metadata_summary_rows) > 0:
        global_metadata_summary_df = pd.concat(
            global_metadata_summary_rows,
            axis=0,
            ignore_index=True
        )

        global_metadata_summary_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_metadata_error_summaries.csv",
            index=False
        )

    if len(best_model_selection_rows) > 0:
        best_model_selection_df = pd.DataFrame(best_model_selection_rows)

        best_model_selection_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODEL_SELECTED_FOR_EACH_WINDOW_CONFIG.csv",
            index=False
        )

    if len(global_condition_performance_rows) > 0:
        global_condition_performance_df = pd.concat(
            global_condition_performance_rows,
            axis=0,
            ignore_index=True
        )

        global_condition_performance_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_condition_specific_performance.csv",
            index=False
        )

    if len(global_rowwise_alignment_rows) > 0:
        global_rowwise_alignment_df = pd.DataFrame(
            global_rowwise_alignment_rows
        )

        global_rowwise_alignment_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_rowwise_alignment_summary.csv",
            index=False
        )

    if len(global_duplicate_window_rows) > 0:
        global_duplicate_window_df = pd.DataFrame(
            global_duplicate_window_rows
        )

        global_duplicate_window_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_duplicate_window_summary.csv",
            index=False
        )

    if len(global_ambiguous_window_rows) > 0:
        global_ambiguous_window_df = pd.concat(
            global_ambiguous_window_rows,
            axis=0,
            ignore_index=True
        )

        global_ambiguous_window_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_ambiguous_window_cases_summary.csv",
            index=False
        )

    if len(global_robustness_class_rows) > 0:
        global_robustness_class_df = pd.concat(
            global_robustness_class_rows,
            axis=0,
            ignore_index=True
        )

        global_robustness_class_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_performance_robust_vs_non_robust_by_true_class.csv",
            index=False
        )

    if len(global_transition_type_focus_rows) > 0:
        global_transition_type_focus_df = pd.concat(
            global_transition_type_focus_rows,
            axis=0,
            ignore_index=True
        )

        global_transition_type_focus_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_performance_by_transition_phase_interpretation.csv",
            index=False
        )

    if len(global_clean_initiation_termination_rows) > 0:
        global_clean_initiation_termination_df = pd.concat(
            global_clean_initiation_termination_rows,
            axis=0,
            ignore_index=True
        )

        global_clean_initiation_termination_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_performance_clean_1s_static_walking_initiation_termination.csv",
            index=False
        )

    if len(global_label_fraction_bin_rows) > 0:
        global_label_fraction_bin_df = pd.concat(
            global_label_fraction_bin_rows,
            axis=0,
            ignore_index=True
        )

        global_label_fraction_bin_df.to_csv(
            GLOBAL_OUTPUT_DIR /
            "BEST_MODELS_BY_WINDOW_performance_by_true_class_and_final_label_fraction_bin.csv",
            index=False
        )

    print("\n[SUCCESS] Global misclassification analysis completed.")
    print("[SUCCESS] Main summary saved to:")
    print(global_summary_path)
    print("[SUCCESS] Global output directory:")
    print(GLOBAL_OUTPUT_DIR)
