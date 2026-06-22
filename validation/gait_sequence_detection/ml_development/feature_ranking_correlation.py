"""
Generic correlation-based feature ranking for GSD datasets.

This script:
- loads the clustered training CSV of the selected dataset;
- optionally keeps only robust windows;
- computes a correlation-based feature ranking;
- saves the full ranking and the top-N feature list.
"""

# -----------------------------------------------------------------------------
# Import role
# -----------------------------------------------------------------------------
# This script only performs feature ranking; it does not train a classifier.
# The imported libraries are therefore limited to:
# - path management;
# - table manipulation;
# - numerical operations;
# - point-biserial correlation;
# - simple preprocessing needed only to make the correlation computation stable.

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =====================================================
# DATASET SELECTION
# =====================================================

# DATASET_NAME selects which already-built feature CSV must be ranked.
# The same script can be reused for the POLITO datasets by changing this value.
DATASET_NAME = "WearGaitPD"

# TOP_N defines how many features will be exported in the short text file used by
# downstream scripts. In the current workflow, the first 20 ranked features are
# passed to the bottom-up wrapper and to the final ML training scripts.
TOP_N = 20

# Window configurations to process.
# These names must match the feature CSVs produced by feature computation
# and the clustered train/internal-test split script.
WINDOW_CONFIG_NAMES = [
    "1s_50p_overlap",
    "2s_50p_overlap",
    "5s_50p_overlap",
    "10s_50p_overlap",
]

# Robust windows are clean, non-ambiguous windows according to the construction
# metadata. Ranking on robust windows reduces the risk that static/walking
# transition windows distort the feature-label association.
USE_ONLY_ROBUST_WINDOWS_FOR_RANKING = True

# Each dataset uses a different results folder and file-name prefix.
# This dictionary centralizes those conventions, so the rest of the code can be
# written once and then reused across datasets.
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


# Resolve the project root from the script location.
# This is safer than hard-coding a user-specific Windows path for source files.
PROJECT_ROOT = find_project_root(Path(__file__))

# Results are resolved from the project root.
# This avoids hard-coded user-specific paths and keeps the script portable
# across different Windows PCs.
LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"

# Retrieve the folder and file-name conventions associated with DATASET_NAME.
CONFIG = DATASET_CONFIGS[DATASET_NAME]

RESULTS_DIR = LOCAL_RESULTS_ROOT / CONFIG["results_folder"]
FEATURES_DIR = RESULTS_DIR / "features"
DATASET_SLUG = CONFIG["dataset_slug"]

def get_ranking_paths(window_config_name: str):
    """
    Build input and output paths for one window configuration.

    The ranking must be computed separately for each window length because
    feature distributions can change between 1 s, 2 s, 5 s and 10 s windows.
    """

    input_csv = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_features_{window_config_name}_clustered_train.csv"
    )

    output_dir = (
        FEATURES_DIR /
        "feature_ranking_correlation" /
        window_config_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    return input_csv, output_dir

# Binary GSD target column.
# Current convention: 0 = static/non-walking, 1 = walking.
TARGET_COL = "label"

# These columns describe windows, subjects, task context or labels.
# They must be excluded from the candidate feature pool because they are
# metadata, not input features extracted from the IMU signals.
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
    "activity_detail",
    "activity_detail_name",
    "group",
    "window_index",
    "dataset",
    "synthetic_source",
]


# =====================================================
# FEATURE SELECTION UTILITIES
# =====================================================

# -----------------------------------------------------------------------------
# Function role: candidate feature-column extraction
# -----------------------------------------------------------------------------
# The feature CSV contains both numerical features and metadata.
# This function keeps only columns that can be used as model input.

def get_numerical_feature_columns(df: pd.DataFrame) -> list:
    """
    Select numerical feature columns only.

    Metadata columns and label columns are excluded.
    """

    # Start from an empty list and scan all columns one by one.
    # A column is kept only if it is numerical and not listed as metadata.
    feature_cols = []

    for col in df.columns:
        # Skip known metadata columns even if they are numeric.
        # Example: label, window_start_sample, path_type.
        if col in METADATA_COLS:
            continue

        # Keep only numerical columns. Non-numerical columns cannot enter the
        # correlation computation or standard ML feature matrices directly.
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


# -----------------------------------------------------------------------------
# Function role: safe feature-label correlation score
# -----------------------------------------------------------------------------
# Point-biserial correlation measures the association between one continuous
# feature and a binary class label. Here it is used as a model-independent
# relevance score for each feature.

def safe_pointbiserial_correlation(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute absolute point-biserial correlation safely.

    Paolo's original function uses pointbiserialr(label, feature).
    Here the same idea is used, but NaN/inf cases are handled explicitly.

    If a feature is constant or invalid, its score is set to 0.
    """

    # A constant feature carries no discriminative information and may also make
    # the correlation computation unstable. Its score is therefore set to zero.
    if np.nanstd(x) < 1e-12:
        return 0.0

    # The correlation computation is wrapped in try/except because some features
    # can contain pathological values after imputation/scaling.
    try:
        coeff = pointbiserialr(y, x).correlation
    except Exception:
        return 0.0

    if coeff is None or np.isnan(coeff) or np.isinf(coeff):
        return 0.0

    return float(abs(coeff))


# -----------------------------------------------------------------------------
# Function role: feature-redundancy penalty
# -----------------------------------------------------------------------------
# A feature can correlate well with the target but still be redundant if it is
# very similar to many other features. This function estimates that redundancy.

def compute_redundancy_penalty(
    feature_name: str,
    feature_cols: list,
    corr_matrix: pd.DataFrame
) -> float:
    """
    Compute the mean absolute correlation between one feature and all other features.

    This approximates the feature-feature correlation penalty used in
    correlation-based feature selection.

    A feature is considered less useful if it is highly redundant with many
    other features.
    """

    # Compare the current feature only with the other candidate features.
    other_features = [col for col in feature_cols if col != feature_name]

    if len(other_features) == 0:
        return 0.0

    # Use absolute feature-feature correlation because both strong positive and
    # strong negative correlations indicate redundancy.
    redundancy_values = corr_matrix.loc[feature_name, other_features].abs()

    return float(redundancy_values.mean())


# -----------------------------------------------------------------------------
# Function role: correlation-based feature ranking
# -----------------------------------------------------------------------------
# This is a filter method, not a wrapper method.
# It scores each feature without training RF, KNN, SVM, LR or any other
# classifier. The ranking is based only on target association and redundancy.

def compute_correlation_based_ranking(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Compute a global correlation-based ranking.

    For each feature:
    - feature_class_corr measures association with the target
    - feature_feature_corr measures redundancy with the other features
    - ranking_score rewards target correlation and penalizes redundancy

    This is not a wrapper method.
    It is a filter/ranking step used before bottom-up wrapper selection.
    """

    # Separate the candidate feature matrix and the binary GSD target.
    # y is converted to integer because point-biserial correlation expects a
    # binary numerical target.
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).to_numpy()

    # Median imputation avoids NaN problems in correlation computation.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Standardization makes feature scales comparable.
    # Correlation itself is scale-invariant, but standardization is useful
    # for numerical stability and consistency with later ML pipelines.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_cols
    )

    # Feature-feature correlation matrix used to quantify redundancy.
    # This matrix does not involve the class label; it only compares features
    # with each other.
    corr_matrix = X_scaled_df.corr()

    # Each row of this list will contain the score components for one feature.
    rows = []

    # Score each candidate feature independently.
    for feature in feature_cols:

        # Association with the binary GSD label. Higher is better.
        feature_class_corr = safe_pointbiserial_correlation(
            y=y,
            x=X_scaled_df[feature].to_numpy()
        )

        # Mean absolute correlation with all other features. Higher means the
        # current feature is more redundant with the remaining feature set.
        feature_feature_corr = compute_redundancy_penalty(
            feature_name=feature,
            feature_cols=feature_cols,
            corr_matrix=corr_matrix
        )

        # Correlation-based score:
        # high feature-target correlation is good;
        # high feature-feature redundancy is penalized.
        ranking_score = feature_class_corr / np.sqrt(1.0 + feature_feature_corr)

        rows.append({
            "feature": feature,
            "feature_class_corr_abs": feature_class_corr,
            "mean_feature_feature_corr_abs": feature_feature_corr,
            "ranking_score": ranking_score,
        })

    # Convert the list of feature scores into a sortable table.
    ranking_df = pd.DataFrame(rows)

    # Sort by the final score first, then by feature-label correlation as a
    # tie-breaker. Rank 1 is the most useful feature according to this filter.
    ranking_df = ranking_df.sort_values(
        by=["ranking_score", "feature_class_corr_abs"],
        ascending=False
    ).reset_index(drop=True)

    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    return ranking_df


# =====================================================
# MAIN
# =====================================================

def process_one_window_config(window_config_name: str):
    """
    Run correlation-based feature ranking for one window configuration.
    """

    input_csv, output_dir = get_ranking_paths(window_config_name)

    print("\n============================================================")
    print(f"FEATURE RANKING: {DATASET_NAME} | {window_config_name}")
    print("============================================================")

    print("\n[INFO] Loading training feature table:")
    print(input_csv)

    if not input_csv.exists():
        print(f"[WARNING] Training CSV not found. Skipping: {input_csv}")
        return

    df = pd.read_csv(input_csv, low_memory=False)

    print("[INFO] Dataset shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in: {input_csv}"
        )

    print("\n[INFO] Label distribution before robust filtering:")
    print(df["label_name"].value_counts(dropna=False))

    if USE_ONLY_ROBUST_WINDOWS_FOR_RANKING:

        if "is_robust_window" not in df.columns:
            raise ValueError(
                "'is_robust_window' column not found. "
                "Run dataset construction and feature computation again."
            )

        print("\n[INFO] Robust-window distribution before ranking:")
        print(df["is_robust_window"].value_counts(dropna=False))

        # Robust-window values may be loaded either as booleans or as strings,
        # depending on how the CSV was written and read.
        robust_mask = (
            df["is_robust_window"].astype(str).str.lower().isin(["true", "1"])
        )

        df = df[robust_mask].copy()

        print("\n[INFO] Dataset shape after robust filtering:", df.shape)

        print("\n[INFO] Label distribution after robust filtering:")
        print(df["label_name"].value_counts(dropna=False))

    unique_labels = sorted(df[TARGET_COL].dropna().astype(int).unique())

    if len(unique_labels) < 2:
        raise ValueError(
            f"Ranking cannot be computed for {window_config_name}: "
            f"only one class is present after robust filtering. "
            f"Labels found: {unique_labels}"
        )

    feature_cols = get_numerical_feature_columns(df)

    print("\n[INFO] Numerical candidate features:", len(feature_cols))

    if len(feature_cols) == 0:
        raise ValueError(
            f"No numerical candidate feature columns found for {window_config_name}."
        )

    ranking_df = compute_correlation_based_ranking(
        df=df,
        feature_cols=feature_cols
    )

    # Store the window configuration explicitly for traceability.
    ranking_df.insert(1, "window_config", window_config_name)

    full_ranking_path = (
        output_dir /
        f"{DATASET_SLUG}_{window_config_name}_correlation_feature_ranking_full.csv"
    )

    top_path = (
        output_dir /
        f"{DATASET_SLUG}_{window_config_name}_correlation_feature_ranking_top{TOP_N}.csv"
    )

    top_list_path = (
        output_dir /
        f"{DATASET_SLUG}_{window_config_name}_top{TOP_N}_feature_list.txt"
    )

    ranking_df.to_csv(full_ranking_path, index=False)

    top_df = ranking_df.head(TOP_N).copy()
    top_df.to_csv(top_path, index=False)

    with open(top_list_path, "w", encoding="utf-8") as f:
        for feature in top_df["feature"].tolist():
            f.write(feature + "\n")

    print("\n[SUCCESS] Full ranking saved to:")
    print(full_ranking_path)

    print("\n[SUCCESS] Top ranking saved to:")
    print(top_path)

    print("\n[SUCCESS] Top feature list saved to:")
    print(top_list_path)

    print(f"\n===== TOP {TOP_N} FEATURES | {window_config_name} =====")
    print(
        top_df[
            [
                "rank",
                "feature",
                "ranking_score",
                "feature_class_corr_abs",
                "mean_feature_feature_corr_abs",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":

    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print("\n[INFO] Features directory:")
    print(FEATURES_DIR)

    for window_config_name in WINDOW_CONFIG_NAMES:
        process_one_window_config(window_config_name)