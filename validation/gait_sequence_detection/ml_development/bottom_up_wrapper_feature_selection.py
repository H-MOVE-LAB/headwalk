"""
Generic bottom-up wrapper feature selection for GSD datasets.

Pipeline:
1. Load the top-N correlation-based feature list.
2. Start from the first 2 ranked features.
3. Add 2 features at each step.
4. Tune model hyperparameters with Optuna at each step.
5. Train only on robust windows.
6. Validate on the complete validation set, including challenging windows.
7. Save performance timepoints and window-level predictions.
"""

# -----------------------------------------------------------------------------
# Import role
# -----------------------------------------------------------------------------
# This script performs a wrapper-style comparison of increasing feature subsets.
# It therefore needs:
# - JSON/path utilities for saving selected feature lists and parameters;
# - pandas/numpy for data handling;
# - Optuna for hyperparameter tuning;
# - sklearn models and metrics for classifier training and validation.

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# GaussianNB is added as a probabilistic Naive Bayes baseline for continuous
# handcrafted features. It is useful as a simple and fast model family to compare
# against distance-based, tree-based, margin-based and linear classifiers.
from sklearn.naive_bayes import GaussianNB

# Logistic Regression is added as a linear baseline.
# It is useful to check whether the selected features can separate
# static and walking windows with a simple linear decision boundary.
from sklearn.linear_model import LogisticRegression

# Support Vector Machine is added as a margin-based classifier.
# With an RBF kernel, it can model non-linear decision boundaries.
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =====================================================
# DATASET SELECTION
# =====================================================

# The wrapper is used only for model development on WearGaitPD.
# Training is performed on the WearGaitPD clustered training split.
# Internal validation is performed on the WearGaitPD clustered internal-test split.
TRAIN_DATASET_NAME = "WearGaitPD"

# The feature ranking step already saves the complete ranked list.
# The wrapper should not be limited to the top 20 anymore.
# If MAX_RANKED_FEATURES is None, all ranked features are used.
# If it is set to an integer, only the first N ranked features are used.
MAX_RANKED_FEATURES = None

# Window configurations to process.
# These names must match the feature CSVs, split CSVs and ranking outputs.
WINDOW_CONFIG_NAMES = [
    "1s_50p_overlap",
    "2s_50p_overlap",
    "5s_50p_overlap",
    "10s_50p_overlap",
]

# Training is optionally restricted to robust windows, while validation remains
# complete. This means the model learns from cleaner examples and is evaluated
# also on harder transition/ambiguous windows.
USE_ONLY_ROBUST_WINDOWS_FOR_TRAINING = True

# Dataset-specific folder and filename conventions.
# In this script, only the WearGaitPD configuration is used for training,
# internal validation, feature ranking and wrapper feature selection.
# The other dataset configurations should be used later by a separate
# external-validation script, not inside the bottom-up wrapper.
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


# Resolve the project root from the script position.
# This is useful for portability, although the results root below is still a
# manually specified local path.
PROJECT_ROOT = find_project_root(Path(__file__))

# Results are resolved from the project root.
# This avoids user-specific hard-coded paths.
LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"

# Main training configuration.
# From this point on, FEATURES_DIR and DATASET_SLUG always refer to WearGaitPD,
# because WearGaitPD is the only dataset used for training, internal validation,
# feature ranking and wrapper feature selection.
TRAIN_CONFIG = DATASET_CONFIGS[TRAIN_DATASET_NAME]

RESULTS_DIR = LOCAL_RESULTS_ROOT / TRAIN_CONFIG["results_folder"]
FEATURES_DIR = RESULTS_DIR / "features"
DATASET_SLUG = TRAIN_CONFIG["dataset_slug"]

def get_wrapper_paths(window_config_name: str):
    """
    Build all input/output paths for one window configuration.
    """

    train_csv = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_features_{window_config_name}_clustered_train.csv"
    )

    val_csv = (
        FEATURES_DIR /
        f"{DATASET_SLUG}_features_{window_config_name}_clustered_internal_test.csv"
    )

    full_ranking_csv = (
        FEATURES_DIR /
        "feature_ranking_correlation" /
        window_config_name /
        f"{DATASET_SLUG}_{window_config_name}_correlation_feature_ranking_full.csv"
    )

    output_dir = (
        FEATURES_DIR /
        "bottomup_wrapper_feature_selection" /
        window_config_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    return train_csv, val_csv, full_ranking_csv, output_dir

# =====================================================
# CONFIGURATION
# =====================================================

# Reproducibility seed used by Optuna and Random Forest.
RANDOM_STATE = 42

# Number of Optuna trials per model and per feature-count timepoint.
# Current value is intentionally small to keep the search computationally light.
N_OPTUNA_TRIALS = 10

# Binary GSD target column.
TARGET_COL = "label"

# Bottom-up schedule.
# The wrapper tests: top 2, top 4, top 6, ..., MAX_RANKED_FEATURES (max = 365, then 364) top ranked features.
# The wrapper starts from MIN_FEATURES ranked features and then adds
# FEATURE_STEP features at each iteration.
# Early stopping criterion:
# if balanced accuracy improves by less than 0.005 compared with the previous
# iteration, the search for the current model is stopped.
# 0.005 corresponds to 0.5 percentage points because balanced accuracy is stored
# on a normalized 0-1 scale.
MIN_FEATURES = 2
FEATURE_STEP = 2
MIN_BALANCED_ACCURACY_IMPROVEMENT = 0.005

LABEL_ORDER = [0, 1]
LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}

# Models currently included in the wrapper search.
# Each model is processed independently.
# Therefore, each classifier can obtain its own optimal feature subset.
MODELS_TO_RUN = [
    "knn",
    "rf",
    "svm",
    "lr",
    "gnb"
]

# Metadata columns are listed for consistency with the feature-generation
# pipeline. In this specific wrapper, the selected input features are loaded
# from FULL_RANKING_CSV, so metadata columns are not directly used to build X.
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

    "synthetic_source",
]


# =====================================================
# DATA LOADING
# =====================================================

# -----------------------------------------------------------------------------
# Function role: load train/validation feature tables
# -----------------------------------------------------------------------------
# This function reads the feature-level CSV files and optionally removes
# non-robust windows only from training. Validation remains unchanged by design.

def load_feature_tables(train_csv, val_csv):
    """
    Load train and validation feature tables.

    Training can optionally be restricted to robust windows only.
    Validation is always kept complete, including ambiguous transition windows.
    """

    # Load already-computed feature tables.
    # These CSVs are assumed to contain one row per window and one column per
    # handcrafted feature or metadata field.
    df_train = pd.read_csv(train_csv, low_memory=False)
    df_val = pd.read_csv(val_csv, low_memory=False)

    # Add dataset identifiers if missing, so later outputs remain traceable even
    # when the original CSV does not contain an explicit dataset column.
    if "dataset" not in df_train.columns:
        df_train = df_train.assign(dataset=TRAIN_DATASET_NAME)

    if "dataset" not in df_val.columns:
        df_val = df_val.assign(dataset=TRAIN_DATASET_NAME)

    print("[INFO] Original train shape:", df_train.shape)
    print("[INFO] Validation shape:", df_val.shape)

    print("\n[INFO] Original train label distribution:")
    print(df_train["label_name"].value_counts())

    print("\n[INFO] Validation label distribution:")
    print(df_val["label_name"].value_counts())

    if USE_ONLY_ROBUST_WINDOWS_FOR_TRAINING:

        if "is_robust_window" not in df_train.columns:
            raise ValueError(
                "'is_robust_window' column not found in training CSV. "
                "Run construction and feature computation again."
            )

        print("\n[INFO] Robust-window filtering on training set")

        print("\n[INFO] Train robust-window distribution:")
        print(df_train["is_robust_window"].value_counts(dropna=False))

        # Train only on robust windows.
        # This does not remove ambiguous windows from validation, so performance
        # still reflects behavior on difficult samples.
        robust_mask = (
            df_train["is_robust_window"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )

        df_train = df_train[robust_mask].copy()
        print("\n[INFO] Train shape after robust filtering:", df_train.shape)

        print("\n[INFO] Train label distribution after robust filtering:")
        print(df_train["label_name"].value_counts())

    print("\n[INFO] Validation set is kept unchanged.")
    print("[INFO] Final validation shape:", df_val.shape)

    return df_train, df_val

# -----------------------------------------------------------------------------
# Function role: load ordered top-N feature list
# -----------------------------------------------------------------------------
# This list comes from the correlation-based filter ranking. The wrapper does not
# reorder features; it tests increasingly long prefixes of this fixed order.

def load_ranked_features(full_ranking_csv):
    """
    Load the complete correlation-based feature ranking.

    The previous implementation loaded only the top-20 feature list.
    This was too restrictive because the wrapper could only evaluate subsets
    inside the first 20 ranked features.

    The new logic loads the full ranking CSV produced by
    feature_ranking_correlation.py. This allows the bottom-up wrapper to use all
    available ranked features as candidate features.

    The ranking itself is still model-agnostic:
    it is based on feature-target correlation and feature-feature redundancy.
    The model-specific part is introduced later, because each classifier runs
    its own wrapper loop and stops at its own optimal number of features.
    """

    if not full_ranking_csv.exists():
        raise FileNotFoundError(
            "Full correlation-based feature ranking not found.\n"
            f"Expected file: {full_ranking_csv}\n"
            "Run feature_ranking_correlation.py before the wrapper selection."
        )

    ranking_df = pd.read_csv(full_ranking_csv)

    if "feature" not in ranking_df.columns:
        raise ValueError(
            "The full ranking CSV must contain a column named 'feature'."
        )

    # If the ranking file contains an explicit rank column, sort by it to make
    # sure the wrapper follows the intended feature order.
    if "rank" in ranking_df.columns:
        ranking_df = ranking_df.sort_values("rank").reset_index(drop=True)

    ranked_features = ranking_df["feature"].dropna().astype(str).tolist()

    if MAX_RANKED_FEATURES is not None:
        ranked_features = ranked_features[:MAX_RANKED_FEATURES]
        ranking_df = ranking_df[ranking_df["feature"].isin(ranked_features)].copy()

    print(f"\n[INFO] Loaded ranked feature pool: {len(ranked_features)} features")

    # Print only a short preview of the ranked feature pool.
    # This does not limit the wrapper to 20 features:
    # the complete ranked_features list is still used to build the bottom-up
    # feature-count schedule.
    n_preview_features = min(20, len(ranked_features))

    print(f"\n[INFO] Preview of the first {n_preview_features} ranked features:")
    for i, feature in enumerate(ranked_features[:n_preview_features], start=1):
        print(f"{i:02d}. {feature}")

    if len(ranked_features) > n_preview_features:
        print(
            f"[INFO] Only the first {n_preview_features} features are printed here. "
            f"The wrapper will use the full ranked pool of {len(ranked_features)} features."
        )

    return ranked_features, ranking_df


# -----------------------------------------------------------------------------
# Function role: selected-feature availability check
# -----------------------------------------------------------------------------
# This prevents silent errors where the top-feature list contains a feature name
# that is absent from train or validation CSVs.

def validate_feature_presence(df_train, df_val, ranked_features):
    """
    Check that all selected ranked features exist in both train and validation CSVs.
    """

    missing_train = [f for f in ranked_features if f not in df_train.columns]
    missing_val = [f for f in ranked_features if f not in df_val.columns]

    if len(missing_train) > 0 or len(missing_val) > 0:
        raise ValueError(
            "Some ranked features are missing from the feature tables.\n"
            f"Missing in train: {missing_train}\n"
            f"Missing in validation: {missing_val}"
        )

def build_feature_counts(n_ranked_features):
    """
    Build the list of feature-subset sizes to evaluate.

    Example:
    if MIN_FEATURES = 2 and FEATURE_STEP = 2, the wrapper evaluates:
    2, 4, 6, 8, ...

    If the total number of ranked features is odd, the last value is also added
    explicitly, so the full candidate pool can still be evaluated if early
    stopping does not stop the loop earlier.
    """

    if n_ranked_features < MIN_FEATURES:
        raise ValueError(
            f"Not enough ranked features. Found {n_ranked_features}, "
            f"but MIN_FEATURES is {MIN_FEATURES}."
        )

    feature_counts = list(
        range(
            MIN_FEATURES,
            n_ranked_features + 1,
            FEATURE_STEP
        )
    )

    if feature_counts[-1] != n_ranked_features:
        feature_counts.append(n_ranked_features)

    return feature_counts


def save_selected_feature_table(
    ranking_df,
    selected_features,
    output_path,
    model_name,
    n_features,
    balanced_accuracy
):
    """
    Save the selected feature subset in a readable tabular format.

    This table is useful because the JSON feature list inside the performance
    file is compact but not very readable.

    The output contains:
    - model name;
    - selected feature order inside the current subset;
    - original correlation-based rank;
    - feature name;
    - correlation-ranking values, if available;
    - balanced accuracy obtained with this subset.
    """

    ranking_lookup = ranking_df.set_index("feature", drop=False)

    rows = []

    for selected_order, feature_name in enumerate(selected_features, start=1):

        if feature_name in ranking_lookup.index:
            ranking_row = ranking_lookup.loc[feature_name].to_dict()
        else:
            ranking_row = {
                "feature": feature_name,
                "rank": None,
                "ranking_score": None,
                "feature_class_corr_abs": None,
                "mean_feature_feature_corr_abs": None,
            }

        rows.append({
            "model": model_name,
            "n_features": n_features,
            "selected_order": selected_order,
            "feature": feature_name,
            "original_rank": ranking_row.get("rank", None),
            "ranking_score": ranking_row.get("ranking_score", None),
            "feature_class_corr_abs": ranking_row.get("feature_class_corr_abs", None),
            "mean_feature_feature_corr_abs": ranking_row.get(
                "mean_feature_feature_corr_abs",
                None
            ),
            "balanced_accuracy": balanced_accuracy,
        })

    selected_feature_df = pd.DataFrame(rows)
    selected_feature_df.to_csv(output_path, index=False)

    print("[SUCCESS] Selected feature table saved to:")
    print(output_path)

# =====================================================
# MODEL BUILDERS
# =====================================================

# -----------------------------------------------------------------------------
# Function role: KNN model builder for Optuna
# -----------------------------------------------------------------------------
# Optuna calls this function repeatedly. Each trial proposes one KNN
# hyperparameter configuration and receives a full sklearn Pipeline.

def build_knn_model(trial):
    """
    Build a KNN model with Optuna-suggested hyperparameters.

    KNN is distance-based, so StandardScaler is included in the pipeline.
    """

    # Hyperparameters suggested by Optuna for the current trial.
    # n_neighbors controls local neighborhood size.
    n_neighbors = trial.suggest_int("n_neighbors", 3, 25)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)

    # Pipeline structure:
    # 1. median imputation for missing feature values;
    # 2. standardization because KNN is distance-based;
    # 3. KNN classifier.
    # Random Forest does not require scaling because it is tree-based.
    # The pipeline still includes median imputation for missing feature values.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p
        ))
    ])

    return model


# -----------------------------------------------------------------------------
# Function role: Random Forest model builder for Optuna
# -----------------------------------------------------------------------------
# This confirms that RF is explicitly used inside the bottom-up wrapper.
# Optuna tunes RF hyperparameters for each tested feature subset.

def build_rf_model(trial):
    """
    Build a Random Forest model with Optuna-suggested hyperparameters.

    Random Forest does not require feature scaling.
    """

    # RF hyperparameters suggested by Optuna.
    # The search is deliberately limited to a small grid to keep the wrapper fast.
    n_estimators = trial.suggest_categorical("n_estimators", [100, 200, 300])
    max_depth = trial.suggest_categorical("max_depth", [None, 5, 10, 20])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return model

def build_svm_model(trial):
    """
    Build an SVM model with Optuna-suggested hyperparameters.

    SVM is sensitive to feature scaling, so StandardScaler is included.
    The RBF kernel is used because it can model non-linear boundaries between
    static and walking windows.
    """

    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    return model

# -----------------------------------------------------------------------------
# Function role: Gaussian Naive Bayes model builder for Optuna
# -----------------------------------------------------------------------------
# Optuna tunes the variance-smoothing term and whether class priors should be
# estimated from the training distribution or fixed as balanced priors.

def build_gnb_model(trial):
    """
    Build a Gaussian Naive Bayes model with Optuna-suggested hyperparameters.

    GaussianNB is suitable for the current feature table because the inputs are
    continuous handcrafted IMU features.

    StandardScaler is included to apply the same Z-score normalization used in
    the final ML training workflow. It is not strictly required by GaussianNB,
    but it improves numerical consistency when feature magnitudes are very
    different and makes var_smoothing easier to interpret across features.
    """

    # var_smoothing adds a small stability term to the estimated variances.
    # Searching it on a log scale keeps the default value 1e-9 inside the tested
    # range while also allowing slightly stronger or weaker smoothing.
    var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)

    # GaussianNB does not implement class_weight. For the binary GSD task, balanced
    # priors can be tested as the closest simple equivalent to avoid favoring the
    # majority class only because it is more frequent in the training split.
    prior_mode = trial.suggest_categorical("prior_mode", ["empirical", "balanced"])

    if prior_mode == "balanced":
        priors = [0.5, 0.5]
    else:
        priors = None

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB(
            priors=priors,
            var_smoothing=var_smoothing
        ))
    ])

    return model


def build_lr_model(trial):
    """
    Build a Logistic Regression model with Optuna-suggested hyperparameters.

    Logistic Regression is a linear baseline. It is useful to check whether the
    selected features are already sufficient to separate the two GSD classes
    with a linear decision boundary.

    Logistic Regression is sensitive to feature scaling, so StandardScaler is
    included in the pipeline.
    """

    C = trial.suggest_float("C", 1e-3, 1e2, log=True)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])

    return model


# -----------------------------------------------------------------------------
# Function role: model-family dispatcher
# -----------------------------------------------------------------------------
# The wrapper loop uses model_name strings. This function maps each string to
# the corresponding Optuna-aware builder.

def build_model(model_name, trial):
    """
    Select the requested model family.

    Each model has its own pipeline and its own Optuna hyperparameter space.
    This is important because the wrapper feature selection is now model-specific:
    RF, KNN, SVM and LR are allowed to stop at different feature subsets.
    """

    if model_name == "knn":
        return build_knn_model(trial)

    if model_name == "rf":
        return build_rf_model(trial)

    if model_name == "svm":
        return build_svm_model(trial)

    if model_name == "lr":
        return build_lr_model(trial)

    if model_name == "gnb":
        return build_gnb_model(trial)

    raise ValueError(f"Unknown model name: {model_name}")


# =====================================================
# EVALUATION
# =====================================================

# -----------------------------------------------------------------------------
# Function role: validation metric computation
# -----------------------------------------------------------------------------
# All wrapper timepoints are summarized with the same metric set. Balanced
# accuracy is the optimization criterion because it compensates class imbalance.

def compute_metrics(y_true, y_pred):
    """
    Compute the key GSD metrics.

    Balanced accuracy is the main metric requested by Paolo.
    """

    recalls = recall_score(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average=None,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_static": recalls[0],
        "recall_walking": recalls[1],
    }

    return metrics


# -----------------------------------------------------------------------------
# Function role: Optuna tuning for one model and one feature subset
# -----------------------------------------------------------------------------
# This is the true wrapper step: a classifier is trained and validated for the
# current selected feature subset, and its validation balanced accuracy is used
# to compare configurations.

def tune_model_with_optuna(model_name, X_train, y_train, X_val, y_val):
    """
    Tune a model using Optuna on the current feature subset.

    The objective is validation balanced accuracy.
    """

    def objective(trial):
        # Build one model instance using the hyperparameters suggested for this
        # Optuna trial.
        model = build_model(model_name, trial)

        # Fit on the current training set and current feature subset.
        model.fit(X_train, y_train)

        # Predict on the complete validation set.
        y_pred = model.predict(X_val)

        # Optuna maximizes validation balanced accuracy.
        return balanced_accuracy_score(y_val, y_pred)

    # Create a new Optuna study for this specific model/feature-count pair.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False
    )

    # Rebuild and refit the best model configuration returned by Optuna.
    best_model = build_model(model_name, study.best_trial)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)

    return best_model, study.best_params, study.best_value, metrics, y_pred


# =====================================================
# MAIN BOTTOM-UP LOOP
# =====================================================

def run_wrapper_for_window_config(window_config_name: str):
    """
    Run the complete bottom-up wrapper for one window configuration.
    """

    train_csv, val_csv, full_ranking_csv, output_dir = get_wrapper_paths(
        window_config_name
    )

    print("\n============================================================")
    print(f"BOTTOM-UP WRAPPER: {TRAIN_DATASET_NAME} | {window_config_name}")
    print("============================================================")

    if not train_csv.exists():
        print(f"[WARNING] Training CSV not found. Skipping: {train_csv}")
        return

    if not val_csv.exists():
        print(f"[WARNING] Validation CSV not found. Skipping: {val_csv}")
        return

    if not full_ranking_csv.exists():
        print(f"[WARNING] Ranking CSV not found. Skipping: {full_ranking_csv}")
        return

    df_train, df_val = load_feature_tables(
        train_csv=train_csv,
        val_csv=val_csv
    )

    print("\n[DEBUG] Train unique labels:")
    print(df_train["label"].value_counts().sort_index())

    print("\n[DEBUG] Validation unique labels:")
    print(df_val["label"].value_counts().sort_index())

    print("\n[DEBUG] Train subjects:", df_train["subject_id"].nunique())
    print("[DEBUG] Validation subjects:", df_val["subject_id"].nunique())

    shared_subjects = set(df_train["subject_id"]).intersection(
        set(df_val["subject_id"])
    )

    print("[DEBUG] Shared subjects train/validation:", len(shared_subjects))

    if len(shared_subjects) > 0:
        print("[WARNING] Shared subjects:")
        print(sorted(shared_subjects))

    # ------------------------------------------------------------
    # Load complete ranked feature pool
    # ------------------------------------------------------------
    # The wrapper now uses the full ranking produced by the correlation-based
    # ranking script, not only the fixed top-20 list.
    ranked_features, ranking_df = load_ranked_features(
        full_ranking_csv=full_ranking_csv
    )

    validate_feature_presence(
        df_train=df_train,
        df_val=df_val,
        ranked_features=ranked_features
    )

    y_train = df_train[TARGET_COL].astype(int)
    y_val = df_val[TARGET_COL].astype(int)

    # Build the feature-subset sizes to evaluate.
    # Example: 2, 4, 6, ... up to the complete ranked feature pool.
    feature_counts = build_feature_counts(
        n_ranked_features=len(ranked_features)
    )

    print("\n[INFO] Feature-subset sizes to evaluate:")
    print(feature_counts)

    all_results = []
    best_rows = []

    for model_name in MODELS_TO_RUN:

        print(f"\n==============================")
        print(f"MODEL: {model_name.upper()}")
        print(f"==============================")

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_results = []
        previous_balanced_accuracy = None

        for n_features in feature_counts:

            selected_features = ranked_features[:n_features]

            print("\n--------------------------------")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] Number of features: {n_features}")
            print("[INFO] Selected features:")
            for feature in selected_features:
                print(" -", feature)

            X_train = df_train[selected_features].copy()
            X_val = df_val[selected_features].copy()

            best_model, best_params, best_value, metrics, y_pred = tune_model_with_optuna(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )

            current_balanced_accuracy = metrics["balanced_accuracy"]

            # ------------------------------------------------------------
            # Early stopping check
            # ------------------------------------------------------------
            # The first iteration has no previous balanced accuracy.
            # From the second iteration onward, the current subset is compared
            # with the immediately previous subset.
            #
            # If the improvement is smaller than 0.005, the search for the
            # current model is stopped after saving the current timepoint.
            if previous_balanced_accuracy is None:
                balanced_accuracy_delta = None
                early_stop_triggered = False
            else:
                balanced_accuracy_delta = (
                    current_balanced_accuracy
                    -
                    previous_balanced_accuracy
                )

                early_stop_triggered = (
                    balanced_accuracy_delta
                    < MIN_BALANCED_ACCURACY_IMPROVEMENT
                )

            row = {
                "window_config": window_config_name,
                "model": model_name,
                "n_features": n_features,
                "candidate_pool_size": len(ranked_features),
                "features": json.dumps(selected_features),
                "best_optuna_value": best_value,
                "best_params": json.dumps(best_params),
                "previous_balanced_accuracy": previous_balanced_accuracy,
                "balanced_accuracy_delta_from_previous": balanced_accuracy_delta,
                "early_stop_min_improvement": MIN_BALANCED_ACCURACY_IMPROVEMENT,
                "early_stop_triggered": early_stop_triggered,
                **metrics,
            }

            all_results.append(row)
            model_results.append(row)

            print("[RESULT] Balanced accuracy:", metrics["balanced_accuracy"])
            print("[RESULT] Accuracy:", metrics["accuracy"])
            print("[RESULT] Recall static:", metrics["recall_static"])
            print("[RESULT] Recall walking:", metrics["recall_walking"])
            print("[RESULT] Best params:", best_params)

            if balanced_accuracy_delta is not None:
                print(
                    "[RESULT] Balanced accuracy delta from previous subset:",
                    balanced_accuracy_delta
                )

            # ------------------------------------------------------------
            # Save model checkpoint for this timepoint
            # ------------------------------------------------------------
            # The model is saved even if early stopping is triggered, because
            # the current point is still a documented evaluated configuration.
            model_path = model_output_dir / f"{model_name}_top{n_features}_features.joblib"
            joblib.dump(best_model, model_path)

            # ------------------------------------------------------------
            # Save selected feature table for this timepoint
            # ------------------------------------------------------------
            selected_features_path = (
                model_output_dir /
                f"{model_name}_top{n_features}_selected_features.csv"
            )

            save_selected_feature_table(
                ranking_df=ranking_df,
                selected_features=selected_features,
                output_path=selected_features_path,
                model_name=model_name,
                n_features=n_features,
                balanced_accuracy=current_balanced_accuracy
            )

            # ------------------------------------------------------------
            # Save confusion matrix for this timepoint
            # ------------------------------------------------------------
            cm = confusion_matrix(y_val, y_pred, labels=LABEL_ORDER)

            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{LABEL_NAME_MAP[x]}" for x in LABEL_ORDER],
                columns=[f"pred_{LABEL_NAME_MAP[x]}" for x in LABEL_ORDER]
            )

            cm_path = model_output_dir / f"{model_name}_top{n_features}_confusion_matrix.csv"
            cm_df.to_csv(cm_path)

            # ------------------------------------------------------------
            # Save classification report for this timepoint
            # ------------------------------------------------------------
            report = classification_report(
                y_val,
                y_pred,
                labels=LABEL_ORDER,
                target_names=[LABEL_NAME_MAP[x] for x in LABEL_ORDER],
                zero_division=0
            )

            report_path = model_output_dir / f"{model_name}_top{n_features}_classification_report.txt"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            # ------------------------------------------------------------
            # Save window-level predictions for misclassification analysis
            # ------------------------------------------------------------
            # This table preserves all validation-window metadata and adds the
            # true/predicted labels for the current model and feature subset.
            prediction_df = df_val.copy()

            prediction_df["y_true"] = np.asarray(y_val).astype(int)
            prediction_df["y_pred"] = np.asarray(y_pred).astype(int)

            prediction_df["y_true_name"] = prediction_df["y_true"].map(LABEL_NAME_MAP)
            prediction_df["y_pred_name"] = prediction_df["y_pred"].map(LABEL_NAME_MAP)

            prediction_df["model"] = model_name
            prediction_df["window_config"] = window_config_name
            prediction_df["n_features"] = n_features
            prediction_df["selected_features"] = json.dumps(selected_features)
            prediction_df["best_params"] = json.dumps(best_params)
            prediction_df["balanced_accuracy"] = metrics["balanced_accuracy"]
            prediction_df["accuracy"] = metrics["accuracy"]
            prediction_df["recall_static"] = metrics["recall_static"]
            prediction_df["recall_walking"] = metrics["recall_walking"]
            prediction_df["balanced_accuracy_delta_from_previous"] = balanced_accuracy_delta
            prediction_df["early_stop_triggered"] = early_stop_triggered

            prediction_path = (
                model_output_dir /
                f"{model_name}_top{n_features}_window_predictions.csv"
            )

            prediction_df.to_csv(prediction_path, index=False)

            print("[SUCCESS] Window-level predictions saved to:")
            print(prediction_path)
            print("[DEBUG] Prediction dataframe shape:", prediction_df.shape)

            # ------------------------------------------------------------
            # Save cumulative performance after each evaluated subset
            # ------------------------------------------------------------
            results_df = pd.DataFrame(all_results)
            results_path = output_dir / "bottomup_wrapper_performance_timepoints.csv"
            results_df.to_csv(results_path, index=False)

            # Stop after saving the current evaluated point.
            if early_stop_triggered:
                print(
                    "\n[EARLY STOP] Stopping feature search for model:",
                    model_name
                )
                print(
                    "[EARLY STOP] Improvement was smaller than:",
                    MIN_BALANCED_ACCURACY_IMPROVEMENT
                )
                break

            previous_balanced_accuracy = current_balanced_accuracy

        # ------------------------------------------------------------
        # Save best feature subset for the current model
        # ------------------------------------------------------------
        # The best subset is selected using validation balanced accuracy among
        # all evaluated timepoints for this model.
        model_results_df = pd.DataFrame(model_results)

        best_model_row = model_results_df.sort_values(
            by="balanced_accuracy",
            ascending=False
        ).iloc[0]

        # The best row is the configuration with the highest validation balanced
        # accuracy. It is not necessarily the row that triggered early stopping.
        best_row_dict = best_model_row.to_dict()

        # The last evaluated row describes where the search actually stopped.
        last_evaluated_row = model_results_df.iloc[-1]

        search_stopped_by_early_stop = bool(
            model_results_df["early_stop_triggered"].any()
        )

        best_row_dict["best_row_triggered_early_stop"] = bool(
            best_model_row["early_stop_triggered"]
        )

        best_row_dict["search_stopped_by_early_stop"] = search_stopped_by_early_stop
        best_row_dict["last_evaluated_n_features"] = int(
            last_evaluated_row["n_features"]
        )

        best_row_dict["last_evaluated_balanced_accuracy"] = float(
            last_evaluated_row["balanced_accuracy"]
        )

        best_row_dict["stopped_at_n_features"] = (
            int(last_evaluated_row["n_features"])
            if search_stopped_by_early_stop
            else None
        )

        best_row_dict["n_evaluated_feature_subsets"] = len(model_results_df)

        best_rows.append(best_row_dict)

        best_n_features = int(best_model_row["n_features"])
        best_selected_features = json.loads(best_model_row["features"])

        best_features_path = (
            model_output_dir /
            f"{model_name}_BEST_selected_features.csv"
        )

        save_selected_feature_table(
            ranking_df=ranking_df,
            selected_features=best_selected_features,
            output_path=best_features_path,
            model_name=model_name,
            n_features=best_n_features,
            balanced_accuracy=best_model_row["balanced_accuracy"]
        )

        best_summary_path = (
            model_output_dir /
            f"{model_name}_BEST_configuration.csv"
        )

        pd.DataFrame([best_row_dict]).to_csv(best_summary_path, index=False)

        print("\n===== BEST CONFIGURATION FOR MODEL =====")
        print(best_model_row.to_string())
        print("[SUCCESS] Best configuration saved to:")
        print(best_summary_path)

    # ------------------------------------------------------------
    # Final global saves
    # ------------------------------------------------------------
    final_results_df = pd.DataFrame(all_results)

    final_results_path = output_dir / "bottomup_wrapper_performance_timepoints.csv"
    final_results_df.to_csv(final_results_path, index=False)

    best_by_model_df = pd.DataFrame(best_rows)

    best_by_model_path = output_dir / "bottomup_wrapper_BEST_by_model.csv"
    best_by_model_df.to_csv(best_by_model_path, index=False)

    print("\n[SUCCESS] Bottom-up wrapper feature selection completed.")
    print("[SUCCESS] Performance timepoints saved to:")
    print(final_results_path)

    print("\n[SUCCESS] Best configuration by model saved to:")
    print(best_by_model_path)

    print("\n===== BEST CONFIGURATION BY MODEL =====")
    print(
        best_by_model_df[
            [
                "model",
                "n_features",
                "balanced_accuracy",
                "accuracy",
                "recall_static",
                "recall_walking",
                "best_row_triggered_early_stop",
                "search_stopped_by_early_stop",
                "stopped_at_n_features",
                "last_evaluated_n_features",
                "n_evaluated_feature_subsets",
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":

    print("[INFO] Project root:")
    print(PROJECT_ROOT)

    print("\n[INFO] Training dataset:")
    print(TRAIN_DATASET_NAME)

    print("\n[INFO] Training/internal-validation features directory:")
    print(FEATURES_DIR)

    for window_config_name in WINDOW_CONFIG_NAMES:
        run_wrapper_for_window_config(window_config_name)