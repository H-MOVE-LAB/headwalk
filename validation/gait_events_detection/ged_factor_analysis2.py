"""
pd_clinical_factor_analysis.py

Clinical factor analysis on PD trials only.

For each selected algorithm and each clinical factor:
- creates one figure with 2 subplots:
    top: F1 boxplots
    bottom: laterality accuracy boxplots
- runs non-parametric statistical testing:
    * Mann-Whitney U if there are 2 levels
    * Kruskal-Wallis + pairwise Mann-Whitney U with Bonferroni correction if > 2 levels
- adds an asterisk next to levels involved in significant differences
- saves summary CSV files

Author: generated for your workflow
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
METADATA_XLSX = ROOT / "data" / "ICICLE Gait - 20250414_Paolo.xlsx"

# ============================================================
# USER CONFIGURATION
# ============================================================

BASE_DIR = Path("results/_analysis_plots")

GAIT_CSV = BASE_DIR / "per_trial_metrics_ss_optimized.csv"
LATERALITY_CSV = BASE_DIR / "per_trial_laterality.csv"
OUTPUT_CSV_PATH = BASE_DIR / "aggregated_factors_table.csv"

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "results" / "_pd_clinical_factor_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Algorithms to analyze for now
ALGORITHMS_TO_ANALYZE = ["Fawden", "TCN"]

# Event types to keep
EVENT_TYPES = ["IC", "FC"]

# Minimum number of samples per factor level to include the level in tests
MIN_SAMPLES_PER_LEVEL = 2

# Whether to exclude missing / unknown groups from plots and stats
DROP_UNKNOWN_LEVELS = True

# Filenames contain subject/timepoint like: INGC116F2 or INGP023F3
SUBJECT_RE = re.compile(r"(ING[CP]\d{3})(?:[^\w]|_)?(F[234])", re.IGNORECASE)

# Map filename timepoint to metadata columns
TIMEPOINT_MAP = {
    "F2": {
        "phenotype": "Motor_Phenotype_t36",
        "updrs_ii": "UPDRS_II_t36",
        "updrs_iii": "UPDRS_III_t36",
        "hy": "HY_t36",
        "abc": "ABC_t36",
        "gds": "GDS_t36",
        "moca": "MoCA_total_t36",
        "mmse": "MMSE_total_t36",
        "fog_binary": "NFOG_Binary_t36",
        "ledd": "LEDD_t36",
    },
    "F3": {
        "phenotype": "Motor_Phenotype_t54",
        "updrs_ii": "UPDRS_II_t54",
        "updrs_iii": "UPDRS_III_t54",
        "hy": "HY_t54",
        "abc": "ABC_t54",
        "gds": "GDS_t54",
        "moca": "MoCA_total_t54",
        "mmse": "MMSE_total_t54",
        "fog_binary": "NFOG_Binary_t54",
        "ledd": "LEDD_t54",
    },
    "F4": {
        "phenotype": "Motor_Phenotype_t72",
        "updrs_ii": "UPDRS_II_t72",
        "updrs_iii": "UPDRS_III_t72",
        "hy": "HY_t72",
        "abc": "ABC_t72",
        "gds": "GDS_t72",
        "moca": "MoCA_total_t72",
        "mmse": "MMSE_total_t72",
        "fog_binary": "NFOG_Binary_t72",
        "ledd": "LEDD_t72",
    },
}

# Clinical scale bins.
# Easy to modify in future runs.
CLINICAL_BINS = {
    "updrs_ii": {
        "bins": [-np.inf, 10, 20, np.inf],
        "labels": ["Low", "Moderate", "High"],
    },
    "updrs_iii": {
        "bins": [-np.inf, 20, 35, np.inf],
        "labels": ["Low", "Moderate", "High"],
    },
    "hy": {
        "bins": [-np.inf, 2, 3, np.inf],
        "labels": ["Early", "Mid", "Advanced"],
    },
    "abc": { # https://www.physio-pedia.com/Activities-Specific_Balance_Confidence_Scale
        "bins": [-np.inf, 50, 80, np.inf],
        "labels": ["Low confidence", "Moderate confidence", "High confidence"],
    },
    "gds": { # https://www.sralab.org/rehabilitation-measures/geriatric-depression-scale
        "bins": [-np.inf, 5, 10, np.inf],
        "labels": ["No/low symptoms", "Possible depression", "Probable depression"],
    },
    "moca": { # https://pubmed.ncbi.nlm.nih.gov/39471638/
        "bins": [-np.inf, 22, 26, np.inf],
        "labels": ["Moderate/Severe", "Mild impairment", "Normal"],
    },
    "mmse": { # https://pubmed.ncbi.nlm.nih.gov/39471638/
        "bins": [-np.inf, 24, 27, np.inf],
        "labels": ["Moderate/Severe", "Mild impairment", "Normal"],
    },
    "ledd": {
        "bins": [-np.inf, 399, 699, np.inf],
        "labels": ["Low", "Medium", "High"],
    },
}

# Desired plotting order for factor levels
FACTOR_LEVEL_ORDER = {
    "Phenotype": None,  # sorted alphabetically if None
    "UPDRS_II": ["Low", "Moderate", "High"],
    "UPDRS_III": ["Low", "Moderate", "High"],
    "Hohen_Yahr": ["Early", "Mid", "Advanced"],
    "ABC": ["Low confidence", "Moderate confidence", "High confidence"],
    "GDS": ["No/low symptoms", "Possible depression", "Probable depression"],
    "MOCA": ["Moderate/Severe", "Mild impairment", "Normal"],
    "MMSE": ["Moderate/Severe", "Mild impairment", "Normal"],
    "FOG": ["No", "Yes"],
    "LEDD": ["Low", "Medium", "High"],
}

# Factors to analyze
FACTORS_CONFIG = [
    {"name": "Phenotype", "source_col": "phenotype", "group_col": "phenotype_group"},
    {"name": "UPDRS_II", "source_col": "updrs_ii", "group_col": "updrs_ii_group"},
    {"name": "UPDRS_III", "source_col": "updrs_iii", "group_col": "updrs_iii_group"},
    {"name": "Hohen_Yahr", "source_col": "hy", "group_col": "hy_group"},
    {"name": "ABC", "source_col": "abc", "group_col": "abc_group"},
    {"name": "GDS", "source_col": "gds", "group_col": "gds_group"},
    {"name": "MOCA", "source_col": "moca", "group_col": "moca_group"},
    {"name": "MMSE", "source_col": "mmse", "group_col": "mmse_group"},
    {"name": "FOG", "source_col": "fog_binary", "group_col": "fog_group"},
    {"name": "LEDD", "source_col": "ledd", "group_col": "ledd_group"},
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def standardize_cohort(series: pd.Series) -> pd.Series:
    """
    Standardize cohort labels to PD / Control.
    """
    mapping = {
        "p": "PD",
        "pd": "PD",
        "c": "Control",
        "control": "Control",
    }
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(mapping).fillna(series.astype(str).str.strip())


def extract_subject_and_timepoint(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract subject code and timepoint from filename.
    Example:
        20140616-104931-INGP116F2_SC.h5
    Returns:
        ("INGP116", "F2")
    """
    match = SUBJECT_RE.search(str(filename))
    if not match:
        return None, None
    return match.group(1).upper(), match.group(2).upper()


def sanitize_label(value: Any) -> str:
    """
    Convert any factor value into a clean string label.
    """
    if pd.isna(value):
        return "Unknown"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"

    text = str(value).strip()
    return text if text else "Unknown"


def normalize_fog_value(value: Any) -> str:
    """
    Normalize binary FOG values to Yes / No / Unknown.
    """
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true"}:
        return "Yes"
    if text in {"0", "no", "n", "false"}:
        return "No"
    return "Unknown"


def bin_numeric_value(value: Any, bins: List[float], labels: List[str]) -> str:
    """
    Assign a numeric value to one of the configured bins.
    Intervals are:
      (-inf, b1], (b1, b2], (b2, b3], ...
    """
    if pd.isna(value):
        return "Unknown"

    try:
        x = float(value)
    except Exception:
        return "Unknown"

    for i in range(len(labels)):
        lower = bins[i]
        upper = bins[i + 1]

        if i == 0:
            if x <= upper:
                return labels[i]
        else:
            if lower < x <= upper:
                return labels[i]

    return "Unknown"


def format_p_value(p: float) -> str:
    """
    Format p-value for logging.
    """
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


# ============================================================
# DATA LOADING
# ============================================================

def load_gait_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load gait event detection CSV.
    Required columns:
        algo, cohort, event_type, filename, f1
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Gait CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"algo", "cohort", "event_type", "filename", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gait CSV missing required columns: {sorted(missing)}")

    df["algo"] = df["algo"].astype(str).str.strip()
    df["cohort"] = standardize_cohort(df["cohort"])
    df["event_type"] = df["event_type"].astype(str).str.strip().str.upper()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    return df


def load_laterality_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load laterality detection CSV.
    Required columns:
        algo, cohort, event_type, filename, accuracy
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Laterality CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"algo", "cohort", "event_type", "filename", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Laterality CSV missing required columns: {sorted(missing)}")

    df["algo"] = df["algo"].astype(str).str.strip()
    df["cohort"] = standardize_cohort(df["cohort"])
    df["event_type"] = df["event_type"].astype(str).str.strip().str.upper()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    return df


def load_metadata_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    Load metadata Excel file.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Metadata Excel not found: {xlsx_path}")

    meta = pd.read_excel(xlsx_path)
    meta.columns = [c.strip() for c in meta.columns]

    if "First_name" not in meta.columns:
        raise ValueError("Metadata Excel must contain 'First_name' column")

    meta["First_name"] = meta["First_name"].astype(str).str.strip().str.upper()
    return meta


# ============================================================
# MERGING AND FACTOR ATTACHMENT
# ============================================================

def merge_trial_metrics(gait_df: pd.DataFrame, laterality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-trial gait and laterality metrics.
    Join keys:
        algo, cohort, event_type, filename
    """
    merge_keys = ["algo", "cohort", "event_type", "filename"]

    merged = pd.merge(
        gait_df[merge_keys + ["f1"]],
        laterality_df[merge_keys + ["accuracy"]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    subject_codes = []
    timepoints = []

    for filename in merged["filename"]:
        subject_code, timepoint = extract_subject_and_timepoint(filename)
        subject_codes.append(subject_code)
        timepoints.append(timepoint)

    merged["subject_code"] = subject_codes
    merged["timepoint"] = timepoints

    return merged


def attach_clinical_factors(per_trial_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach clinical variables to each per-trial row using:
    - subject code
    - filename timepoint
    """
    metadata_lookup = metadata_df.set_index("First_name", drop=False)
    output_rows = []

    factor_keys = [
        "phenotype",
        "updrs_ii",
        "updrs_iii",
        "hy",
        "abc",
        "gds",
        "moca",
        "mmse",
        "fog_binary",
        "ledd",
    ]

    for _, row in per_trial_df.iterrows():
        row_out = row.to_dict()

        subject_code = row["subject_code"]
        timepoint = row["timepoint"]

        # Initialize with missing values
        for key in factor_keys:
            row_out[key] = np.nan

        if subject_code is None or timepoint is None:
            output_rows.append(row_out)
            continue

        if subject_code not in metadata_lookup.index:
            output_rows.append(row_out)
            continue

        tp_map = TIMEPOINT_MAP.get(timepoint)
        if tp_map is None:
            output_rows.append(row_out)
            continue

        meta_row = metadata_lookup.loc[subject_code]

        for key in factor_keys:
            metadata_col = tp_map.get(key)
            if metadata_col is not None:
                row_out[key] = meta_row.get(metadata_col, np.nan)

        output_rows.append(row_out)

    df = pd.DataFrame(output_rows)

    # Standardize factor columns
    df["phenotype_group"] = df["phenotype"].apply(sanitize_label)

    for numeric_col in ["updrs_ii", "updrs_iii", "hy", "abc", "gds", "moca", "mmse", "ledd"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    df["fog_binary"] = df["fog_binary"].apply(normalize_fog_value)
    df["fog_group"] = df["fog_binary"].apply(sanitize_label)

    # Create grouped categorical versions for numeric scales
    for scale_name, config in CLINICAL_BINS.items():
        df[f"{scale_name}_group"] = df[scale_name].apply(
            lambda x: bin_numeric_value(x, config["bins"], config["labels"])
        )

    return df


# ============================================================
# DESCRIPTIVE TABLE
# ============================================================

def build_aggregated_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an aggregated performance table:
    rows   = algorithm x cohort
    cols   = IC/FC x (F1, Side Accuracy)
    values = mean
    """
    data = df.copy()

    grouped = (
        data.groupby(["algo", "cohort", "event_type"], dropna=False)
        .agg(
            f1_mean=("f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            n=("filename", "count"),
        )
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["algo", "cohort"],
        columns="event_type",
        values=["f1_mean", "accuracy_mean", "n"],
        aggfunc="first",
    )

    # Reorder columns
    desired_cols = []
    for metric in ["f1_mean", "accuracy_mean", "n"]:
        for event_type in ["IC", "FC"]:
            desired_cols.append((metric, event_type))

    existing_cols = [col for col in desired_cols if col in pivot.columns]
    pivot = pivot.reindex(columns=existing_cols)

    return pivot


# ============================================================
# GROUPING AND STATISTICS
# ============================================================

def get_ordered_levels(df: pd.DataFrame, factor_name: str, group_col: str) -> List[str]:
    """
    Get factor levels in the desired order.
    """
    levels = [sanitize_label(v) for v in df[group_col].dropna().tolist()]
    levels = list(dict.fromkeys(levels))

    if DROP_UNKNOWN_LEVELS:
        levels = [level for level in levels if level != "Unknown"]

    custom_order = FACTOR_LEVEL_ORDER.get(factor_name)
    if custom_order is not None:
        return [level for level in custom_order if level in levels]

    return sorted(levels)


def summarize_metric_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_levels: List[str],
) -> pd.DataFrame:
    """
    Descriptive statistics by factor level for one metric.
    """
    rows = []

    for level in ordered_levels:
        values = pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna()

        rows.append(
            {
                "level": level,
                "metric": metric_col,
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if len(values) > 0 else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "median": float(values.median()) if len(values) > 0 else np.nan,
                "q1": float(values.quantile(0.25)) if len(values) > 0 else np.nan,
                "q3": float(values.quantile(0.75)) if len(values) > 0 else np.nan,
                "min": float(values.min()) if len(values) > 0 else np.nan,
                "max": float(values.max()) if len(values) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def pairwise_mannwhitney_bonferroni(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_levels: List[str],
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.
    """
    valid_results = []
    placeholders = []

    for level_a, level_b in combinations(ordered_levels, 2):
        x = pd.to_numeric(df.loc[df[group_col] == level_a, metric_col], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(df.loc[df[group_col] == level_b, metric_col], errors="coerce").dropna().to_numpy()

        if len(x) < MIN_SAMPLES_PER_LEVEL or len(y) < MIN_SAMPLES_PER_LEVEL:
            placeholders.append(
                {
                    "metric": metric_col,
                    "group_1": level_a,
                    "group_2": level_b,
                    "n_1": len(x),
                    "n_2": len(y),
                    "statistic": np.nan,
                    "p_uncorrected": np.nan,
                    "p_bonferroni": np.nan,
                    "significant": False,
                    "note": "Insufficient samples",
                }
            )
            continue

        stat, p_uncorrected = stats.mannwhitneyu(x, y, alternative="two-sided")
        valid_results.append((level_a, level_b, len(x), len(y), float(stat), float(p_uncorrected)))

    n_comparisons = len(valid_results)

    output_rows = []
    for level_a, level_b, n1, n2, stat, p_uncorrected in valid_results:
        p_bonf = min(1.0, p_uncorrected * n_comparisons) if n_comparisons > 0 else p_uncorrected

        output_rows.append(
            {
                "metric": metric_col,
                "group_1": level_a,
                "group_2": level_b,
                "n_1": n1,
                "n_2": n2,
                "statistic": stat,
                "p_uncorrected": p_uncorrected,
                "p_bonferroni": p_bonf,
                "significant": bool(p_bonf < 0.05),
                "note": "",
            }
        )

    output_rows.extend(placeholders)

    posthoc_df = pd.DataFrame(output_rows)

    if not posthoc_df.empty:
        posthoc_df = posthoc_df.sort_values(by=["group_1", "group_2"]).reset_index(drop=True)

    return posthoc_df


def run_statistical_tests(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_levels: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Run omnibus and posthoc tests for one metric.
    Returns:
    - omnibus dataframe
    - posthoc dataframe
    - list of levels involved in significant differences
    """
    valid_levels = []
    group_arrays = []

    for level in ordered_levels:
        values = pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna().to_numpy()
        if len(values) >= MIN_SAMPLES_PER_LEVEL:
            valid_levels.append(level)
            group_arrays.append(values)

    if len(valid_levels) < 2:
        omnibus_df = pd.DataFrame(
            [
                {
                    "metric": metric_col,
                    "test": "not_run",
                    "n_levels": len(valid_levels),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "note": "Fewer than two valid levels",
                }
            ]
        )
        return omnibus_df, pd.DataFrame(), []

    if len(valid_levels) == 2:
        stat, p_value = stats.mannwhitneyu(group_arrays[0], group_arrays[1], alternative="two-sided")

        omnibus_df = pd.DataFrame(
            [
                {
                    "metric": metric_col,
                    "test": "mannwhitneyu",
                    "n_levels": 2,
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "note": "",
                }
            ]
        )

        posthoc_df = pd.DataFrame(
            [
                {
                    "metric": metric_col,
                    "group_1": valid_levels[0],
                    "group_2": valid_levels[1],
                    "n_1": len(group_arrays[0]),
                    "n_2": len(group_arrays[1]),
                    "statistic": float(stat),
                    "p_uncorrected": float(p_value),
                    "p_bonferroni": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "note": "",
                }
            ]
        )

        significant_levels = valid_levels.copy() if p_value < 0.05 else []
        return omnibus_df, posthoc_df, significant_levels

    H, p_value = stats.kruskal(*group_arrays)

    omnibus_df = pd.DataFrame(
        [
            {
                "metric": metric_col,
                "test": "kruskal",
                "n_levels": len(valid_levels),
                "statistic": float(H),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "note": "",
            }
        ]
    )

    posthoc_df = pairwise_mannwhitney_bonferroni(df, group_col, metric_col, valid_levels)

    significant_levels_set = set()
    if not posthoc_df.empty:
        sig_rows = posthoc_df[posthoc_df["significant"] == True]
        for _, row in sig_rows.iterrows():
            significant_levels_set.add(row["group_1"])
            significant_levels_set.add(row["group_2"])

    significant_levels = [level for level in valid_levels if level in significant_levels_set]

    return omnibus_df, posthoc_df, significant_levels


# ============================================================
# PLOTTING
# ============================================================

def add_significance_asterisks(
    ax: plt.Axes,
    ordered_levels: List[str],
    data_arrays: List[np.ndarray],
    significant_levels: List[str],
    metric_name: str,
) -> None:
    """
    Add an asterisk beside the boxplot of each level involved in at least
    one significant comparison.
    """
    if not significant_levels:
        return

    finite_values = np.concatenate(
        [arr[np.isfinite(arr)] for arr in data_arrays if len(arr) > 0]
    ) if data_arrays else np.array([])

    if finite_values.size == 0:
        return

    y_min = np.min(finite_values)
    y_max = np.max(finite_values)
    y_range = max(y_max - y_min, 1e-6)
    offset = 0.06 * y_range

    if metric_name in {"f1", "accuracy"}:
        ax.set_ylim(bottom=min(-0.05, y_min - 0.05), top=max(1.05, y_max + 0.12))

    for idx, (level, arr) in enumerate(zip(ordered_levels, data_arrays), start=1):
        if level not in significant_levels:
            continue
        if len(arr) == 0:
            continue

        valid_arr = arr[np.isfinite(arr)]
        if len(valid_arr) == 0:
            continue

        y_pos = np.max(valid_arr) + offset
        ax.text(idx + 0.18, y_pos, "*", fontsize=16, fontweight="bold", va="center")


def make_factor_figure(
    df: pd.DataFrame,
    algorithm: str,
    factor_name: str,
    group_col: str,
    ordered_levels: List[str],
    significant_levels_f1: List[str],
    significant_levels_acc: List[str],
    output_png: Path,
) -> None:
    """
    Save figure with:
    - top subplot: F1
    - bottom subplot: laterality accuracy
    """
    f1_data = [
        pd.to_numeric(df.loc[df[group_col] == level, "f1"], errors="coerce").dropna().to_numpy()
        for level in ordered_levels
    ]
    acc_data = [
        pd.to_numeric(df.loc[df[group_col] == level, "accuracy"], errors="coerce").dropna().to_numpy()
        for level in ordered_levels
    ]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(max(10, 1.5 * len(ordered_levels)), 9),
        sharex=True,
    )
    fig.suptitle(f"{algorithm} - {factor_name} (PD trials)", fontsize=14)

    # Top: F1
    ax1.boxplot(f1_data, labels=ordered_levels, patch_artist=True, showfliers=False)
    ax1.set_ylabel("F1 score")
    ax1.set_title("Gait event detection performance")
    ax1.grid(axis="y", alpha=0.3)
    add_significance_asterisks(ax1, ordered_levels, f1_data, significant_levels_f1, metric_name="f1")

    # Bottom: Accuracy
    ax2.boxplot(acc_data, labels=ordered_levels, patch_artist=True, showfliers=False)
    ax2.set_ylabel("Laterality accuracy")
    ax2.set_title("Laterality detection performance")
    ax2.grid(axis="y", alpha=0.3)
    add_significance_asterisks(ax2, ordered_levels, acc_data, significant_levels_acc, metric_name="accuracy")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


# ============================================================
# FACTOR ANALYSIS
# ============================================================

def analyze_factor_for_algorithm(
    df_algorithm_pd: pd.DataFrame,
    algorithm_name: str,
    factor_config: Dict[str, str],
    output_dir_algorithm: Path,
) -> None:
    """
    Analyze one factor for one algorithm.
    Save:
    - figure
    - data used
    - summary CSV
    - omnibus CSV
    - posthoc CSV
    """
    factor_name = factor_config["name"]
    group_col = factor_config["group_col"]

    if group_col not in df_algorithm_pd.columns:
        print(f"[SKIP] {algorithm_name} - {factor_name}: column '{group_col}' not found.")
        return

    df_factor = df_algorithm_pd.copy()

    if DROP_UNKNOWN_LEVELS:
        df_factor = df_factor[df_factor[group_col] != "Unknown"].copy()

    ordered_levels = get_ordered_levels(df_factor, factor_name, group_col)

    if len(ordered_levels) < 2:
        print(f"[SKIP] {algorithm_name} - {factor_name}: fewer than 2 usable levels.")
        return

    df_factor = df_factor[df_factor[group_col].isin(ordered_levels)].copy()

    # Save raw data used for this factor
    factor_data_path = output_dir_algorithm / f"{factor_name}_data_used.csv"
    df_factor.to_csv(factor_data_path, index=False)

    # Summary statistics
    summary_f1 = summarize_metric_by_group(df_factor, group_col, "f1", ordered_levels)
    summary_acc = summarize_metric_by_group(df_factor, group_col, "accuracy", ordered_levels)
    summary_df = pd.concat([summary_f1, summary_acc], ignore_index=True)

    # Statistical tests
    omnibus_f1, posthoc_f1, sig_levels_f1 = run_statistical_tests(df_factor, group_col, "f1", ordered_levels)
    omnibus_acc, posthoc_acc, sig_levels_acc = run_statistical_tests(df_factor, group_col, "accuracy", ordered_levels)

    omnibus_df = pd.concat([omnibus_f1, omnibus_acc], ignore_index=True)

    posthoc_parts = []
    if not posthoc_f1.empty:
        posthoc_f1["factor"] = factor_name
        posthoc_parts.append(posthoc_f1)
    if not posthoc_acc.empty:
        posthoc_acc["factor"] = factor_name
        posthoc_parts.append(posthoc_acc)

    posthoc_df = pd.concat(posthoc_parts, ignore_index=True) if posthoc_parts else pd.DataFrame()

    # Figure
    figure_path = output_dir_algorithm / f"{factor_name}_boxplots.png"
    make_factor_figure(
        df=df_factor,
        algorithm=algorithm_name,
        factor_name=factor_name,
        group_col=group_col,
        ordered_levels=ordered_levels,
        significant_levels_f1=sig_levels_f1,
        significant_levels_acc=sig_levels_acc,
        output_png=figure_path,
    )

    # Save CSV files
    summary_path = output_dir_algorithm / f"{factor_name}_summary.csv"
    omnibus_path = output_dir_algorithm / f"{factor_name}_omnibus.csv"
    posthoc_path = output_dir_algorithm / f"{factor_name}_posthoc.csv"

    summary_df.to_csv(summary_path, index=False)
    omnibus_df.to_csv(omnibus_path, index=False)
    if not posthoc_df.empty:
        posthoc_df.to_csv(posthoc_path, index=False)

    print(f"[SAVED] {algorithm_name} - {factor_name}")
    print(f"        Figure  : {figure_path}")
    print(f"        Summary : {summary_path}")
    print(f"        Omnibus : {omnibus_path}")
    if not posthoc_df.empty:
        print(f"        Posthoc : {posthoc_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Loading input files...")
    gait_df = load_gait_csv(GAIT_CSV)
    laterality_df = load_laterality_csv(LATERALITY_CSV)
    metadata_df = load_metadata_excel(METADATA_XLSX)

    print("Merging gait and laterality trial-level data...")
    merged_df = merge_trial_metrics(gait_df, laterality_df)

    print("Attaching clinical factors from metadata...")
    analysis_df = attach_clinical_factors(merged_df, metadata_df)

    # Save global merged file
    merged_output_path = OUTPUT_DIR / "merged_trial_level_data.csv"
    analysis_df.to_csv(merged_output_path, index=False)

    # Save aggregated performance table
    aggregated_table = build_aggregated_performance_table(analysis_df)
    aggregated_table.to_csv(OUTPUT_CSV_PATH)
    print(f"Aggregated performance table saved to: {OUTPUT_CSV_PATH.resolve()}")

    # PD only
    analysis_df = analysis_df[analysis_df["cohort"] == "PD"].copy()
    if analysis_df.empty:
        raise RuntimeError("No PD trials found after filtering cohort == 'PD'.")

    # Keep only IC / FC
    analysis_df = analysis_df[analysis_df["event_type"].isin(EVENT_TYPES)].copy()
    if analysis_df.empty:
        raise RuntimeError(f"No PD trials found for event types: {EVENT_TYPES}")

    print("\nStarting PD clinical factor analysis...\n")

    for algorithm_name in ALGORITHMS_TO_ANALYZE:
        df_algorithm = analysis_df[analysis_df["algo"] == algorithm_name].copy()

        if df_algorithm.empty:
            print(f"[SKIP] No PD trials found for algorithm: {algorithm_name}")
            continue

        output_dir_algorithm = OUTPUT_DIR / algorithm_name
        output_dir_algorithm.mkdir(parents=True, exist_ok=True)

        # Save algorithm-specific merged data
        df_algorithm.to_csv(output_dir_algorithm / "merged_pd_trials_used.csv", index=False)

        for factor_config in FACTORS_CONFIG:
            analyze_factor_for_algorithm(
                df_algorithm_pd=df_algorithm,
                algorithm_name=algorithm_name,
                factor_config=factor_config,
                output_dir_algorithm=output_dir_algorithm,
            )

    print("\nAll analyses completed.")
    print(f"Results saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()