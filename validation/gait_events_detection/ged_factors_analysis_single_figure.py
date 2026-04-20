"""
pd_clinical_factor_analysis.py

PD-only clinical factor analysis.

This script:
- loads per-trial gait and laterality performance
- merges them with clinical metadata
- keeps PD trials only
- analyzes 9 clinical factors (UPDRS II excluded)
- creates exactly two figures:
    1) gait event detection figure (3x3 subplots, F1 score)
    2) laterality detection figure (3x3 subplots, laterality accuracy)
- each subplot contains boxplots for one factor
- each factor uses its own color palette
- each factor level displays:
    * sample size (n)
    * median value, written at the median line of the boxplot
- saves descriptive and statistical CSV files

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
OUTPUT_CSV_PATH = BASE_DIR / "aggregated_performance_table.csv"

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "results" / "_pd_clinical_factor_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Algorithms to analyze
ALGORITHMS_TO_ANALYZE = ["Fawden", "TCN"]

# Event types to keep
EVENT_TYPES = ["IC", "FC"]

# Whether to pool IC and FC together in the clinical plots.
# If False, you should add additional filtering logic.
POOL_EVENT_TYPES = True

# Minimum number of samples per factor level for statistical testing
MIN_SAMPLES_PER_LEVEL = 2

# Exclude missing / unknown levels from plots and stats
DROP_UNKNOWN_LEVELS = True

# ------------------------------------------------------------
# GRAPHICS CONFIGURATION
# ------------------------------------------------------------

GRAPHICS = {
    "figure_size": (20, 16),
    "figure_dpi": 300,
    "subplot_wspace": 0.28,
    "subplot_hspace": 0.38,
    "title_fontsize": 18,
    "suptitle_fontsize": 22,
    "axis_label_fontsize": 12,
    "subplot_title_fontsize": 13,
    "tick_label_fontsize": 10,
    "annotation_fontsize": 9,
    "n_annotation_fontsize": 9,
    "asterisk_fontsize": 15,
    "font_family": "sans-serif",
    "box_width": 0.55,
    "show_fliers": False,
    "grid_alpha": 0.25,
    "grid_linestyle": "--",
    "median_linewidth": 2.0,
    "box_alpha": 0.75,
    "box_linewidth": 1.2,
    "whisker_linewidth": 1.1,
    "cap_linewidth": 1.1,
    "axis_spine_linewidth": 1.0,
    "x_tick_rotation": 25,
    "annotation_x_offset": 0.00,
    "n_text_y_axes_fraction": 0.04,
    "asterisk_x_offset": 0.18,
    "asterisk_y_offset_fraction": 0.06,
    "median_text_bbox_alpha": 0.75,
}

plt.rcParams["font.family"] = GRAPHICS["font_family"]

# ------------------------------------------------------------
# FACTOR COLOR PALETTES
# One palette per factor; colors are cycled if needed.
# ------------------------------------------------------------

FACTOR_COLORS = {
    "Phenotype": ["#c0392b", "#e74c3c", "#f1948a", "#f5b7b1"],
    "UPDRS_III": ["#1f618d", "#2e86c1", "#85c1e9", "#aed6f1"],
    "Hohen_Yahr": ["#117a65", "#17a589", "#73c6b6", "#a3e4d7"],
    "ABC": ["#7d6608", "#b7950b", "#f4d03f", "#f9e79f"],
    "GDS": ["#6c3483", "#8e44ad", "#c39bd3", "#e8daef"],
    "MOCA": ["#7b241c", "#a93226", "#d98880", "#f2d7d5"],
    "MMSE": ["#0e6251", "#138d75", "#76d7c4", "#d1f2eb"],
    "FOG": ["#5b2c6f", "#884ea0", "#c39bd3"],
    "LEDD": ["#784212", "#af601a", "#d68910", "#f8c471"],
}

# ------------------------------------------------------------
# Factor configuration
# UPDRS II intentionally excluded
# ------------------------------------------------------------

FACTORS_CONFIG = [
    {"name": "Phenotype", "source_col": "phenotype", "group_col": "phenotype_group"},
    {"name": "UPDRS_III", "source_col": "updrs_iii", "group_col": "updrs_iii_group"},
    {"name": "Hohen_Yahr", "source_col": "hy", "group_col": "hy_group"},
    {"name": "ABC", "source_col": "abc", "group_col": "abc_group"},
    {"name": "GDS", "source_col": "gds", "group_col": "gds_group"},
    {"name": "MOCA", "source_col": "moca", "group_col": "moca_group"},
    {"name": "MMSE", "source_col": "mmse", "group_col": "mmse_group"},
    {"name": "FOG", "source_col": "fog_binary", "group_col": "fog_group"},
    {"name": "LEDD", "source_col": "ledd", "group_col": "ledd_group"},
]

CLINICAL_BINS = {
    "updrs_iii": {
        "bins": [-np.inf, 20, 35, np.inf],
        "labels": ["Low", "Moderate", "High"],
    },
    "hy": {
        "bins": [-np.inf, 2, 3, np.inf],
        "labels": ["Early", "Mid", "Advanced"],
    },
    "abc": {
        "bins": [-np.inf, 49, 79, np.inf],
        "labels": ["Low confidence", "Moderate confidence", "High confidence"],
    },
    "gds": {
        "bins": [-np.inf, 5, 10, np.inf],
        "labels": ["No/low symptoms", "Possible depression", "Probable depression"],
    },
    "moca": {
        "bins": [-np.inf, 17, 25, np.inf],
        "labels": ["Moderate/Severe", "Mild impairment", "Normal"],
    },
    "mmse": {
        "bins": [-np.inf, 17, 23, np.inf],
        "labels": ["Moderate/Severe", "Mild impairment", "Normal"],
    },
    "ledd": {
        "bins": [-np.inf, 399, 699, np.inf],
        "labels": ["Low", "Medium", "High"],
    },
    "fog_binary": {
        "bins": [-np.inf, 0.5, np.inf],
        "labels": ["No", "Yes"],
    },
}

FACTOR_LEVEL_ORDER = {
    "Phenotype": None,
    "UPDRS_III": ["Low", "Moderate", "High"],
    "Hohen_Yahr": ["Early", "Mid", "Advanced"],
    "ABC": ["Low confidence", "Moderate confidence", "High confidence"],
    "GDS": ["No/low symptoms", "Possible depression", "Probable depression"],
    "MOCA": ["Moderate/Severe", "Mild impairment", "Normal"],
    "MMSE": ["Moderate/Severe", "Mild impairment", "Normal"],
    "FOG": ["No", "Yes"],
    "LEDD": ["Low", "Medium", "High"],
}

SUBJECT_RE = re.compile(r"(ING[CP]\d{3})(?:[^\w]|_)?(F[234])", re.IGNORECASE)

TIMEPOINT_MAP = {
    "F2": {
        "phenotype": "Motor_Phenotype_t36",
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


# ============================================================
# UTILITIES
# ============================================================

def standardize_cohort(series: pd.Series) -> pd.Series:
    mapping = {
        "p": "PD",
        "pd": "PD",
        "c": "Control",
        "control": "Control",
    }
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(mapping).fillna(series.astype(str).str.strip())


def extract_subject_and_timepoint(filename: str) -> Tuple[Optional[str], Optional[str]]:
    match = SUBJECT_RE.search(str(filename))
    if not match:
        return None, None
    return match.group(1).upper(), match.group(2).upper()


def sanitize_label(value: Any) -> str:
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
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true"}:
        return "Yes"
    if text in {"0", "no", "n", "false"}:
        return "No"
    return "Unknown"


def bin_numeric_value(value: Any, bins: List[float], labels: List[str]) -> str:
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_gait_csv(csv_path: Path) -> pd.DataFrame:
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
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Metadata Excel not found: {xlsx_path}")

    meta = pd.read_excel(xlsx_path)
    meta.columns = [c.strip() for c in meta.columns]

    if "First_name" not in meta.columns:
        raise ValueError("Metadata Excel must contain 'First_name' column")

    meta["First_name"] = meta["First_name"].astype(str).str.strip().str.upper()
    return meta


# ============================================================
# MERGING / FACTOR ATTACHMENT
# ============================================================

def merge_trial_metrics(gait_df: pd.DataFrame, laterality_df: pd.DataFrame) -> pd.DataFrame:
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
    metadata_lookup = metadata_df.set_index("First_name", drop=False)
    output_rows = []

    factor_keys = [
        "phenotype",
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

    df["phenotype_group"] = df["phenotype"].apply(sanitize_label)

    for numeric_col in ["updrs_iii", "hy", "abc", "gds", "moca", "mmse", "ledd"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    df["fog_binary"] = df["fog_binary"].apply(normalize_fog_value)
    df["fog_group"] = df["fog_binary"].apply(sanitize_label)

    for scale_name, config in CLINICAL_BINS.items():
        df[f"{scale_name}_group"] = df[scale_name].apply(
            lambda x: bin_numeric_value(x, config["bins"], config["labels"])
        )

    return df


# ============================================================
# AGGREGATED PERFORMANCE TABLE
# ============================================================

def build_aggregated_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["algo", "cohort", "event_type"], dropna=False)
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

    desired_cols = []
    for metric in ["f1_mean", "accuracy_mean", "n"]:
        for event_type in ["IC", "FC"]:
            desired_cols.append((metric, event_type))

    existing_cols = [col for col in desired_cols if col in pivot.columns]
    pivot = pivot.reindex(columns=existing_cols)
    return pivot


# ============================================================
# GROUP / STATS HELPERS
# ============================================================

def get_ordered_levels(df: pd.DataFrame, factor_name: str, group_col: str) -> List[str]:
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
    valid_levels = []
    group_arrays = []

    for level in ordered_levels:
        values = pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna().to_numpy()
        if len(values) >= MIN_SAMPLES_PER_LEVEL:
            valid_levels.append(level)
            group_arrays.append(values)

    if len(valid_levels) < 2:
        omnibus_df = pd.DataFrame(
            [{
                "metric": metric_col,
                "test": "not_run",
                "n_levels": len(valid_levels),
                "statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "note": "Fewer than two valid levels",
            }]
        )
        return omnibus_df, pd.DataFrame(), []

    if len(valid_levels) == 2:
        stat, p_value = stats.mannwhitneyu(group_arrays[0], group_arrays[1], alternative="two-sided")

        omnibus_df = pd.DataFrame(
            [{
                "metric": metric_col,
                "test": "mannwhitneyu",
                "n_levels": 2,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "note": "",
            }]
        )

        posthoc_df = pd.DataFrame(
            [{
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
            }]
        )

        significant_levels = valid_levels.copy() if p_value < 0.05 else []
        return omnibus_df, posthoc_df, significant_levels

    H, p_value = stats.kruskal(*group_arrays)

    omnibus_df = pd.DataFrame(
        [{
            "metric": metric_col,
            "test": "kruskal",
            "n_levels": len(valid_levels),
            "statistic": float(H),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "note": "",
        }]
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
# PLOTTING HELPERS
# ============================================================

def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=GRAPHICS["grid_alpha"], linestyle=GRAPHICS["grid_linestyle"])
    for spine in ax.spines.values():
        spine.set_linewidth(GRAPHICS["axis_spine_linewidth"])
    ax.tick_params(axis="both", labelsize=GRAPHICS["tick_label_fontsize"])


def get_factor_palette(factor_name: str, n_levels: int) -> List[str]:
    palette = FACTOR_COLORS.get(factor_name, ["#7f8c8d"])
    if len(palette) >= n_levels:
        return palette[:n_levels]

    extended = []
    for i in range(n_levels):
        extended.append(palette[i % len(palette)])
    return extended


def add_n_annotations(ax: plt.Axes, ordered_levels: List[str], counts: List[int]) -> None:
    y_frac = GRAPHICS["n_text_y_axes_fraction"]
    for idx, count in enumerate(counts, start=1):
        ax.text(
            idx,
            y_frac,
            f"n={count}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=GRAPHICS["n_annotation_fontsize"],
        )


def add_median_annotations(ax: plt.Axes, ordered_levels: List[str], medians: List[float]) -> None:
    for idx, median_value in enumerate(medians, start=1):
        if pd.isna(median_value):
            continue
        ax.text(
            idx + GRAPHICS["annotation_x_offset"],
            median_value,
            f"{median_value:.2f}",
            ha="center",
            va="center",
            fontsize=GRAPHICS["annotation_fontsize"],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": GRAPHICS["median_text_bbox_alpha"],
                "pad": 1.5,
            },
        )


def add_significance_asterisks(
    ax: plt.Axes,
    ordered_levels: List[str],
    data_arrays: List[np.ndarray],
    significant_levels: List[str],
) -> None:
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
    offset = GRAPHICS["asterisk_y_offset_fraction"] * y_range

    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max(current_ylim[1], y_max + 2.5 * offset))

    for idx, (level, arr) in enumerate(zip(ordered_levels, data_arrays), start=1):
        if level not in significant_levels:
            continue
        valid_arr = arr[np.isfinite(arr)]
        if len(valid_arr) == 0:
            continue

        y_pos = np.max(valid_arr) + offset
        ax.text(
            idx + GRAPHICS["asterisk_x_offset"],
            y_pos,
            "*",
            fontsize=GRAPHICS["asterisk_fontsize"],
            fontweight="bold",
            va="center",
        )


def draw_factor_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    factor_name: str,
    group_col: str,
    metric_col: str,
    y_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered_levels = get_ordered_levels(df, factor_name, group_col)

    if len(ordered_levels) < 2:
        ax.set_title(f"{factor_name}\nNot enough levels", fontsize=GRAPHICS["subplot_title_fontsize"])
        ax.axis("off")
        return pd.DataFrame(), pd.DataFrame()

    data_arrays = [
        pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna().to_numpy()
        for level in ordered_levels
    ]
    counts = [int(len(arr)) for arr in data_arrays]
    medians = [float(np.median(arr)) if len(arr) > 0 else np.nan for arr in data_arrays]

    summary_df = summarize_metric_by_group(df, group_col, metric_col, ordered_levels)
    omnibus_df, posthoc_df, significant_levels = run_statistical_tests(df, group_col, metric_col, ordered_levels)

    palette = get_factor_palette(factor_name, len(ordered_levels))

    bp = ax.boxplot(
        data_arrays,
        labels=ordered_levels,
        widths=GRAPHICS["box_width"],
        patch_artist=True,
        showfliers=GRAPHICS["show_fliers"],
        medianprops={"linewidth": GRAPHICS["median_linewidth"], "color": "black"},
        boxprops={"linewidth": GRAPHICS["box_linewidth"]},
        whiskerprops={"linewidth": GRAPHICS["whisker_linewidth"]},
        capprops={"linewidth": GRAPHICS["cap_linewidth"]},
    )

    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(GRAPHICS["box_alpha"])

    ax.set_title(factor_name, fontsize=GRAPHICS["subplot_title_fontsize"])
    ax.set_ylabel(y_label, fontsize=GRAPHICS["axis_label_fontsize"])
    ax.tick_params(axis="x", rotation=GRAPHICS["x_tick_rotation"])
    style_axes(ax)

    add_n_annotations(ax, ordered_levels, counts)
    add_median_annotations(ax, ordered_levels, medians)
    add_significance_asterisks(ax, ordered_levels, data_arrays, significant_levels)

    if metric_col in {"f1", "accuracy"}:
        current_ylim = ax.get_ylim()
        ax.set_ylim(min(-0.05, current_ylim[0]), max(1.05, current_ylim[1]))

    omnibus_df = omnibus_df.copy()
    omnibus_df["factor"] = factor_name
    omnibus_df["levels"] = " | ".join(ordered_levels)

    if not posthoc_df.empty:
        posthoc_df = posthoc_df.copy()
        posthoc_df["factor"] = factor_name
        posthoc_df["levels"] = " | ".join(ordered_levels)

    summary_df = summary_df.copy()
    summary_df["factor"] = factor_name
    summary_df["levels"] = " | ".join(ordered_levels)

    return summary_df, pd.concat([omnibus_df, posthoc_df], ignore_index=True) if not posthoc_df.empty else omnibus_df


# ============================================================
# FIGURE CREATION
# ============================================================

def create_metric_figure_for_algorithm(
    df_algorithm_pd: pd.DataFrame,
    algorithm_name: str,
    metric_col: str,
    metric_title: str,
    y_label: str,
    output_png: Path,
    output_summary_csv: Path,
    output_stats_csv: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=GRAPHICS["figure_size"], dpi=GRAPHICS["figure_dpi"])
    fig.suptitle(
        f"{algorithm_name} - {metric_title} - PD trials",
        fontsize=GRAPHICS["suptitle_fontsize"],
    )

    summary_parts = []
    stats_parts = []

    for ax, factor_cfg in zip(axes.flatten(), FACTORS_CONFIG):
        factor_name = factor_cfg["name"]
        group_col = factor_cfg["group_col"]

        df_factor = df_algorithm_pd.copy()
        if DROP_UNKNOWN_LEVELS:
            df_factor = df_factor[df_factor[group_col] != "Unknown"].copy()

        summary_df, stats_df = draw_factor_subplot(
            ax=ax,
            df=df_factor,
            factor_name=factor_name,
            group_col=group_col,
            metric_col=metric_col,
            y_label=y_label,
        )

        if not summary_df.empty:
            summary_df["algorithm"] = algorithm_name
            summary_df["metric_panel"] = metric_title
            summary_parts.append(summary_df)

        if not stats_df.empty:
            stats_df["algorithm"] = algorithm_name
            stats_df["metric_panel"] = metric_title
            stats_parts.append(stats_df)

    plt.subplots_adjust(
        wspace=GRAPHICS["subplot_wspace"],
        hspace=GRAPHICS["subplot_hspace"],
    )
    fig.savefig(output_png, dpi=GRAPHICS["figure_dpi"], bbox_inches="tight")
    plt.close(fig)

    if summary_parts:
        pd.concat(summary_parts, ignore_index=True).to_csv(output_summary_csv, index=False)

    if stats_parts:
        pd.concat(stats_parts, ignore_index=True).to_csv(output_stats_csv, index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Loading input files...")
    gait_df = load_gait_csv(GAIT_CSV)
    laterality_df = load_laterality_csv(LATERALITY_CSV)
    metadata_df = load_metadata_excel(METADATA_XLSX)

    print("Merging trial-level gait and laterality data...")
    merged_df = merge_trial_metrics(gait_df, laterality_df)

    print("Attaching clinical factors...")
    analysis_df = attach_clinical_factors(merged_df, metadata_df)

    merged_output_path = OUTPUT_DIR / "merged_trial_level_data.csv"
    analysis_df.to_csv(merged_output_path, index=False)

    aggregated_table = build_aggregated_performance_table(analysis_df)
    aggregated_table.to_csv(OUTPUT_CSV_PATH)
    print(f"Aggregated performance table saved to: {OUTPUT_CSV_PATH.resolve()}")

    analysis_df = analysis_df[analysis_df["cohort"] == "PD"].copy()
    if analysis_df.empty:
        raise RuntimeError("No PD trials found after filtering cohort == 'PD'.")

    analysis_df = analysis_df[analysis_df["event_type"].isin(EVENT_TYPES)].copy()
    if analysis_df.empty:
        raise RuntimeError(f"No PD trials found for event types: {EVENT_TYPES}")

    print("\nStarting PD factor plots...\n")

    for algorithm_name in ALGORITHMS_TO_ANALYZE:
        df_algorithm = analysis_df[analysis_df["algo"] == algorithm_name].copy()

        if df_algorithm.empty:
            print(f"[SKIP] No PD trials found for algorithm: {algorithm_name}")
            continue

        algorithm_dir = OUTPUT_DIR / algorithm_name
        ensure_dir(algorithm_dir)

        df_algorithm.to_csv(algorithm_dir / "merged_pd_trials_used.csv", index=False)

        gait_fig_path = algorithm_dir / f"{algorithm_name}_gait_events_detection_3x3.png"
        gait_summary_path = algorithm_dir / f"{algorithm_name}_gait_events_detection_summary.csv"
        gait_stats_path = algorithm_dir / f"{algorithm_name}_gait_events_detection_stats.csv"

        laterality_fig_path = algorithm_dir / f"{algorithm_name}_laterality_detection_3x3.png"
        laterality_summary_path = algorithm_dir / f"{algorithm_name}_laterality_detection_summary.csv"
        laterality_stats_path = algorithm_dir / f"{algorithm_name}_laterality_detection_stats.csv"

        create_metric_figure_for_algorithm(
            df_algorithm_pd=df_algorithm,
            algorithm_name=algorithm_name,
            metric_col="f1",
            metric_title="Gait Event Detection",
            y_label="F1 score",
            output_png=gait_fig_path,
            output_summary_csv=gait_summary_path,
            output_stats_csv=gait_stats_path,
        )

        create_metric_figure_for_algorithm(
            df_algorithm_pd=df_algorithm,
            algorithm_name=algorithm_name,
            metric_col="accuracy",
            metric_title="Laterality Detection",
            y_label="Laterality accuracy",
            output_png=laterality_fig_path,
            output_summary_csv=laterality_summary_path,
            output_stats_csv=laterality_stats_path,
        )

        print(f"[SAVED] {algorithm_name}")
        print(f"        {gait_fig_path}")
        print(f"        {laterality_fig_path}")

    print("\nAll analyses completed.")
    print(f"Results saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()