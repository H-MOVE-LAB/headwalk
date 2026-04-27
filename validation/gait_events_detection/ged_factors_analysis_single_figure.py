"""
pd_clinical_factor_analysis.py

PD-only clinical factor analysis.

This script:
- loads per-trial gait event metrics and laterality metrics
- merges them with clinical metadata
- keeps PD trials only
- analyzes clinical factors across PD trials
- creates two 3x3 figures per algorithm:
    1) gait event detection (F1 score)
    2) laterality detection (accuracy)
- saves merged tables, descriptive summaries, and statistical results

Important fix:
- FOG is handled as a categorical Yes/No factor and is NOT rebinned as numeric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
METADATA_XLSX = ROOT / "data" / "ICICLE Gait - 20250414_Paolo.xlsx"

BASE_DIR = Path("results/_analysis_plots")
GAIT_CSV = BASE_DIR / "per_trial_metrics_ss_optimized.csv"
LATERALITY_CSV = BASE_DIR / "per_trial_laterality.csv"
AGGREGATED_OUTPUT_CSV = BASE_DIR / "aggregated_performance_table.csv"

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "results" / "_pd_clinical_factor_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS_TO_ANALYZE = ["TCN"]#["Fawden", "TCN"]
EVENT_TYPES = ["IC"]
POOL_EVENT_TYPES = False

MIN_SAMPLES_PER_LEVEL = 2
ALPHA = 0.001
DROP_UNKNOWN_LEVELS = True


# ============================================================
# GRAPHICS
# ============================================================

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


# FACTOR_COLORS = {
#     "Phenotype": ["#c0392b", "#e74c3c", "#f1948a", "#f5b7b1"],
#     "UPDRS_III": ["#1f618d", "#2e86c1", "#85c1e9", "#aed6f1"],
#     "Hohen_Yahr": ["#117a65", "#17a589", "#73c6b6", "#a3e4d7"],
#     "ABC": ["#7d6608", "#b7950b", "#f4d03f", "#f9e79f"],
#     "GDS": ["#6c3483", "#8e44ad", "#c39bd3", "#e8daef"],
#     "MOCA": ["#7b241c", "#a93226", "#d98880", "#f2d7d5"],
#     "MMSE": ["#0e6251", "#138d75", "#76d7c4", "#d1f2eb"],
#     "FOG": ["#5b2c6f", "#884ea0", "#c39bd3"],
#     "LEDD": ["#784212", "#af601a", "#d68910", "#f8c471"],
# }
a = ["#000080", "#3A5FCD", "#63B8FF"]
# a = ["#1f618d", "#2e86c1", "#85c1e9", "#aed6f1"]
FACTOR_COLORS = { # only blue
    "Phenotype": a,
    "UPDRS_III": a,
    "Hohen_Yahr": a,
    "ABC": a,
    "GDS": a,
    "MOCA": a,
    "MMSE": a,
    "FOG": a,
    "LEDD": a,
}

# ============================================================
# FACTOR CONFIG
# ============================================================

FACTORS_CONFIG = [
    {"name": "Phenotype", "raw_col": "phenotype", "group_col": "phenotype_group"},
    {"name": "UPDRS_III", "raw_col": "updrs_iii", "group_col": "updrs_iii_group"},
    {"name": "Hohen_Yahr", "raw_col": "hy", "group_col": "hy_group"},
    {"name": "ABC", "raw_col": "abc", "group_col": "abc_group"},
    {"name": "GDS", "raw_col": "gds", "group_col": "gds_group"},
    {"name": "MOCA", "raw_col": "moca", "group_col": "moca_group"},
    {"name": "MMSE", "raw_col": "mmse", "group_col": "mmse_group"},
    {"name": "FOG", "raw_col": "fog_binary", "group_col": "fog_group"},
    {"name": "LEDD", "raw_col": "ledd", "group_col": "ledd_group"},
]
# (lower < x <= upper. es. ABC: 49 (excluded) - 79 (included) )
NUMERIC_BIN_CONFIG = {
    "updrs_iii": { # https://pubmed.ncbi.nlm.nih.gov/25466406/ Martined-Martin et al. (2014)
        "bins": [-np.inf, 32, 58, np.inf],
        "labels": ["Low", "Moderate", "Severe"],
    },
    "hy": {
        "bins": [-np.inf, 2, 3, np.inf],
        "labels": ["Early", "Mid", "Advanced"],
    },
    "abc": { # https://strokengine.ca/en/assessments/activities-specific-balance-confidence-scale-abc-scale/ - Myers et al. (1998)
        "bins": [-np.inf, 49, 79, np.inf],
        "labels": ["Low", "Moderate", "High"],
    },
    "gds": { # https://www.sralab.org/rehabilitation-measures/geriatric-depression-scale - McDowell et al. (2006)
        # "bins": [-np.inf, 5, 10, np.inf],
        "bins": [-np.inf, 9, np.inf],

        # "labels": ["No/low symptoms", "Possible depression", "Probable depression"],
        "labels": ["No depression", "Depression"],

    },
    "moca": { # https://pubmed.ncbi.nlm.nih.gov/39471638/ # Fiorenzato et al. (2024)
        # "bins": [-np.inf, 17, 25, np.inf],
        # "labels": ["Moderate/Severe", "Mild impairment", "Normal"],
        "bins": [-np.inf, 22, np.inf],
        "labels": ["Mild", "Normal"],
    },
    "mmse": { # https://pubmed.ncbi.nlm.nih.gov/39471638/ # Fiorenzato et al. (2024)
        "bins": [-np.inf, 24, np.inf],
        "labels": ["Dementia", "Mild"],
    },
    "ledd": { # https://pmc.ncbi.nlm.nih.gov/articles/PMC10525064/#:~:text=LEDD%20Covariate,total%20daily%20dopaminergic%20medication%20dosing.&text=Table%201%20shows%20PD%20treatment,as%20a%20primary%20categorical%20predictor.
        "bins": [-np.inf, 399, 699, np.inf],
        "labels": ["Low", "Medium", "High"],
    },
}

FACTOR_LEVEL_ORDER = {
    "Phenotype": None,
    "UPDRS_III": ["Low", "Moderate", "Severe"],
    "Hohen_Yahr": ["Early", "Mid", "Advanced"],
    "ABC": ["Low", "Moderate", "High"],
    "GDS": ["No depression", "Depression"],
    "MOCA": ["Mild", "Normal"],
    "MMSE": ["Dementia", "Mild"],
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
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def standardize_cohort(series: pd.Series) -> pd.Series:
    mapping = {
        "p": "PD",
        "pd": "PD",
        "c": "Control",
        "control": "Control",
    }
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(mapping).fillna(series.astype(str).str.strip())


def extract_subject_and_timepoint(filename: str) -> tuple[str | None, str | None]:
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

    if text in {"1", "1.0", "yes", "y", "true"}:
        return "Yes"
    if text in {"0", "0.0", "no", "n", "false"}:
        return "No"

    try:
        num = float(text)
        if num > 0:
            return "Yes"
        return "No"
    except Exception:
        return "Unknown"


def bin_numeric_value(value: Any, bins: list[float], labels: list[str]) -> str:
    if pd.isna(value):
        return "Unknown"

    try:
        x = float(value)
    except Exception:
        return "Unknown"

    for i, label in enumerate(labels):
        lower = bins[i]
        upper = bins[i + 1]
        if i == 0:
            if x <= upper:
                return label
        else:
            if lower < x <= upper:
                return label

    return "Unknown"


# ============================================================
# LOADING
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

    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    if "First_name" not in df.columns:
        raise ValueError("Metadata Excel must contain 'First_name'")

    df["First_name"] = df["First_name"].astype(str).str.strip().str.upper()
    return df


# ============================================================
# MERGING
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

    parsed = merged["filename"].apply(extract_subject_and_timepoint)
    merged["subject_code"] = [x[0] for x in parsed]
    merged["timepoint"] = [x[1] for x in parsed]

    return merged


def attach_clinical_factors(per_trial_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    factor_keys = ["phenotype", "updrs_iii", "hy", "abc", "gds", "moca", "mmse", "fog_binary", "ledd"]
    metadata_lookup = metadata_df.set_index("First_name", drop=False)

    rows = []

    for _, trial in per_trial_df.iterrows():
        out = trial.to_dict()

        for key in factor_keys:
            out[key] = np.nan

        subject_code = trial.get("subject_code")
        timepoint = trial.get("timepoint")

        if pd.isna(subject_code) or pd.isna(timepoint) or subject_code not in metadata_lookup.index:
            rows.append(out)
            continue

        tp_map = TIMEPOINT_MAP.get(str(timepoint).upper())
        if tp_map is None:
            rows.append(out)
            continue

        meta_row = metadata_lookup.loc[subject_code]

        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]

        for key in factor_keys:
            source_col = tp_map.get(key)
            if source_col is not None and source_col in meta_row.index:
                out[key] = meta_row[source_col]

        rows.append(out)

    df = pd.DataFrame(rows)

    # Normalize numeric factors
    numeric_cols = ["updrs_iii", "hy", "abc", "gds", "moca", "mmse", "ledd"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Grouped categorical factors
    df["phenotype_group"] = df["phenotype"].apply(sanitize_label)

    for col, cfg in NUMERIC_BIN_CONFIG.items():
        df[f"{col}_group"] = df[col].apply(lambda x: bin_numeric_value(x, cfg["bins"], cfg["labels"]))

    # FOG handled separately
    df["fog_binary"] = df["fog_binary"].apply(normalize_fog_value)
    df["fog_group"] = df["fog_binary"]

    return df


# ============================================================
# AGGREGATED TABLE
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

    existing_cols = [c for c in desired_cols if c in pivot.columns]
    return pivot.reindex(columns=existing_cols)


# ============================================================
# STATS
# ============================================================

def get_ordered_levels(df: pd.DataFrame, factor_name: str, group_col: str) -> list[str]:
    levels = [sanitize_label(v) for v in df[group_col].dropna().tolist()]
    levels = list(dict.fromkeys(levels))

    if DROP_UNKNOWN_LEVELS:
        levels = [x for x in levels if x != "Unknown"]

    custom = FACTOR_LEVEL_ORDER.get(factor_name)
    if custom is not None:
        return [x for x in custom if x in levels]

    return sorted(levels)


def summarize_metric_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_levels: list[str],
) -> pd.DataFrame:
    rows = []

    for level in ordered_levels:
        values = pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna()

        rows.append({
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
        })

    return pd.DataFrame(rows)


def pairwise_mannwhitney_bonferroni(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_levels: list[str],
) -> pd.DataFrame:
    valid_results = []
    insufficient = []

    for a, b in combinations(ordered_levels, 2):
        x = pd.to_numeric(df.loc[df[group_col] == a, metric_col], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(df.loc[df[group_col] == b, metric_col], errors="coerce").dropna().to_numpy()

        if len(x) < MIN_SAMPLES_PER_LEVEL or len(y) < MIN_SAMPLES_PER_LEVEL:
            insufficient.append({
                "metric": metric_col,
                "group_1": a,
                "group_2": b,
                "n_1": len(x),
                "n_2": len(y),
                "statistic": np.nan,
                "p_uncorrected": np.nan,
                "p_bonferroni": np.nan,
                "significant": False,
                "note": "Insufficient samples",
            })
            continue

        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        valid_results.append((a, b, len(x), len(y), float(stat), float(p)))

    n_comparisons = len(valid_results)
    rows = []

    for a, b, n1, n2, stat, p in valid_results:
        p_bonf = min(1.0, p * n_comparisons) if n_comparisons > 0 else p
        rows.append({
            "metric": metric_col,
            "group_1": a,
            "group_2": b,
            "n_1": n1,
            "n_2": n2,
            "statistic": stat,
            "p_uncorrected": p,
            "p_bonferroni": p_bonf,
            "significant": bool(p_bonf < ALPHA),
            "note": "",
        })

    rows.extend(insufficient)
    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(by=["group_1", "group_2"]).reset_index(drop=True)

    return out


def run_statistical_tests(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    ordered_levels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    valid_levels = []
    arrays = []

    for level in ordered_levels:
        vals = pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna().to_numpy()
        if len(vals) >= MIN_SAMPLES_PER_LEVEL:
            valid_levels.append(level)
            arrays.append(vals)

    if len(valid_levels) < 2:
        omnibus = pd.DataFrame([{
            "metric": metric_col,
            "test": "not_run",
            "n_levels": len(valid_levels),
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "note": "Fewer than two valid levels",
        }])
        return omnibus, pd.DataFrame(), []

    if len(valid_levels) == 2:
        stat, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        omnibus = pd.DataFrame([{
            "metric": metric_col,
            "test": "mannwhitneyu",
            "n_levels": 2,
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < ALPHA),
            "note": "",
        }])

        posthoc = pd.DataFrame([{
            "metric": metric_col,
            "group_1": valid_levels[0],
            "group_2": valid_levels[1],
            "n_1": len(arrays[0]),
            "n_2": len(arrays[1]),
            "statistic": float(stat),
            "p_uncorrected": float(p),
            "p_bonferroni": float(p),
            "significant": bool(p < ALPHA),
            "note": "",
        }])

        significant_levels = valid_levels.copy() if p < ALPHA else []
        return omnibus, posthoc, significant_levels

    H, p = stats.kruskal(*arrays)
    omnibus = pd.DataFrame([{
        "metric": metric_col,
        "test": "kruskal",
        "n_levels": len(valid_levels),
        "statistic": float(H),
        "p_value": float(p),
        "significant": bool(p < ALPHA),
        "note": "",
    }])

    posthoc = pairwise_mannwhitney_bonferroni(df, group_col, metric_col, valid_levels)

    significant_levels = set()
    if not posthoc.empty:
        sig = posthoc[posthoc["significant"] == True]
        for _, row in sig.iterrows():
            significant_levels.add(row["group_1"])
            significant_levels.add(row["group_2"])

    return omnibus, posthoc, [x for x in valid_levels if x in significant_levels]


# ============================================================
# PLOTTING
# ============================================================

def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=GRAPHICS["grid_alpha"], linestyle=GRAPHICS["grid_linestyle"])
    for spine in ax.spines.values():
        spine.set_linewidth(GRAPHICS["axis_spine_linewidth"])
    ax.tick_params(axis="both", labelsize=GRAPHICS["tick_label_fontsize"])


def get_factor_palette(factor_name: str, n_levels: int) -> list[str]:
    palette = FACTOR_COLORS.get(factor_name, ["#7f8c8d"])
    if len(palette) >= n_levels:
        return palette[:n_levels]
    return [palette[i % len(palette)] for i in range(n_levels)]


def add_n_annotations(ax: plt.Axes, counts: list[int]) -> None:
    for idx, count in enumerate(counts, start=1):
        ax.text(
            idx,
            GRAPHICS["n_text_y_axes_fraction"],
            f"n={count}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=GRAPHICS["n_annotation_fontsize"],
        )


def add_median_annotations(ax: plt.Axes, medians: list[float]) -> None:
    for idx, median in enumerate(medians, start=1):
        if pd.isna(median):
            continue
        ax.text(
            idx + GRAPHICS["annotation_x_offset"],
            median,
            f"{median:.2f}",
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
    ordered_levels: list[str],
    data_arrays: list[np.ndarray],
    significant_levels: list[str],
) -> None:
    if not significant_levels:
        return

    finite = np.concatenate([a[np.isfinite(a)] for a in data_arrays if len(a) > 0]) if data_arrays else np.array([])
    if finite.size == 0:
        return

    y_min = np.min(finite)
    y_max = np.max(finite)
    y_range = max(y_max - y_min, 1e-6)
    offset = GRAPHICS["asterisk_y_offset_fraction"] * y_range

    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], max(ylim[1], y_max + 2.5 * offset))

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
            "", #"*",
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_levels = get_ordered_levels(df, factor_name, group_col)

    if len(ordered_levels) < 2:
        ax.set_title(f"{factor_name}\nNot enough levels", fontsize=GRAPHICS["subplot_title_fontsize"])
        ax.axis("off")
        return pd.DataFrame(), pd.DataFrame()

    data_arrays = [
        pd.to_numeric(df.loc[df[group_col] == level, metric_col], errors="coerce").dropna().to_numpy()
        for level in ordered_levels
    ]

    counts = [len(arr) for arr in data_arrays]
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

    add_n_annotations(ax, counts)
    add_median_annotations(ax, medians)
    add_significance_asterisks(ax, ordered_levels, data_arrays, significant_levels)

    if metric_col == "f1":
        ax.set_ylim(0.7, 1.0)
    elif metric_col == "accuracy":
        ylim = ax.get_ylim()
        ax.set_ylim(min(-0.05, ylim[0]), max(1.05, ylim[1]))

    summary_df = summary_df.copy()
    summary_df["factor"] = factor_name
    summary_df["levels"] = " | ".join(ordered_levels)

    omnibus_df = omnibus_df.copy()
    omnibus_df["factor"] = factor_name
    omnibus_df["levels"] = " | ".join(ordered_levels)

    if not posthoc_df.empty:
        posthoc_df = posthoc_df.copy()
        posthoc_df["factor"] = factor_name
        posthoc_df["levels"] = " | ".join(ordered_levels)
        stats_df = pd.concat([omnibus_df, posthoc_df], ignore_index=True)
    else:
        stats_df = omnibus_df

    return summary_df, stats_df


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

    print("Merging gait and laterality trial-level tables...")
    merged_df = merge_trial_metrics(gait_df, laterality_df)

    print("Attaching clinical factors...")
    analysis_df = attach_clinical_factors(merged_df, metadata_df)

    merged_output_path = OUTPUT_DIR / "merged_trial_level_data.csv"
    analysis_df.to_csv(merged_output_path, index=False)

    aggregated_table = build_aggregated_performance_table(analysis_df)
    aggregated_table.to_csv(AGGREGATED_OUTPUT_CSV)
    print(f"Aggregated performance table saved to: {AGGREGATED_OUTPUT_CSV.resolve()}")

    analysis_df = analysis_df[analysis_df["cohort"] == "PD"].copy()
    if analysis_df.empty:
        raise RuntimeError("No PD trials found after filtering cohort == 'PD'.")

    analysis_df = analysis_df[analysis_df["event_type"].isin(EVENT_TYPES)].copy()
    if analysis_df.empty:
        raise RuntimeError(f"No PD trials found for event types: {EVENT_TYPES}")

    if POOL_EVENT_TYPES:
        print("Pooling IC and FC together for factor analysis.")
    else:
        print("POOL_EVENT_TYPES=False is not implemented separately in this script.")

    print("\nFOG distribution check:")
    if "fog_group" in analysis_df.columns:
        print(analysis_df["fog_group"].value_counts(dropna=False))
    else:
        print("fog_group column not found.")

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