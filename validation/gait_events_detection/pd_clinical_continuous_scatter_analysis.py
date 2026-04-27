"""
pd_clinical_continuous_scatter_analysis.py

PD-only clinical factor analysis using continuous clinical variables.

This script:
- loads per-trial gait event metrics and laterality metrics
- merges them with clinical metadata
- keeps PD trials only
- treats selected clinical factors as continuous variables
- creates paper-ready 3x3 figures per algorithm:
    1) gait event detection (F1 score)
    2) laterality detection (accuracy)
- each subplot shows:
    * one dot per trial
    * linear regression line
    * Spearman correlation coefficient in the title
    * significance status in the title
- saves merged tables and correlation result CSV files

Notes:
- Spearman correlation is used for inference/association reporting
- Linear regression line is displayed only as a visual trend line
"""

from __future__ import annotations

import re
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

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "results" / "_pd_clinical_continuous_scatter_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS_TO_ANALYZE = ["Fawden", "TCN"]
EVENT_TYPES = ["IC", "FC"]
POOL_EVENT_TYPES = True

ALPHA = 0.001
MIN_SAMPLES_PER_FACTOR = 3


# ============================================================
# GRAPHIC STYLE
# ============================================================

GRAPHICS = {
    "figure_size": (20, 16),
    "figure_dpi": 300,
    "subplot_wspace": 0.28,
    "subplot_hspace": 0.40,
    "suptitle_fontsize": 22,
    "subplot_title_fontsize": 12,
    "axis_label_fontsize": 12,
    "tick_label_fontsize": 10,
    "annotation_fontsize": 10,
    "font_family": "sans-serif",
    "grid_alpha": 0.20,
    "grid_linestyle": "--",
    "axis_spine_linewidth": 1.0,
    "point_size": 28,
    "point_alpha": 0.45,
    "line_width": 2.0,
    "title_pad": 10,
}

plt.rcParams["font.family"] = GRAPHICS["font_family"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# COLORS
# Uncomment alternatives if you want to quickly try other styles
# ============================================================

POINT_COLOR = "#6BAED6"     # light blue
REGRESSION_LINE_COLOR = "#000000"  # black
TITLE_COLOR = "#111111"
AXIS_LABEL_COLOR = "#111111"
GRID_COLOR = "#BDBDBD"

# Alternative point colors:
# POINT_COLOR = "#4C78A8"   # muted blue
# POINT_COLOR = "#76B7B2"   # teal
# POINT_COLOR = "#A0CBE8"   # paler blue
# POINT_COLOR = "#9ECAE1"   # soft sky blue
# POINT_COLOR = "#B3CDE3"   # pastel blue

# Alternative regression line colors:
# REGRESSION_LINE_COLOR = "#222222"  # dark gray
# REGRESSION_LINE_COLOR = "#444444"  # gray
# REGRESSION_LINE_COLOR = "#1B1B1B"  # almost black
# REGRESSION_LINE_COLOR = "#C44E52"  # muted red
# REGRESSION_LINE_COLOR = "#2F4B7C"  # dark blue

# Alternative grid colors:
# GRID_COLOR = "#D9D9D9"
# GRID_COLOR = "#CCCCCC"
# GRID_COLOR = "#E0E0E0"


# ============================================================
# CONTINUOUS FACTORS
# ============================================================

CONTINUOUS_FACTORS = [
    {"name": "UPDRS III", "key": "updrs_iii", "xlabel": "UPDRS III"},
    {"name": "Hohen & Yahr", "key": "hy", "xlabel": "Hohen & Yahr"},
    {"name": "ABC", "key": "abc", "xlabel": "ABC"},
    {"name": "GDS", "key": "gds", "xlabel": "GDS"},
    {"name": "MoCA", "key": "moca", "xlabel": "MoCA"},
    {"name": "MMSE", "key": "mmse", "xlabel": "MMSE"},
    {"name": "FOG", "key": "fog_binary_numeric", "xlabel": "FOG (0 = No, 1 = Yes)"},
    {"name": "LEDD", "key": "ledd", "xlabel": "LEDD"},
]

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


def normalize_fog_numeric(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text in {"1", "1.0", "yes", "y", "true"}:
        return 1.0
    if text in {"0", "0.0", "no", "n", "false"}:
        return 0.0

    try:
        num = float(text)
        return 1.0 if num > 0 else 0.0
    except Exception:
        return np.nan


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
    factor_keys = ["updrs_iii", "hy", "abc", "gds", "moca", "mmse", "fog_binary", "ledd"]
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

    numeric_cols = ["updrs_iii", "hy", "abc", "gds", "moca", "mmse", "ledd"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fog_binary_numeric"] = df["fog_binary"].apply(normalize_fog_numeric)

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
# PLOTTING / STATS
# ============================================================

def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=GRAPHICS["grid_alpha"], linestyle=GRAPHICS["grid_linestyle"], color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(GRAPHICS["axis_spine_linewidth"])
    ax.tick_params(axis="both", labelsize=GRAPHICS["tick_label_fontsize"])
    ax.set_facecolor("white")


def compute_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def create_scatter_subplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    factor_name: str,
    factor_col: str,
    metric_col: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any]:
    sub = df[[factor_col, metric_col]].copy()
    sub[factor_col] = pd.to_numeric(sub[factor_col], errors="coerce")
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.dropna()

    result = {
        "factor": factor_name,
        "metric": metric_col,
        "n": int(len(sub)),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "significant": False,
        "slope": np.nan,
        "intercept": np.nan,
    }

    if len(sub) < MIN_SAMPLES_PER_FACTOR or sub[factor_col].nunique() < 2 or sub[metric_col].nunique() < 2:
        ax.set_title(f"{factor_name}\nNot enough variation", fontsize=GRAPHICS["subplot_title_fontsize"], pad=GRAPHICS["title_pad"])
        ax.axis("off")
        return result

    x = sub[factor_col].to_numpy(dtype=float)
    y = sub[metric_col].to_numpy(dtype=float)

    rho, p = compute_spearman(x, y)
    significant = bool(p < ALPHA)

    slope, intercept = fit_linear_regression(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = slope * x_line + intercept

    ax.scatter(
        x,
        y,
        s=GRAPHICS["point_size"],
        color=POINT_COLOR,
        alpha=GRAPHICS["point_alpha"],
        edgecolors="none",
        zorder=2,
    )

    ax.plot(
        x_line,
        y_line,
        color=REGRESSION_LINE_COLOR,
        linewidth=GRAPHICS["line_width"],
        zorder=3,
    )

    sig_text = "significant" if significant else "not significant"
    title = f"{factor_name}\nSpearman rho = {rho:.2f}, p = {p:.3g} ({sig_text})"
    ax.set_title(title, fontsize=GRAPHICS["subplot_title_fontsize"], color=TITLE_COLOR, pad=GRAPHICS["title_pad"])

    ax.set_xlabel(x_label, fontsize=GRAPHICS["axis_label_fontsize"], color=AXIS_LABEL_COLOR)
    ax.set_ylabel(y_label, fontsize=GRAPHICS["axis_label_fontsize"], color=AXIS_LABEL_COLOR)

    style_axes(ax)

    if metric_col == "f1":
        ax.set_ylim(0.7, 1.0)
    elif metric_col == "accuracy":
        ax.set_ylim(-0.05, 1.05)

    result.update({
        "spearman_rho": rho,
        "spearman_p": p,
        "significant": significant,
        "slope": slope,
        "intercept": intercept,
    })

    return result


def create_metric_figure_for_algorithm(
    df_algorithm_pd: pd.DataFrame,
    algorithm_name: str,
    metric_col: str,
    metric_title: str,
    y_label: str,
    output_png: Path,
    output_pdf: Path,
    output_csv: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=GRAPHICS["figure_size"], dpi=GRAPHICS["figure_dpi"])
    fig.suptitle(
        f"{algorithm_name} - {metric_title} - PD trials",
        fontsize=GRAPHICS["suptitle_fontsize"],
    )

    results = []

    axes_flat = axes.flatten()

    for i, factor_cfg in enumerate(CONTINUOUS_FACTORS):
        ax = axes_flat[i]

        result = create_scatter_subplot(
            ax=ax,
            df=df_algorithm_pd,
            factor_name=factor_cfg["name"],
            factor_col=factor_cfg["key"],
            metric_col=metric_col,
            x_label=factor_cfg["xlabel"],
            y_label=y_label,
        )
        result["algorithm"] = algorithm_name
        result["metric_panel"] = metric_title
        results.append(result)

    # Turn off unused subplot(s)
    for j in range(len(CONTINUOUS_FACTORS), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.subplots_adjust(
        wspace=GRAPHICS["subplot_wspace"],
        hspace=GRAPHICS["subplot_hspace"],
    )

    fig.savefig(output_png, dpi=GRAPHICS["figure_dpi"], bbox_inches="tight")
    fig.savefig(output_pdf, dpi=GRAPHICS["figure_dpi"], bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(results).to_csv(output_csv, index=False)


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
        print("Pooling IC and FC together for continuous-factor analysis.")
    else:
        print("POOL_EVENT_TYPES=False is not separately implemented in this script.")

    print("\nFOG numeric distribution check:")
    if "fog_binary_numeric" in analysis_df.columns:
        print(analysis_df["fog_binary_numeric"].value_counts(dropna=False))
    else:
        print("fog_binary_numeric column not found.")

    for algorithm_name in ALGORITHMS_TO_ANALYZE:
        df_algorithm = analysis_df[analysis_df["algo"] == algorithm_name].copy()

        if df_algorithm.empty:
            print(f"[SKIP] No PD trials found for algorithm: {algorithm_name}")
            continue

        algorithm_dir = OUTPUT_DIR / algorithm_name
        ensure_dir(algorithm_dir)

        df_algorithm.to_csv(algorithm_dir / "merged_pd_trials_used.csv", index=False)

        gait_fig_png = algorithm_dir / f"{algorithm_name}_gait_events_detection_scatter_3x3.png"
        gait_fig_pdf = algorithm_dir / f"{algorithm_name}_gait_events_detection_scatter_3x3.pdf"
        gait_csv = algorithm_dir / f"{algorithm_name}_gait_events_detection_correlations.csv"

        laterality_fig_png = algorithm_dir / f"{algorithm_name}_laterality_detection_scatter_3x3.png"
        laterality_fig_pdf = algorithm_dir / f"{algorithm_name}_laterality_detection_scatter_3x3.pdf"
        laterality_csv = algorithm_dir / f"{algorithm_name}_laterality_detection_correlations.csv"

        create_metric_figure_for_algorithm(
            df_algorithm_pd=df_algorithm,
            algorithm_name=algorithm_name,
            metric_col="f1",
            metric_title="Gait Event Detection",
            y_label="F1 score",
            output_png=gait_fig_png,
            output_pdf=gait_fig_pdf,
            output_csv=gait_csv,
        )

        create_metric_figure_for_algorithm(
            df_algorithm_pd=df_algorithm,
            algorithm_name=algorithm_name,
            metric_col="accuracy",
            metric_title="Laterality Detection",
            y_label="Laterality accuracy",
            output_png=laterality_fig_png,
            output_pdf=laterality_fig_pdf,
            output_csv=laterality_csv,
        )

        print(f"[SAVED] {algorithm_name}")
        print(f"        {gait_fig_png}")
        print(f"        {laterality_fig_png}")

    print("\nAll analyses completed.")
    print(f"Results saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()