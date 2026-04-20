from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIGURATION
# ============================================================

BASE_DIR = Path("results/_analysis_plots")

GAIT_CSV_PATH = BASE_DIR / "per_trial_metrics_ss_optimized.csv"
LATERALITY_CSV_PATH = BASE_DIR / "per_trial_laterality.csv"
OUTPUT_CSV_PATH = BASE_DIR / "aggregated_performance_table.csv"

# Metrics are defined here so they can be easily added or removed.
# Each item specifies:
# - source: which CSV contains the metric
# - input_column: metric column in the input CSV
# - output_label: column name to show in the final table
METRICS_CONFIG = [
    {"source": "gait", "input_column": "f1", "output_label": "F1"},
    {"source": "gait", "input_column": "mae", "output_label": "MAE"},
    {"source": "laterality", "input_column": "accuracy", "output_label": "Side Accuracy"},
]

# Desired order in the final table
EVENT_TYPE_ORDER = ["IC", "FC"]
COHORT_ORDER = ["PD", "Control"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def standardize_cohort(series: pd.Series) -> pd.Series:
    """
    Standardize cohort labels so that common variants map to:
    - PD
    - Control
    """
    mapping = {
        "pd": "PD",
        "p": "PD",
        "control": "Control",
        "c": "Control",
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(series.astype(str).str.strip())
    )


def standardize_event_type(series: pd.Series) -> pd.Series:
    """
    Standardize event_type labels to uppercase (e.g., IC, FC).
    """
    return series.astype(str).str.strip().str.upper()


def format_mean_loa(values: pd.Series) -> str:
    """
    Return a string formatted as:
    mean [lower, upper]

    Limits of agreement are computed as:
    mean ± 1.96 * SD

    If fewer than 2 valid samples are available, lower and upper are set to NaN.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
        return ""

    mean_value = values.mean()

    if len(values) < 2:
        lower = np.nan
        upper = np.nan
    else:
        std = values.std()
        se = std / np.sqrt(len(values))
        lower = mean_value - 1.96 * se
        upper = mean_value + 1.96 * se

    if np.isnan(lower) or np.isnan(upper):
        return f"{mean_value:.4f} [nan, nan]"

    return f"{mean_value:.4f} [{lower:.4f}, {upper:.4f}]"


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV and standardize shared grouping columns.
    """
    df = pd.read_csv(csv_path)

    required_columns = {"algo", "cohort", "event_type", "filename"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}"
        )

    df["algo"] = df["algo"].astype(str).str.strip()
    df["cohort"] = standardize_cohort(df["cohort"])
    df["event_type"] = standardize_event_type(df["event_type"])

    return df


def aggregate_metric(
    df: pd.DataFrame,
    metric_column: str,
    output_label: str,
) -> pd.DataFrame:
    """
    Aggregate one metric by:
    algo, cohort, event_type

    Returns a pivoted DataFrame with:
    index   = [algo, cohort]
    columns = event_type
    values  = formatted mean [lower, upper]
    """
    if metric_column not in df.columns:
        raise ValueError(f"Column '{metric_column}' not found in input DataFrame.")

    grouped = (
        df.groupby(["algo", "cohort", "event_type"], dropna=False)[metric_column]
        .apply(format_mean_loa)
        .reset_index(name=output_label)
    )

    pivoted = grouped.pivot_table(
        index=["algo", "cohort"],
        columns="event_type",
        values=output_label,
        aggfunc="first",
    )

    # Ensure consistent column order
    pivoted = pivoted.reindex(columns=EVENT_TYPE_ORDER)

    # Convert single-level columns into MultiIndex: (event_type, metric_name)
    pivoted.columns = pd.MultiIndex.from_tuples(
        [(event_type, output_label) for event_type in pivoted.columns]
    )

    return pivoted


def build_final_table(
    gait_df: pd.DataFrame,
    laterality_df: pd.DataFrame,
    metrics_config: list[dict],
) -> pd.DataFrame:
    """
    Build the final summary table by combining all configured metrics.
    """
    metric_tables = []

    for metric in metrics_config:
        source = metric["source"]
        input_column = metric["input_column"]
        output_label = metric["output_label"]

        if source == "gait":
            metric_df = aggregate_metric(gait_df, input_column, output_label)
        elif source == "laterality":
            metric_df = aggregate_metric(laterality_df, input_column, output_label)
        else:
            raise ValueError(f"Unknown source '{source}' in METRICS_CONFIG.")

        metric_tables.append(metric_df)

    if not metric_tables:
        raise ValueError("No metrics defined in METRICS_CONFIG.")

    final_table = pd.concat(metric_tables, axis=1)

    # Build complete row index so each algorithm has both PD and Control rows
    all_algorithms = sorted(
        set(gait_df["algo"].dropna().unique()).union(
            set(laterality_df["algo"].dropna().unique())
        )
    )

    full_index = pd.MultiIndex.from_product(
        [all_algorithms, COHORT_ORDER],
        names=["Algorithm", "Cohort"]
    )

    final_table.index = final_table.index.set_names(["Algorithm", "Cohort"])
    final_table = final_table.reindex(full_index)

    # Build desired column order: IC/FC on the outside, metrics inside
    metric_order = [item["output_label"] for item in metrics_config]
    desired_columns = pd.MultiIndex.from_product(
        [EVENT_TYPE_ORDER, metric_order]
    )
    final_table = final_table.reindex(columns=desired_columns)

    return final_table


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    gait_df = prepare_dataframe(GAIT_CSV_PATH)
    laterality_df = prepare_dataframe(LATERALITY_CSV_PATH)

    final_table = build_final_table(gait_df, laterality_df, METRICS_CONFIG)

    print("\nAggregated performance table:\n")
    print(final_table.to_string())

    final_table.to_csv(OUTPUT_CSV_PATH)
    print(f"\nCSV saved to: {Path(OUTPUT_CSV_PATH).resolve()}")


if __name__ == "__main__":
    main()