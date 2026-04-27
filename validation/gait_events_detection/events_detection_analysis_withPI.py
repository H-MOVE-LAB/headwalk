from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
ROOT = Path(__file__).resolve().parents[0]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "_analysis_plots"
STEP_DURATION_CSV = Path(__file__).resolve().parents[2] / "data" / "mean_step_duration_per_recording.csv"

COHORT_RE = re.compile(r"ING([CP])", re.IGNORECASE)
FILENAME_RE = re.compile(
    r"^\d{8}-\d{6}-(?P<subject>ING[CP]\d+)(?P<timepoint>F[234])_(?P<task>SC|SI)\.h5$",
    re.IGNORECASE,
)

EXCLUDE_LIST = [
    "INGC106F2_SI",
    "INGC123F2_SI",
    "INGC129F2_SI",
    "INGC129F2_SC",
    "INGC191F2_SI",
    "INGC191F2_SC",
    "INGP069F2_SI",
    "INGP083F2_SI",
    "INGP083F2_SC",
    "INGC144F3_SI",
    "INGC144F3_SC",
    "INGC108F4_SI",
    "INGC108F4_SC",
    "INGC115F4_SI",
    "INGC115F4_SC",
    "INGC125F4_SI",
    "INGC125F4_SC",
    "INGC157F4_SI",
    "INGC157F4_SC",
    "INGP042F4_SC",
    "INGP052F4_SI",
    "INGP055F4_SI",
]


@dataclass(frozen=True)
class TrialMetrics:
    algo: str
    cohort: str
    event_type: str
    filename: str
    f1: float
    mae: float
    missed_percent: float
    extra_percent: float
    sensitivity: float
    ppv: float
    relative_absolute_error: float
    mae_benefit_score: float
    relative_error_benefit_score: float
    performance_index: float
    recording_mean_step_duration_s: float


def _extract_cohort(filename: str) -> str:
    """
    Extract cohort from filename, e.g. "...-INGC116F2_SC.h5" -> "C".
    Returns "U" if not found.
    """
    m = COHORT_RE.search(filename)
    if not m:
        return "U"
    return m.group(1).upper()


def _parse_filename_info(filename: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse filename like:
        20140616-104931-INGC116F2_SC.h5

    Returns:
        First_name: INGC116F2
        Timepoint: F2
        WalkTask: Cont or Interm
    """
    name = Path(filename).name
    m = FILENAME_RE.match(name)
    if not m:
        return None, None, None

    subject = m.group("subject").upper()
    timepoint = m.group("timepoint").upper()
    task_code = m.group("task").upper()

    first_name = f"{subject}{timepoint}"
    walktask = {"SC": "Cont", "SI": "Interm"}[task_code]

    return first_name, timepoint, walktask


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return float((2 * tp) / denom)


def _safe_sensitivity(tp: int, fn: int) -> float:
    denom = tp + fn
    if denom <= 0:
        return float("nan")
    return float(tp / denom)


def _safe_ppv(tp: int, fp: int) -> float:
    denom = tp + fp
    if denom <= 0:
        return float("nan")
    return float(tp / denom)


def _cost_to_benefit_score(x: float) -> float:
    """
    Cost metric -> benefit-like score in [0, 1], using:
        x_norm = 1 - exp(-x)
        flipped_score = 1 - x_norm = exp(-x)
    """
    if not np.isfinite(x):
        return float("nan")
    return float(np.exp(-x))


def _safe_performance_index(
    sensitivity: float,
    ppv: float,
    mae: float,
    relative_absolute_error: float,
) -> tuple[float, float, float]:
    """
    Benefit metrics are used as-is:
      - sensitivity
      - ppv

    Cost metrics are transformed as:
      x_norm = 1 - exp(-x)
      benefit_score = 1 - x_norm = exp(-x)

    Final weighted index:
      sensitivity * 0.237
      + ppv * 0.229
      + mae_benefit_score * 0.267
      + relative_error_benefit_score * 0.267
    """
    mae_benefit_score = _cost_to_benefit_score(mae)
    relative_error_benefit_score = _cost_to_benefit_score(relative_absolute_error)

    if not (
        np.isfinite(sensitivity)
        and np.isfinite(ppv)
        and np.isfinite(mae_benefit_score)
        and np.isfinite(relative_error_benefit_score)
    ):
        return float("nan"), mae_benefit_score, relative_error_benefit_score

    performance_index = (
        sensitivity * 0.237
        + ppv * 0.229
        + mae_benefit_score * 0.267
        + relative_error_benefit_score * 0.267
    )

    return float(performance_index), mae_benefit_score, relative_error_benefit_score


def _load_recording_step_duration_table(step_duration_csv: Path) -> pd.DataFrame:
    """
    Load mean step duration per recording.
    The mean_step_duration in the CSV is assumed to be in milliseconds and is converted to seconds.
    """
    df = pd.read_csv(step_duration_csv)
    df.columns = [c.strip() for c in df.columns]

    required = {"First_name", "Group", "WalkTask", "Timepoint", "mean_step_duration", "n_steps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{step_duration_csv} is missing columns: {missing}")

    df["First_name"] = df["First_name"].astype(str).str.strip().str.upper()
    df["Group"] = df["Group"].astype(str).str.strip().str.upper()
    df["WalkTask"] = df["WalkTask"].astype(str).str.strip()
    df["Timepoint"] = df["Timepoint"].astype(str).str.strip().str.upper()
    df["mean_step_duration"] = pd.to_numeric(df["mean_step_duration"], errors="coerce")
    df["n_steps"] = pd.to_numeric(df["n_steps"], errors="coerce")

    df = df.dropna(
        subset=["First_name", "Group", "WalkTask", "Timepoint", "mean_step_duration", "n_steps"]
    ).copy()

    # Convert ms -> s
    df["mean_step_duration"] = df["mean_step_duration"] / 1000.0

    # One row per recording should already exist, but keep a safe groupby in case of duplicates
    result = (
        df.groupby(["First_name", "Group", "WalkTask", "Timepoint"], as_index=False)
        .agg(
            recording_mean_step_duration_s=("mean_step_duration", "mean"),
            recording_n_steps=("n_steps", "sum"),
        )
    )

    return result


def _compute_metrics_for_algo(
    matches_csv: Path,
    algo_name: str,
    recording_step_duration_df: pd.DataFrame,
) -> list[TrialMetrics]:
    """
    Compute per-trial metrics for one algorithm, separately for IC and FC.
    """
    df = pd.read_csv(matches_csv)
    df.columns = [c.strip() for c in df.columns]

    required = {"status", "type", "filename", "diff", "time_pred", "time_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{matches_csv} is missing columns: {missing}")

    df["filename"] = df["filename"].astype(str)
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df["type"] = df["type"].astype(str).str.upper().str.strip()
    df["diff"] = pd.to_numeric(df["diff"], errors="coerce")

    df["cohort"] = df["filename"].map(_extract_cohort)

    parsed = df["filename"].map(_parse_filename_info)
    df["First_name"] = [p[0] for p in parsed]
    df["Timepoint"] = [p[1] for p in parsed]
    df["WalkTask"] = [p[2] for p in parsed]

    df["First_name"] = df["First_name"].astype(str).str.strip().str.upper()
    df["Timepoint"] = df["Timepoint"].astype(str).str.strip().str.upper()
    df["WalkTask"] = df["WalkTask"].astype(str).str.strip()

    cohort_map = {"C": "C", "P": "P", "U": np.nan}
    df["Group"] = df["cohort"].map(cohort_map)

    df = df.merge(
        recording_step_duration_df,
        how="left",
        on=["First_name", "Group", "WalkTask", "Timepoint"],
    )

    out: list[TrialMetrics] = []

    for event_type in ["IC", "FC"]:
        df_t = df[df["type"] == event_type].copy()
        if df_t.empty:
            continue

        for (fname, cohort), g in df_t.groupby(["filename", "cohort"], dropna=False):
            if any(e in fname for e in EXCLUDE_LIST):
                continue

            status = g["status"]

            tp = int((status == "TP").sum())
            fp = int((status == "FP").sum())
            fn = int((status == "FN").sum())

            f1 = _safe_f1(tp=tp, fp=fp, fn=fn)
            sensitivity = _safe_sensitivity(tp=tp, fn=fn)
            ppv = _safe_ppv(tp=tp, fp=fp)

            ref_count = tp + fn
            missed_percent = float(fn / ref_count * 100.0) if ref_count > 0 else float("nan")
            extra_percent = float(fp / ref_count * 100.0) if ref_count > 0 else float("nan")

            g_tp = g[status == "TP"].copy()

            if not g_tp.empty:
                diff = g_tp["diff"].to_numpy(dtype=float)
                mae = float(np.nanmean(np.abs(diff))) if np.isfinite(diff).any() else float("nan")

                rec_step = pd.to_numeric(
                    g_tp["recording_mean_step_duration_s"], errors="coerce"
                ).to_numpy(dtype=float)
                recording_mean_step_duration_s = (
                    float(np.nanmean(rec_step)) if np.isfinite(rec_step).any() else float("nan")
                )

                if (
                    np.isfinite(mae)
                    and np.isfinite(recording_mean_step_duration_s)
                    and recording_mean_step_duration_s > 0
                ):
                    relative_absolute_error = float(mae / recording_mean_step_duration_s)
                else:
                    relative_absolute_error = float("nan")
            else:
                mae = float("nan")
                f1 = float("nan")
                sensitivity = float("nan")
                ppv = float("nan")
                missed_percent = float("nan")
                extra_percent = float("nan")
                relative_absolute_error = float("nan")
                recording_mean_step_duration_s = float("nan")

            performance_index, mae_benefit_score, relative_error_benefit_score = _safe_performance_index(
                sensitivity=sensitivity,
                ppv=ppv,
                mae=mae,
                relative_absolute_error=relative_absolute_error,
            )

            out.append(
                TrialMetrics(
                    algo=algo_name,
                    cohort=cohort,
                    event_type=event_type,
                    filename=str(fname),
                    f1=float(f1),
                    mae=float(mae),
                    missed_percent=float(missed_percent),
                    extra_percent=float(extra_percent),
                    sensitivity=float(sensitivity),
                    ppv=float(ppv),
                    relative_absolute_error=float(relative_absolute_error),
                    mae_benefit_score=float(mae_benefit_score),
                    relative_error_benefit_score=float(relative_error_benefit_score),
                    performance_index=float(performance_index),
                    recording_mean_step_duration_s=float(recording_mean_step_duration_s),
                )
            )

    return out


def _discover_algorithms(results_dir: Path) -> list[tuple[str, Path]]:
    """
    Find all results/<algo_name>/matches_ss_optimized.csv.
    """
    algos: list[tuple[str, Path]] = []
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for algo_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        matches = algo_dir / "matches_ss_optimized.csv"
        if matches.exists():
            algos.append((algo_dir.name, matches))

    if not algos:
        raise FileNotFoundError(
            f"No matches.csv found under: {results_dir}/<algo_name>/matches_ss_optimized.csv"
        )
    return algos


def _generate_color_map(algos: list[str]) -> dict[str, tuple]:
    """
    Assign a consistent color to each algorithm.
    """
    cmap = plt.get_cmap("tab10")
    color_map = {}
    for i, algo in enumerate(algos):
        color_map[algo] = cmap(i % 10)
    return color_map


def _boxplot_by_algo(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    algos_order: list[str],
    color_map: dict[str, tuple],
    title: str,
    ylabel: str,
) -> list[str]:

    data = []
    labels = []
    colors = []

    for algo in algos_order:
        vals = df.loc[df["algo"] == algo, metric_col].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        data.append(vals)
        labels.append(algo)
        colors.append(color_map[algo])

    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return []

    box = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    return labels


def _plot_cohort_event(
    metrics_df: pd.DataFrame,
    *,
    cohort: str,
    event_type: str,
    algos_order: list[str],
    out_dir: Path,
) -> Path:

    df = metrics_df[
        (metrics_df["cohort"] == cohort) &
        (metrics_df["event_type"] == event_type)
    ].copy()

    color_map = _generate_color_map(algos_order)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    cohort_name = (
        "Controls (C)" if cohort == "C"
        else "Parkinson (P)" if cohort == "P"
        else f"Cohort {cohort}"
    )

    fig.suptitle(
        f"{event_type} — {cohort_name} — Per-trial distributions",
        fontsize=16
    )

    plotted_f1 = _boxplot_by_algo(
        ax1, df, "f1", algos_order, color_map,
        title="F1-score distribution (per trial)",
        ylabel="F1",
    )
    if plotted_f1:
        ax1.set_ylim([-0.05, 1.05])

    _boxplot_by_algo(
        ax2, df, "mae", algos_order, color_map,
        title="Mean Absolute Error distribution (TP only, per trial)",
        ylabel="MAE (s)",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_type}_{cohort}_boxplots_ss_optimized.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


# ----------------------------
# Publication-ready combined plots
# ----------------------------
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COHORT_COLORS = {
    "C": (0.121, 0.466, 0.705),
    "P": (1.000, 0.498, 0.054),
}


def _collect_algo_cohort_vals(df: pd.DataFrame, algos_order: list[str], metric: str, event_type: str):
    cols_C = []
    cols_P = []
    has_any = []
    for algo in algos_order:
        sub = df[(df["algo"] == algo) & (df["event_type"] == event_type)]
        vals_C = sub[sub["cohort"] == "C"][metric].dropna().astype(float).to_numpy()
        vals_P = sub[sub["cohort"] == "P"][metric].dropna().astype(float).to_numpy()
        cols_C.append(vals_C)
        cols_P.append(vals_P)
        has_any.append((len(vals_C) > 0) or (len(vals_P) > 0))
    return cols_C, cols_P, has_any


def _plot_grouped_boxplots(
    ax: plt.Axes,
    algos: list[str],
    cols_C: list[np.ndarray],
    cols_P: list[np.ndarray],
    colors: dict[str, tuple],
    ylabel: str,
    show_ylim: tuple | None = None,
    add_jitter: bool = True,
):
    n = len(algos)
    spacing = 3.0
    offset = 0.4

    pos_C = []
    pos_P = []
    data_for_box = []
    positions = []
    labels_positions = []
    labels = []

    for i in range(n):
        base = i * spacing
        if len(cols_C[i]) > 0:
            pos_c = base - offset
            pos_C.append(pos_c)
            data_for_box.append(cols_C[i])
            positions.append(pos_c)
        if len(cols_P[i]) > 0:
            pos_p = base + offset
            pos_P.append(pos_p)
            data_for_box.append(cols_P[i])
            positions.append(pos_p)

        labels_positions.append(base)
        labels.append(algos[i])

    if not data_for_box:
        ax.text(0.5, 0.5, "No data for this event type", ha="center", va="center")
        ax.set_axis_off()
        return

    box = ax.boxplot(
        data_for_box,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
    )

    for patch, pos in zip(box["boxes"], positions):
        if any(abs(pos - p) < 1e-6 for p in pos_C):
            patch.set_facecolor(colors["C"])
        else:
            patch.set_facecolor(colors["P"])
        patch.set_alpha(0.7)

    if add_jitter:
        rng = np.random.default_rng(42)
        for i in range(n):
            base = i * spacing
            if len(cols_C[i]) > 0:
                xpos = base - offset
                y = cols_C[i]
                jitter = rng.normal(scale=0.08, size=len(y))
                ax.scatter(
                    np.full(len(y), xpos) + jitter, y,
                    color=colors["C"], alpha=0.6, s=10, edgecolors="none"
                )
            if len(cols_P[i]) > 0:
                xpos = base + offset
                y = cols_P[i]
                jitter = rng.normal(scale=0.08, size=len(y))
                ax.scatter(
                    np.full(len(y), xpos) + jitter, y,
                    color=colors["P"], alpha=0.6, s=10, edgecolors="none"
                )

    ax.set_xticks([i * spacing for i in range(n)])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlim(-spacing * 0.6, spacing * (n - 1) + spacing * 0.6)

    if show_ylim is not None:
        ax.set_ylim(show_ylim)

    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel(ylabel)


def _plot_event_combined(
    metrics_df: pd.DataFrame,
    *,
    event_type: str,
    algos_order: list[str],
    out_dir: Path,
):
    df = metrics_df[metrics_df["event_type"] == event_type].copy()

    cols_C_f1, cols_P_f1, _ = _collect_algo_cohort_vals(df, algos_order, "f1", event_type)
    cols_C_mae, cols_P_mae, _ = _collect_algo_cohort_vals(df, algos_order, "mae", event_type)

    algos_shown = algos_order[:]

    if not algos_shown:
        print(f"No data for event_type={event_type} for any algorithm. Skipping plot.")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, 1.4 * len(algos_shown)), 8), sharex=True)
    fig.suptitle(f"{event_type} — Per-trial distributions (Control vs Parkinson)", fontsize=14)

    _plot_grouped_boxplots(
        ax1,
        algos_shown,
        cols_C_f1,
        cols_P_f1,
        colors=COHORT_COLORS,
        ylabel="F1",
        show_ylim=(-0.05, 1.05),
        add_jitter=True,
    )
    ax1.set_title("F1-score (per trial) — Control (blue) vs Parkinson (orange)")

    _plot_grouped_boxplots(
        ax2,
        algos_shown,
        cols_C_mae,
        cols_P_mae,
        colors=COHORT_COLORS,
        ylabel="MAE (s)",
        show_ylim=None,
        add_jitter=True,
    )
    ax2.set_title("MAE on matched events (TP only) — Control (blue) vs Parkinson (orange)")

    handles = [
        mpl.patches.Patch(facecolor=COHORT_COLORS["C"], label="Control (C)", alpha=0.7),
        mpl.patches.Patch(facecolor=COHORT_COLORS["P"], label="Parkinson (P)", alpha=0.7),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.96, 0.92))

    plt.tight_layout(rect=[0, 0.01, 0.94, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{event_type}_Control_vs_PD_boxplots_ss_optimized.png"
    out_pdf = out_dir / f"{event_type}_Control_vs_PD_boxplots_ss_optimized.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)

    return out_png, out_pdf


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    recording_step_duration_df = _load_recording_step_duration_table(STEP_DURATION_CSV)

    algos = _discover_algorithms(RESULTS_DIR)
    algo_names = [a for a, _ in algos]

    all_metrics: list[TrialMetrics] = []
    for algo_name, matches_path in algos:
        algo_metrics = _compute_metrics_for_algo(
            matches_csv=matches_path,
            algo_name=algo_name,
            recording_step_duration_df=recording_step_duration_df,
        )
        all_metrics.extend(algo_metrics)

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])

    # -----------------------------
    # Print Performance Index (IC only) per algorithm
    # -----------------------------
    print("\nPerformance Index (IC only) per algorithm:")

    df_ic = metrics_df[metrics_df["event_type"] == "IC"].copy()

    for algo in sorted(df_ic["algo"].unique()):
        vals = df_ic.loc[df_ic["algo"] == algo, "performance_index"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            pi_mean = float("nan")
        else:
            pi_mean = float(np.mean(vals))

        print(f"{algo}: PI = {pi_mean:.4f}")

    # Keep only cohorts C and P for plotting
    metrics_df = metrics_df[metrics_df["cohort"].isin(["C", "P"])].copy()

    # Save aggregated table
    summary_csv = PLOTS_DIR / "per_trial_metrics_ss_optimized.csv"
    metrics_df.to_csv(summary_csv, index=False)

    # Save one metrics CSV per algorithm with the new columns added
    for algo_name, _ in algos:
        algo_df = metrics_df[metrics_df["algo"] == algo_name].copy()
        algo_metrics_csv = RESULTS_DIR / algo_name / "metrics_per_trial_ss_optimized.csv"
        algo_df.to_csv(algo_metrics_csv, index=False)

    for event_type in ["IC", "FC"]:
        out = _plot_event_combined(
            metrics_df,
            event_type=event_type,
            algos_order=algo_names,
            out_dir=PLOTS_DIR,
        )

        if out is not None:
            png_path, pdf_path = out
            print(f"Saved: {png_path}")
            print(f"Saved: {pdf_path}")

    print("\nResults analysis completed.")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()