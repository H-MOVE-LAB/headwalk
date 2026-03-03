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

COHORT_RE = re.compile(r"ING([CP])", re.IGNORECASE)  # INGC... or INGP...
EXCLUDE_LIST = [
    "INGC123F2_SI",
    "INGC128F2_SI",
    "INGC128F2_SC",
    "INGC129F2_SI",
    "INGC129F2_SC",
    "INGC170F2_SI",
    "INGC191F2_SI",
    "INGC191F2_SC",
    "INGP069F2_SI",
    "INGP080F2_SI",
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
    "INGP055F4_SI",
]

@dataclass(frozen=True)
class TrialMetrics:
    algo: str
    cohort: str  # "C" or "P"
    event_type: str  # "IC" or "FC"
    filename: str
    f1: float
    mae: float  # NaN if no TP


def _extract_cohort(filename: str) -> str:
    """
    Extract cohort from filename, e.g. "...-INGC116F2_SC.h5" -> "C".
    Returns "U" if not found.
    """
    m = COHORT_RE.search(filename)
    if not m:
        return "U"
    return m.group(1).upper()


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    """
    Robust F1. If precision or recall undefined, returns 0.0.
    """
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return float((2 * tp) / denom)


def _compute_metrics_for_algo(matches_csv: Path, algo_name: str) -> list[TrialMetrics]:
    """
    Compute per-trial metrics for one algorithm, separately for IC and FC.
    """
    df = pd.read_csv(matches_csv)

    # Normalize columns just in case
    df.columns = [c.strip() for c in df.columns]

    required = {"status", "type", "filename", "diff", "time_pred", "time_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{matches_csv} is missing columns: {missing}")

    # Extract cohort per row, then we aggregate per trial
    df["cohort"] = df["filename"].astype(str).map(_extract_cohort)

    out: list[TrialMetrics] = []

    for event_type in ["IC", "FC"]:
        df_t = df[df["type"] == event_type].copy()
        if df_t.empty:
            continue

        # Group by trial (filename) and cohort
        for (fname, cohort), g in df_t.groupby(["filename", "cohort"], dropna=False):
            if any([e in fname for e in EXCLUDE_LIST]):
                continue
            status = g["status"].astype(str).str.upper()

            tp = int((status == "TP").sum())
            fp = int((status == "FP").sum())
            fn = int((status == "FN").sum())

            f1 = _safe_f1(tp=tp, fp=fp, fn=fn)

            # MAE only on matched events (TP) with valid diff
            g_tp = g[status == "TP"]
            if not g_tp.empty:
                # diff is signed pred-ref; MAE uses abs
                diff = pd.to_numeric(g_tp["diff"], errors="coerce").to_numpy(dtype=float)
                mae = float(np.nanmean(np.abs(diff))) if np.isfinite(diff).any() else float("nan")
            else:
                mae = float("nan")

            out.append(
                TrialMetrics(
                    algo=algo_name,
                    cohort=cohort,
                    event_type=event_type,
                    filename=str(fname),
                    f1=float(f1),
                    mae=float(mae),
                )
            )

    return out


def _discover_algorithms(results_dir: Path) -> list[tuple[str, Path]]:
    """
    Find all results/<algo_name>/matches.csv.
    """
    algos: list[tuple[str, Path]] = []
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for algo_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        matches = algo_dir / "matches_ss_optimized.csv"
        if matches.exists():
            algos.append((algo_dir.name, matches))
    if not algos:
        raise FileNotFoundError(f"No matches.csv found under: {results_dir}/<algo_name>/matches_ss_optimized.csv")
    return algos

def _generate_color_map(algos: list[str]) -> dict[str, tuple]:
    """
    Assign a consistent color to each algorithm.
    """
    cmap = plt.get_cmap("tab10")  # fino a 10 algoritmi; cambia se ne hai di più
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
            continue  # niente spazio vuoto

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
        patch_artist=True,   # <-- fondamentale per colorare
    )

    # Colorazione
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Median line styling
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

    plotted_mae = _boxplot_by_algo(
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
    "C": (0.121, 0.466, 0.705),  # blue (matplotlib tab:blue)
    "P": (1.000, 0.498, 0.054),  # orange (matplotlib tab:orange)
}

def _collect_algo_cohort_vals(df: pd.DataFrame, algos_order: list[str], metric: str, event_type: str):
    """
    Return lists aligned to algos_order:
      cols_C, cols_P are lists of arrays (may be empty arrays if no data for that cohort)
      has_any: list of booleans whether algo has any data (either cohort)
    """
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
    """
    Draw grouped boxplots: for each algo i draw control box at pos i*spacing - offset,
    PD box at pos i*spacing + offset. Returns nothing.
    """
    n = len(algos)
    spacing = 3.0
    offset = 0.4  # half-distance between paired boxes

    pos_C = []
    pos_P = []
    data_for_box = []
    positions = []
    labels_positions = []
    labels = []

    for i in range(n):
        base = i * spacing
        # control
        if len(cols_C[i]) > 0:
            pos_c = base - offset
            pos_C.append(pos_c)
            data_for_box.append(cols_C[i])
            positions.append(pos_c)
        else:
            pos_c = None
        # pd
        if len(cols_P[i]) > 0:
            pos_p = base + offset
            pos_P.append(pos_p)
            data_for_box.append(cols_P[i])
            positions.append(pos_p)
        else:
            pos_p = None
        # label at base
        labels_positions.append(base)
        labels.append(algos[i])

    if not data_for_box:
        ax.text(0.5, 0.5, "No data for this event type", ha="center", va="center")
        ax.set_axis_off()
        return

    # draw boxes
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

    # color the boxes according to whether position in pos_C or pos_P
    # map each box position back to cohort by comparing proximity
    for patch, pos in zip(box["boxes"], positions):
        # find corresponding cohort by matching pos to pos_C/pos_P lists
        if any(abs(pos - p) < 1e-6 for p in pos_C):
            patch.set_facecolor(colors["C"])
        else:
            patch.set_facecolor(colors["P"])
        patch.set_alpha(0.7)

    # overlay jittered points (individual trials)
    if add_jitter:
        rng = np.random.default_rng(42)
        k = 0
        for i in range(n):
            base = i * spacing
            # control points
            if len(cols_C[i]) > 0:
                xpos = base - offset
                y = cols_C[i]
                jitter = rng.normal(scale=0.08, size=len(y))
                ax.scatter(np.full(len(y), xpos) + jitter, y,
                           color=colors["C"], alpha=0.6, s=10, edgecolors="none")
                k += 1
            # pd points
            if len(cols_P[i]) > 0:
                xpos = base + offset
                y = cols_P[i]
                jitter = rng.normal(scale=0.08, size=len(y))
                ax.scatter(np.full(len(y), xpos) + jitter, y,
                           color=colors["P"], alpha=0.6, s=10, edgecolors="none")
                k += 1

    # x labels at center positions
    ax.set_xticks([i * spacing for i in range(n)])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlim(-spacing * 0.6, spacing * (n - 1) + spacing * 0.6)

    if show_ylim is not None:
        ax.set_ylim(show_ylim)

    # grid & style
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel(ylabel)

def _plot_event_combined(
    metrics_df: pd.DataFrame,
    *,
    event_type: str,
    algos_order: list[str],
    out_dir: Path,
):
    """
    Create a publication-ready figure for given event_type where Control and PD are
    shown side-by-side for each algorithm. Top: F1. Bottom: MAE.
    """
    df = metrics_df[metrics_df["event_type"] == event_type].copy()

    # collect values per algo/cohort
    cols_C_f1, cols_P_f1, has_any_f1 = _collect_algo_cohort_vals(df, algos_order, "f1", event_type)
    cols_C_mae, cols_P_mae, has_any_mae = _collect_algo_cohort_vals(df, algos_order, "mae", event_type)

    # decide which algos to show: union where any metric has data
    show_mask = [has_any_f1[i] or has_any_mae[i] for i in range(len(algos_order))]
    algos_shown = [a for a, ok in zip(algos_order, show_mask) if ok]
    if not algos_shown:
        print(f"No data for event_type={event_type} for any algorithm. Skipping plot.")
        return None

    # filter lists to shown algos
    idxs = [i for i, ok in enumerate(show_mask) if ok]
    cols_C_f1 = [cols_C_f1[i] for i in idxs]
    cols_P_f1 = [cols_P_f1[i] for i in idxs]
    cols_C_mae = [cols_C_mae[i] for i in idxs]
    cols_P_mae = [cols_P_mae[i] for i in idxs]

    # create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, 1.4 * len(algos_shown)), 8), sharex=True)
    fig.suptitle(f"{event_type} — Per-trial distributions (Control vs Parkinson)", fontsize=14)

    # top: F1 (range 0-1)
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

    # bottom: MAE (seconds) — don't enforce same ylim
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

    # legend (manual)
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

    algos = _discover_algorithms(RESULTS_DIR)
    algo_names = [a for a, _ in algos]

    all_metrics: list[TrialMetrics] = []
    for algo_name, matches_path in algos:
        all_metrics.extend(_compute_metrics_for_algo(matches_path, algo_name))

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])

    # Keep only cohorts C and P for plotting (ignore unknowns)
    metrics_df = metrics_df[metrics_df["cohort"].isin(["C", "P"])].copy()

    # Save aggregated table for convenience
    summary_csv = PLOTS_DIR / "per_trial_metrics_ss_optimized.csv"
    metrics_df.to_csv(summary_csv, index=False)

    # Create 4 figures: IC/FC x C/P
    out_paths = []
    # for event_type in ["IC", "FC"]:
    #     for cohort in ["C", "P"]:
    #         out_paths.append(
    #             _plot_cohort_event(
    #                 metrics_df,
    #                 cohort=cohort,
    #                 event_type=event_type,
    #                 algos_order=algo_names,
    #                 out_dir=PLOTS_DIR,
    #             )
    #         )
    # # produce 2 figures (IC and FC) with Control vs PD side-by-side
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

    print("Saved:")
    print(f"- Per-trial metrics table: {summary_csv}")
    for p in out_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
