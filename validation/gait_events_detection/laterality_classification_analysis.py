# laterality_classification_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "_analysis_plots"

COHORT_RE = re.compile(r"ING([CP])", re.IGNORECASE)  # INGC... or INGP...
EXCLUDE_LIST = [ # updated list
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
# Publication-ready matplotlib defaults
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COHORT_COLORS = {
    "C": (0.121, 0.466, 0.705),  # tab:blue
    "P": (1.000, 0.498, 0.054),  # tab:orange
}


@dataclass(frozen=True)
class TrialLaterality:
    algo: str
    cohort: str     # "C" or "P"
    event_type: str # "IC" or "FC"
    filename: str
    left_rate: float   # NaN if no reference left events
    right_rate: float  # NaN if no reference right events
    accuracy: float


def _extract_cohort(filename: str) -> str:
    """
    Extract cohort from filename, e.g. "...-INGC116F2_SC.h5" -> "C".
    Returns "U" if not found.
    """
    m = COHORT_RE.search(str(filename))
    if not m:
        return "U"
    return m.group(1).upper()


def _discover_algorithms(results_dir: Path) -> list[tuple[str, Path]]:
    """
    Find all results/<algo_name>/matches.csv.
    """
    algos: list[tuple[str, Path]] = []
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for algo_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        matches = algo_dir / "matches.csv"
        if matches.exists():
            algos.append((algo_dir.name, matches))

    if not algos:
        raise FileNotFoundError(f"No matches.csv found under: {results_dir}/<algo_name>/matches.csv")
    return algos


# -----------------------------------------------------------------------------
# Metrics computation
# -----------------------------------------------------------------------------
def _compute_laterality_for_algo(matches_csv: Path, algo_name: str) -> list[TrialLaterality]:
    """
    Per-trial laterality detection rates.

    Definitions (per trial, per event_type):
      Left Detection Rate  = (# TP with side_ref='L' AND side_pred='L') / (# reference left events)
      Right Detection Rate = (# TP with side_ref='R' AND side_pred='R') / (# reference right events)

    Reference left/right event counts are computed using status in {TP,FN} and side_ref in {L,R}.
    If a trial has zero reference events of a side, the corresponding rate is NaN.
    """
    df = pd.read_csv(matches_csv)
    df.columns = [c.strip() for c in df.columns]

    required = {"status", "type", "filename", "side_pred", "side_ref"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{matches_csv} is missing columns required for laterality: {missing}")

    df["status"] = df["status"].astype(str).str.upper()
    df["type"] = df["type"].astype(str).str.upper()
    df["side_pred"] = df["side_pred"].astype(str).str.upper()
    df["side_ref"] = df["side_ref"].astype(str).str.upper()
    df["cohort"] = df["filename"].astype(str).map(_extract_cohort)

    out: list[TrialLaterality] = []

    for event_type in ["IC", "FC"]:
        df_t = df[df["type"] == event_type].copy()
        if df_t.empty:
            continue

        for (fname, cohort), g in df_t.groupby(["filename", "cohort"], dropna=False):
            if any([e in fname for e in EXCLUDE_LIST]):
                continue
            # Reference events for laterality denominator: those that exist in reference (TP or FN)
            ref_left = g[(g["status"].isin({"TP", "FN"})) & (g["side_ref"] == "L")]
            ref_right = g[(g["status"].isin({"TP", "FN"})) & (g["side_ref"] == "R")]

            # Correct laterality: TP + correct side
            correct_left = g[(g["status"] == "TP") & (g["side_ref"] == "L") & (g["side_pred"] == "L")]
            correct_right = g[(g["status"] == "TP") & (g["side_ref"] == "R") & (g["side_pred"] == "R")]

            left_rate = float(len(correct_left)) / len(ref_left) if len(ref_left) > 0 else float("nan")
            right_rate = float(len(correct_right)) / len(ref_right) if len(ref_right) > 0 else float("nan")
            n_left = len(ref_left)
            n_right = len(ref_right)
            n_tot = n_left + n_right

            n_correct = len(correct_left) + len(correct_right)

            accuracy = n_correct / n_tot if n_tot > 0 else float("nan")
            out.append(
                TrialLaterality(
                    algo=algo_name,
                    cohort=cohort,
                    event_type=event_type,
                    filename=str(fname),
                    left_rate=float(left_rate),
                    right_rate=float(right_rate),
                    accuracy = float(accuracy),
                )
            )

    return out


def compute_laterality_all(results_dir: Path) -> pd.DataFrame:
    """
    Compute laterality per-trial for all algorithms found under results_dir.
    Returns a DataFrame with:
      algo, cohort, event_type, filename, left_rate, right_rate
    """
    algos = _discover_algorithms(results_dir)
    rows: list[TrialLaterality] = []
    for algo_name, matches_path in algos:
        rows.extend(_compute_laterality_for_algo(matches_path, algo_name))

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df[df["cohort"].isin(["C", "P"])].copy()  # ignore unknown cohort
    return df


# -----------------------------------------------------------------------------
# Plotting helpers (Control vs PD grouped per algorithm)
# -----------------------------------------------------------------------------
def _collect_algo_cohort_vals(
    df: pd.DataFrame,
    algos_order: list[str],
    metric: str,
    event_type: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[bool]]:
    """
    For each algorithm, collect values for cohort C and P for a given metric and event_type.
    Returns (vals_C, vals_P, has_any)
    """
    vals_C, vals_P, has_any = [], [], []
    for algo in algos_order:
        sub = df[(df["algo"] == algo) & (df["event_type"] == event_type)]
        c = sub[sub["cohort"] == "C"][metric].dropna().astype(float).to_numpy()
        p = sub[sub["cohort"] == "P"][metric].dropna().astype(float).to_numpy()
        vals_C.append(c)
        vals_P.append(p)
        has_any.append((len(c) > 0) or (len(p) > 0))
    return vals_C, vals_P, has_any


def _plot_grouped_boxplots(
    ax: plt.Axes,
    algos: list[str],
    vals_C: list[np.ndarray],
    vals_P: list[np.ndarray],
    ylabel: str,
    title: str,
    show_ylim: tuple[float, float] = (-0.05, 1.05),
    add_jitter: bool = True,
) -> None:
    """
    Draw grouped boxplots per algorithm (C and P side-by-side).
    """
    n = len(algos)
    spacing = 3.0
    offset = 0.4

    pos_C, pos_P = [], []
    data_for_box, positions = [], []

    for i in range(n):
        base = i * spacing
        if len(vals_C[i]) > 0:
            pc = base - offset
            pos_C.append(pc)
            data_for_box.append(vals_C[i])
            positions.append(pc)
        if len(vals_P[i]) > 0:
            pp = base + offset
            pos_P.append(pp)
            data_for_box.append(vals_P[i])
            positions.append(pp)

    if not data_for_box:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    box = ax.boxplot(
        data_for_box,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
    )

    # Color boxes based on cohort position list
    for patch, pos in zip(box["boxes"], positions):
        if any(abs(pos - p) < 1e-6 for p in pos_C):
            patch.set_facecolor(COHORT_COLORS["C"])
        else:
            patch.set_facecolor(COHORT_COLORS["P"])
        patch.set_alpha(0.75)

    # Jittered points
    if add_jitter:
        rng = np.random.default_rng(42)
        for i in range(n):
            base = i * spacing
            if len(vals_C[i]) > 0:
                x = base - offset
                y = vals_C[i]
                jitter = rng.normal(scale=0.06, size=len(y))
                ax.scatter(np.full(len(y), x) + jitter, y,
                           color=COHORT_COLORS["C"], alpha=0.6, s=10, edgecolors="none")
            if len(vals_P[i]) > 0:
                x = base + offset
                y = vals_P[i]
                jitter = rng.normal(scale=0.06, size=len(y))
                ax.scatter(np.full(len(y), x) + jitter, y,
                           color=COHORT_COLORS["P"], alpha=0.6, s=10, edgecolors="none")

    ax.set_xticks([i * spacing for i in range(n)])
    ax.set_xticklabels(algos, rotation=30, ha="right")
    ax.set_xlim(-spacing * 0.6, spacing * (n - 1) + spacing * 0.6)
    ax.set_ylim(show_ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def plot_laterality_event(
    laterality_df: pd.DataFrame,
    *,
    event_type: str,
    algos_order: list[str],
    out_dir: Path,
) -> tuple[Path, Path] | None:
    """
    Create one figure for event_type with 2 subplots:
      - Top: Left Detection Rate distribution
      - Bottom: Right Detection Rate distribution
    Control vs PD are side-by-side per algorithm.
    """
    df = laterality_df[laterality_df["event_type"] == event_type].copy()

    vals_C_L, vals_P_L, has_any_L = _collect_algo_cohort_vals(df, algos_order, "left_rate", event_type)
    vals_C_R, vals_P_R, has_any_R = _collect_algo_cohort_vals(df, algos_order, "right_rate", event_type)

    # Keep only algorithms that have any data for either metric
    show_mask = [has_any_L[i] or has_any_R[i] for i in range(len(algos_order))]
    algos_shown = [a for a, ok in zip(algos_order, show_mask) if ok]
    if not algos_shown:
        print(f"No laterality data for {event_type}. Skipping plot.")
        return None

    idxs = [i for i, ok in enumerate(show_mask) if ok]
    vals_C_L = [vals_C_L[i] for i in idxs]
    vals_P_L = [vals_P_L[i] for i in idxs]
    vals_C_R = [vals_C_R[i] for i in idxs]
    vals_P_R = [vals_P_R[i] for i in idxs]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(max(10, 1.4 * len(algos_shown)), 8),
        sharex=True
    )
    fig.suptitle(f"{event_type} — Laterality classification (Control vs Parkinson)", fontsize=14)

    _plot_grouped_boxplots(
        ax1, algos_shown, vals_C_L, vals_P_L,
        ylabel="Left Detection Rate",
        title="Left Detection Rate (per trial)",
        show_ylim=(-0.05, 1.05),
        add_jitter=True,
    )

    _plot_grouped_boxplots(
        ax2, algos_shown, vals_C_R, vals_P_R,
        ylabel="Right Detection Rate",
        title="Right Detection Rate (per trial)",
        show_ylim=(-0.05, 1.05),
        add_jitter=True,
    )

    # Legend
    handles = [
        mpl.patches.Patch(facecolor=COHORT_COLORS["C"], label="Control (C)", alpha=0.75),
        mpl.patches.Patch(facecolor=COHORT_COLORS["P"], label="Parkinson (P)", alpha=0.75),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.96, 0.92))

    plt.tight_layout(rect=[0, 0.01, 0.94, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{event_type}_laterality_Control_vs_PD_boxplots.png"
    out_pdf = out_dir / f"{event_type}_laterality_Control_vs_PD_boxplots.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)

    return out_png, out_pdf


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover algorithms first (to keep a stable ordering)
    algos = _discover_algorithms(RESULTS_DIR)
    algo_names = [a for a, _ in algos]

    laterality_df = compute_laterality_all(RESULTS_DIR)

    # Save table
    out_csv = PLOTS_DIR / "per_trial_laterality.csv"
    laterality_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plots
    for event_type in ["IC", "FC"]:
        out = plot_laterality_event(
            laterality_df,
            event_type=event_type,
            algos_order=algo_names,
            out_dir=PLOTS_DIR,
        )
        if out is not None:
            png_path, pdf_path = out
            print(f"Saved: {png_path}")
            print(f"Saved: {pdf_path}")

    print("\nLaterality classification analysis completed.")


if __name__ == "__main__":
    main()
