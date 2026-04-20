# example_single_trial_compare_all_algorithms_with_matching.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.headwalk.gait_events_detection import (
    IcdFang,
    IcdHwang,
    IcdJarchi,
    IcdDiaoComplete,
    IcdSeifer,
    IcdTomc,
    IcdTcn,
    IcdFawden,
    IcdJiang,
    IcdCnn,
    IcdTransformer,
)

# Your project utilities (same as in your full-dataset testing)
from src.headwalk.gait_events_detection import (
    get_all_h5_files,
    preprocess_trial_data,
    match_and_validate,
)


# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXCEL_PATH = DATA_DIR / "Paolo_step_data.xlsx"

FS = 128.0
TOL = 0.25
EXCLUDE: list[str] = []

ALGS: Dict[str, object] = {
    "Fang": IcdFang(),
    "Hwang": IcdHwang(),
    "Jarchi": IcdJarchi(),
    "DiaoIMF": IcdDiaoComplete(),
    "Tomc": IcdTomc(),
    "TCN": IcdTcn(),
    "CNN": IcdCnn(),
    "Seifer": IcdSeifer(),
    "Fawden": IcdFawden(),
    "Jiang": IcdJiang(),
    "Transformer": IcdTransformer(),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_get_pred_events(alg, fs: float, event_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (times_s, sides) for predicted events of type IC or FC.

    If the detector doesn't provide sides, returns 'U' for all.
    """
    event_type = event_type.upper()
    if event_type not in {"IC", "FC"}:
        raise ValueError("event_type must be 'IC' or 'FC'")

    if event_type == "IC":
        lst = getattr(alg, "ic_list_", None)
        side = getattr(alg, "ic_side_", None)
        col = "ic"
    else:
        lst = getattr(alg, "fc_list_", None)
        side = getattr(alg, "fc_side_", None)
        col = "fc"

    if lst is None or len(lst) == 0 or col not in lst.columns:
        return np.array([], dtype=float), np.array([], dtype=str)

    samples = lst[col].to_numpy(dtype=int)
    times = samples / float(fs)

    if side is not None and len(side) == len(samples):
        sides = np.asarray(pd.Series(side).astype(str).str.upper().to_numpy(), dtype=str)
    else:
        sides = np.asarray(["U"] * len(samples), dtype=str)

    return times, sides


def _compute_trial_metrics(matches_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute trial-level metrics exactly like in your testing script.
    Expects columns: 'status', 'diff', 'type'
    """
    status = matches_df["status"].astype(str).str.upper()
    tp = int((status == "TP").sum())
    fp = int((status == "FP").sum())
    fn = int((status == "FN").sum())

    denom = (2 * tp + fp + fn)
    f1 = (2 * tp / denom) if denom > 0 else 0.0

    mae = float(matches_df.loc[status == "TP", "diff"].abs().mean())
    missed_pct = (fn / (tp + fn) * 100.0) if (tp + fn) > 0 else np.nan
    extra_pct = (fp / (tp + fn) * 100.0) if (tp + fn) > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F1": float(f1),
        "MAE": float(mae) if np.isfinite(mae) else np.nan,
        "Missed%": float(missed_pct) if np.isfinite(missed_pct) else np.nan,
        "Extra%": float(extra_pct) if np.isfinite(extra_pct) else np.nan,
    }


def _extract_pred_points_from_matches(matches_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    From matches.csv-like dataframe (output of match_and_validate), extract:
      - matched predicted event times
      - unmatched predicted event times

    We rely on your convention:
      - For FN: time_pred == -999 (no predicted event)
      - For FP: time_ref == -999 (extra predicted event)
      - For TP: both exist and diff is meaningful

    Returns:
      matched_pred_times, unmatched_pred_times  (seconds)
    """
    df = matches_df.copy()
    tp = df[df["status"].astype(str).str.upper() == "TP"]
    fp = df[df["status"].astype(str).str.upper() == "FP"]

    matched = pd.to_numeric(tp["time_pred"], errors="coerce").to_numpy(dtype=float)
    unmatched = pd.to_numeric(fp["time_pred"], errors="coerce").to_numpy(dtype=float)

    matched = matched[np.isfinite(matched) & (matched >= 0)]
    unmatched = unmatched[np.isfinite(unmatched) & (unmatched >= 0)]

    return matched, unmatched


def plot_single_trial_ic_with_matching(
    *,
    time_s: np.ndarray,
    acc_is: np.ndarray,
    ref_df: pd.DataFrame,
    per_algo_matches_ic: Dict[str, pd.DataFrame],
    per_algo_metrics_ic: Dict[str, Dict[str, float]],
    title: str,
) -> None:
    """
    Plot vertical acceleration with:
      - reference ICs as dashed vertical lines
      - per-algorithm predicted ICs:
          triangles = matched (TP)
          circles   = unmatched (FP)
    """
    if "IC" not in ref_df.columns:
        raise ValueError("ref_df must contain column 'IC' (reference IC time in seconds).")

    ref_ic_times = pd.to_numeric(ref_df["IC"], errors="coerce").dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(18, 7), facecolor="white")
    ax.plot(time_s, acc_is, linewidth=3.0, alpha=0.85, color="black", label="acc_is (vertical)")

    # Reference ICs
    for i, t in enumerate(ref_ic_times):
        ax.axvline(
            t,
            linestyle="--",
            linewidth=2.0,
            alpha=0.6,
            color="tab:gray",
            label="Reference IC" if i == 0 else None,
        )

    # Algorithm markers stacked above signal
    y_min = float(np.nanmin(acc_is))
    y_max = float(np.nanmax(acc_is))
    span = (y_max - y_min) if np.isfinite(y_max - y_min) else 1.0

    base_y = y_max + 0.08 * span
    step_y = 0.07 * span

    cmap = plt.get_cmap("tab20")
    algo_names = list(per_algo_matches_ic.keys())

    for k, name in enumerate(algo_names):
        matches_df = per_algo_matches_ic[name]
        matched_t, unmatched_t = _extract_pred_points_from_matches(matches_df)

        y_level = base_y + k * step_y

        color = cmap(k % 20)

        # Matched (TP) -> triangles
        if len(matched_t) > 0:
            ax.scatter(
                matched_t,
                np.full_like(matched_t, y_level),
                marker="^",
                s=100,
                color=color,
                edgecolors="none",
                alpha=0.95,
                label=f"{name} (TP)"
            )

        # Unmatched predictions (FP) -> circles
        if len(unmatched_t) > 0:
            ax.scatter(
                unmatched_t,
                np.full_like(unmatched_t, y_level),
                marker="o",
                s=100,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                alpha=0.95,
                label=f"{name} (FP)" if k == 0 else None,
            )

        # Add a concise text label with metrics at the left
        m = per_algo_metrics_ic.get(name, {})
        txt = f"{name}: F1={m.get('F1', np.nan):.2f}, MAE={m.get('MAE', np.nan):.3f}s"
        ax.text(
            time_s[0],
            y_level,
            txt,
            fontsize=9,
            va="center",
            ha="left",
            color=color,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("acc_is")
    ax.grid(True, alpha=0.25)

    # Expand ylim to show stacked markers
    ax.set_ylim(y_min - 0.1 * span, base_y + max(1, len(algo_names)) * step_y + 0.12 * span)

    # Simple legend (reference only)
    ax.legend(loc="upper right", frameon=False, fontsize = 'large')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("Loading database...")
    db = pd.read_excel(EXCEL_PATH)

    files = get_all_h5_files(DATA_DIR, EXCLUDE)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under: {DATA_DIR}")

    # ------------------------------------------------------------
    # User-defined trial selection (non-interactive)
    # ------------------------------------------------------------
    name_end = "INGP083F2_SI"  # <-- MODIFY HERE

    candidates = [f for f in files if name_end in f.name]

    if len(candidates) == 0:
        raise ValueError(f"No trial filename contains substring: {name_end}")

    if len(candidates) > 1:
        print("Multiple trials match the substring. Using the first one:")
        for f in candidates:
            print(" -", f.name)

    trial_path = candidates[0]
    print(f"\nSelected trial: {trial_path.name}")

    # ------------------------------------------------------------
    # Preprocess trial
    # ------------------------------------------------------------
    df_imu, ref_df, time_rs = preprocess_trial_data(trial_path, db, FS)

    if "acc_is" not in df_imu.columns:
        raise ValueError("df_imu must contain column 'acc_is' for plotting.")

    acc_is = df_imu["acc_is"].to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Run all algorithms + matching
    # ------------------------------------------------------------
    per_algo_matches_ic: Dict[str, pd.DataFrame] = {}
    per_algo_metrics_ic: Dict[str, Dict[str, float]] = {}

    for name, alg in ALGS.items():
        try:
            alg.detect(df_imu, sampling_rate_hz=FS, plot_debug=False)

            p_ic_t, p_ic_s = _safe_get_pred_events(alg, FS, "IC")

            m_ic = match_and_validate(p_ic_t, p_ic_s, ref_df, "IC", TOL)
            m_ic["filename"] = trial_path.name

            per_algo_matches_ic[name] = m_ic
            per_algo_metrics_ic[name] = _compute_trial_metrics(m_ic)

            met = per_algo_metrics_ic[name]
            print(f"{name:14s} -> IC: F1={met['F1']:.3f} | MAE={met['MAE']:.4f}s | TP={met['TP']} FP={met['FP']} FN={met['FN']}")
        except Exception as e:
            print(f"{name:14s} -> ERROR: {e}")
            per_algo_matches_ic[name] = pd.DataFrame()
            per_algo_metrics_ic[name] = {"F1": np.nan, "MAE": np.nan, "TP": 0, "FP": 0, "FN": 0}

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plot_single_trial_ic_with_matching(
        time_s=time_rs,
        acc_is=acc_is,
        ref_df=ref_df,
        per_algo_matches_ic=per_algo_matches_ic,
        per_algo_metrics_ic=per_algo_metrics_ic,
        title=f"Single-trial IC comparison with matching — {trial_path.name} (tol={TOL}s)",
    )

if __name__ == "__main__":
    main()
