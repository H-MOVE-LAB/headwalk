# Load and visually evaluate Gait Sequence Detection on one or two single trials.
#
# This example runs sklearn-based GSD algorithms through the common detect()
# interface and plots detected gait sequences as shaded bands.
#
# It is compatible with WearGaitPD FreeWalk CSV files exported with Forehead IMU
# channels.

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from headwalk.gait_sequence_detection.algo import (  # noqa: E402
    GsdSvm,
    GsdRandomForest,
    GsdKnn,
    GsdLogisticRegression,
    GsdGaussianNaiveBayes,
    GsdRuleBased,
    GsdCnn1D,
)

from headwalk.gait_sequence_detection.preprocessing import (  # noqa: E402
    preprocess_head_imu_trial,
)


RAW_COLUMNS = [
    "acc_vt",
    "acc_ml",
    "acc_ap",
    "gyr_vt",
    "gyr_ml",
    "gyr_ap",
]


PLOT_SIGNAL_COLUMNS = [
    "acc_vt",
    "acc_ml",
    "acc_ap",
    "gyr_vt",
    "gyr_ml",
    "gyr_ap",
]


ALGORITHM_CLASSES = [
    GsdSvm,
    GsdRandomForest,
    GsdKnn,
    GsdLogisticRegression,
    GsdGaussianNaiveBayes,
    GsdRuleBased,
    GsdCnn1D,
]


ALGORITHM_COLORS = {
    "GsdSvm": "#1f77b4",
    "GsdRandomForest": "#ff7f0e",
    "GsdKnn": "#2ca02c",
    "GsdLogisticRegression": "#d62728",
    "GsdGaussianNaiveBayes": "#9467bd",
    "GsdRuleBased": "#8c564b",
    "GsdCnn1D": "#e377c2",
}


ALGORITHM_SHORT_NAMES = {
    "GsdSvm": "SVM",
    "GsdRandomForest": "RF",
    "GsdKnn": "KNN",
    "GsdLogisticRegression": "LR",
    "GsdGaussianNaiveBayes": "GNB",
    "GsdRuleBased": "RB",
    "GsdCnn1D": "CNN",
}


def resolve_cli_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None

    text = str(path_text).strip().strip('"').strip("'")

    if re.match(r"^/[A-Za-z]/", text):
        drive = text[1].upper()
        rest = text[2:]
        return Path(f"{drive}:{rest}")

    return Path(text)


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .lower()
    )


def load_trial_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trial CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def parse_time_column_to_seconds(df: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
    df = df.copy()

    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        return df

    if "time" in df.columns:
        extracted = (
            df["time"]
            .astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        )
        df["time_s"] = pd.to_numeric(extracted, errors="coerce")

        if df["time_s"].notna().any():
            return df

    df["time_s"] = np.arange(len(df)) / sampling_rate_hz
    return df


def add_standard_head_imu_columns(
    df: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """
    Create standardized lowercase head-IMU columns using the same WearGaitPD
    preprocessing convention used during training.

    Important:
    - raw columns are loaded as Forehead_Acc_X/Y/Z and Forehead_Gyr_X/Y/Z;
    - no axis reordering is applied before gravity alignment;
    - acceleration is kept in m/s^2, because WearGaitPD raw Acc norm is ~9.81
      and the original construction did not convert it to g;
    - the first static second is used for gravity alignment, matching the
      WearGaitPD construction.
    """

    df = df.copy()

    raw_acc_columns = [
        "forehead_acc_x",
        "forehead_acc_y",
        "forehead_acc_z",
    ]

    raw_gyr_columns = [
        "forehead_gyr_x",
        "forehead_gyr_y",
        "forehead_gyr_z",
    ]

    missing_sources = [
        col for col in raw_acc_columns + raw_gyr_columns
        if col not in df.columns
    ]

    if missing_sources:
        raise KeyError(
            "Cannot apply WearGaitPD-like GSD preprocessing. "
            f"Missing raw Forehead IMU columns: {missing_sources}"
        )

    acc_raw = df[raw_acc_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    gyr_raw = df[raw_gyr_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    print("[INFO] Applying WearGaitPD-like preprocessing.")
    print("[INFO] Raw Acc order: Forehead_Acc_X, Forehead_Acc_Y, Forehead_Acc_Z")
    print("[INFO] Raw Gyr order: Forehead_Gyr_X, Forehead_Gyr_Y, Forehead_Gyr_Z")
    print("[INFO] Acc unit kept as m/s^2 to match WearGaitPD construction.")

    preprocessing_result = preprocess_head_imu_trial(
        acc_raw=acc_raw,
        gyr_raw=gyr_raw,
        fs=sampling_rate_hz,
        static_duration_s=1.0,
        static_reference_mask=None,
        fixed_axis_transform=None,
        acc_input_unit="m/s2",
        acc_output_unit="m/s2",
        gyr_scale_factor=1.0,
        lowpass_cutoff_hz=15.0,
    )

    X = preprocessing_result.X

    # After preprocessing, columns 0-2 and 3-5 follow the same semantic naming
    # used by feature computation: VT, ML, AP for Acc and Gyr.
    df["acc_vt"] = X[:, 0]
    df["acc_ml"] = X[:, 1]
    df["acc_ap"] = X[:, 2]
    df["gyr_vt"] = X[:, 3]
    df["gyr_ml"] = X[:, 4]
    df["gyr_ap"] = X[:, 5]

    df["imu_quality"] = preprocessing_result.imu_quality

    debug = preprocessing_result.preprocessing_debug_info
    print(
        "[INFO] Preprocessing completed | "
        f"input_unit={debug.get('input_unit_effective')} -> {debug.get('output_unit')} | "
        f"step_frequency={debug.get('step_frequency_hz'):.3f} Hz | "
        f"M3 cutoff={debug.get('m3_cutoff_hz'):.3f} Hz"
    )

    print("[INFO] Mean static acc aligned:", debug.get("mean_static_acc_aligned"))
    print("[INFO] Mean static acc aligned norm:", debug.get("mean_static_acc_aligned_norm"))

    return df


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise KeyError(
            "Missing standardized IMU columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )


def crop_by_time(
    df: pd.DataFrame,
    *,
    start_s: float,
    duration_s: float | None,
) -> pd.DataFrame:
    if duration_s is None:
        out = df[df["time_s"] >= start_s].copy()
    else:
        end_s = start_s + duration_s
        out = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    out = out.reset_index(drop=True)

    if not out.empty:
        out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])

    return out


def zscore_for_plot(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def run_algorithms(
    trial_df: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> dict[str, object]:
    algorithms = []

    for cls in ALGORITHM_CLASSES:
        try:
            algorithms.append(cls())
        except ImportError as exc:
            print(f"[SKIP] {cls.__name__} skipped: {exc}")
        except Exception as exc:
            print(f"[SKIP] {cls.__name__} could not be initialized: {exc}")

    results = {}

    for algorithm in algorithms:
        print(f"[INFO] Running {algorithm.__class__.__name__}")

        algorithm.detect(
            data=trial_df,
            sampling_rate_hz=sampling_rate_hz,
            acc_columns=["acc_vt", "acc_ml", "acc_ap"],
            gyr_columns=["gyr_vt", "gyr_ml", "gyr_ap"],
            min_sequence_duration_s=0.0,
            merge_gap_s=0.0,
        )

        results[algorithm.__class__.__name__] = algorithm

        print(
            f"       windows={len(algorithm.window_detections_)} | "
            f"gs={len(algorithm.gs_list_)}"
        )

    return results


def plot_single_trial(
    trial_df: pd.DataFrame,
    algorithms: dict[str, object],
    *,
    trial_name: str,
    output_path: Path,
    max_bands: int,
) -> None:
    if trial_df.empty:
        raise ValueError("Selected trial segment is empty.")

    n_signal_rows = len(PLOT_SIGNAL_COLUMNS)
    n_algorithm_rows = len(algorithms)
    n_rows = n_signal_rows + n_algorithm_rows

    fig_height = max(8, 1.25 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(16, fig_height),
        sharex=True,
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = [axes]

    time = trial_df["time_s"].to_numpy()

    for ax, column in zip(axes[:n_signal_rows], PLOT_SIGNAL_COLUMNS):
        y = zscore_for_plot(trial_df[column].to_numpy())
        ax.plot(time, y, linewidth=0.9)
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.25)

    reference_signal = zscore_for_plot(trial_df["acc_vt"].to_numpy())

    for ax, (algorithm_name, algorithm) in zip(
        axes[n_signal_rows:],
        algorithms.items(),
    ):
        color = ALGORITHM_COLORS.get(algorithm_name, "tab:gray")

        ax.plot(
            time,
            reference_signal,
            linewidth=0.8,
            color="black",
            alpha=0.65,
        )

        gs_list = algorithm.gs_list_.copy().head(max_bands)

        for _, row in gs_list.iterrows():
            start = float(row["start_s"])
            end = float(row["end_s"])

            ax.axvspan(
                start,
                end,
                facecolor=color,
                alpha=0.20,
                edgecolor=color,
                linewidth=2.0,
            )
            ax.axvline(start, color=color, linewidth=1.4, alpha=0.95)
            ax.axvline(end, color=color, linewidth=1.4, alpha=0.95)

        ax.set_ylabel(ALGORITHM_SHORT_NAMES.get(algorithm_name, algorithm_name))
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time from selected trial segment start [s]")

    fig.suptitle(
        f"GSD single-trial visual evaluation | {trial_name}",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"[SAVED] {output_path}")


def save_outputs(
    algorithms: dict[str, object],
    *,
    trial_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = trial_name.replace(" ", "_").replace("/", "_")

    for algorithm_name, algorithm in algorithms.items():
        window_path = output_dir / f"{safe_name}_{algorithm_name}_window_detections.csv"
        gs_path = output_dir / f"{safe_name}_{algorithm_name}_gs_list.csv"

        algorithm.window_detections_.to_csv(window_path, index=False)
        algorithm.gs_list_.to_csv(gs_path, index=False)

        print(f"[SAVED] {window_path}")
        print(f"[SAVED] {gs_path}")


def process_trial(
    trial_csv: Path,
    *,
    trial_name: str,
    sampling_rate_hz: float,
    output_dir: Path,
    start_s: float,
    duration_s: float | None,
    max_bands: int,
) -> None:
    print("\n============================================================")
    print(f"[TRIAL] {trial_name}")
    print(f"[PATH]  {trial_csv}")
    print("============================================================")

    trial_df = load_trial_csv(trial_csv)
    trial_df = parse_time_column_to_seconds(
        trial_df,
        sampling_rate_hz=sampling_rate_hz,
    )
    trial_df = add_standard_head_imu_columns(
        trial_df,
        sampling_rate_hz=sampling_rate_hz,
    )

    check_required_columns(trial_df, RAW_COLUMNS)

    segment_df = crop_by_time(
        trial_df,
        start_s=start_s,
        duration_s=duration_s,
    )

    if segment_df.empty:
        raise ValueError("Selected segment is empty.")

    print(
        f"[INFO] Selected segment samples={len(segment_df)} | "
        f"duration={segment_df['time_s'].max():.2f} s"
    )

    algorithms = run_algorithms(
        segment_df,
        sampling_rate_hz=sampling_rate_hz,
    )

    safe_name = trial_name.replace(" ", "_").replace("/", "_")
    figure_path = output_dir / f"{safe_name}_gsd_single_trial.png"

    plot_single_trial(
        segment_df,
        algorithms,
        trial_name=trial_name,
        output_path=figure_path,
        max_bands=max_bands,
    )

    save_outputs(
        algorithms,
        trial_name=trial_name,
        output_dir=output_dir / "tables",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--trial-csv", type=str, default=None)
    parser.add_argument("--trial-name", type=str, default="single_trial")

    parser.add_argument("--pd-trial-csv", type=str, default=None)
    parser.add_argument("--control-trial-csv", type=str, default=None)

    parser.add_argument("--sampling-rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/gait_sequence_detection/single_trial_eval",
    )

    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--max-bands", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = resolve_cli_path(args.output_dir)

    trial_csv = resolve_cli_path(args.trial_csv)
    pd_trial_csv = resolve_cli_path(args.pd_trial_csv)
    control_trial_csv = resolve_cli_path(args.control_trial_csv)

    if trial_csv is None and pd_trial_csv is None and control_trial_csv is None:
        raise ValueError(
            "Provide either --trial-csv or at least one of "
            "--pd-trial-csv / --control-trial-csv."
        )

    if trial_csv is not None:
        process_trial(
            trial_csv,
            trial_name=args.trial_name,
            sampling_rate_hz=args.sampling_rate_hz,
            output_dir=output_dir,
            start_s=args.start_s,
            duration_s=args.duration_s,
            max_bands=args.max_bands,
        )

    if pd_trial_csv is not None:
        process_trial(
            pd_trial_csv,
            trial_name="pd_freewalk",
            sampling_rate_hz=args.sampling_rate_hz,
            output_dir=output_dir,
            start_s=args.start_s,
            duration_s=args.duration_s,
            max_bands=args.max_bands,
        )

    if control_trial_csv is not None:
        process_trial(
            control_trial_csv,
            trial_name="control_freewalk",
            sampling_rate_hz=args.sampling_rate_hz,
            output_dir=output_dir,
            start_s=args.start_s,
            duration_s=args.duration_s,
            max_bands=args.max_bands,
        )


if __name__ == "__main__":
    main()
