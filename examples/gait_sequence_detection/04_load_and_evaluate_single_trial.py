# Visual single-trial Gait Sequence Detection example.
#
# This script runs one or more GSD algorithms on compact single-trial CSV files
# and creates a visual comparison plot.
#
# Difference from 01_apply_gsd_single_trial.py:
#
#     01 = minimal raw inference example
#     04 = visual evaluation / debugging example
#
# Both scripts use the same schema-based input adapter:
#
#     _single_trial_input.py
#
# Therefore, raw IMU column selection, optional sensor-prefix handling and
# preprocessing are shared between the minimal and visual examples.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLE_DIR = Path(__file__).resolve().parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


DEFAULT_PD_TRIAL_CSV = (
    PROJECT_ROOT
    / "example_data"
    / "gait_sequence_detection"
    / "weargaitpd_pd_freewalk_nls192.csv"
)

DEFAULT_CONTROL_TRIAL_CSV = (
    PROJECT_ROOT
    / "example_data"
    / "gait_sequence_detection"
    / "weargaitpd_control_freewalk_whc021.csv"
)


from _single_trial_input import (  # noqa: E402
    load_trial_csv,
    normalize_column_name,
    parse_time_column_to_seconds,
    resolve_cli_path,
    standardize_single_trial_imu,
)

from headwalk.gait_sequence_detection.algo import (  # noqa: E402
    GsdSvm,
    GsdRandomForest,
    GsdKnn,
    GsdLogisticRegression,
    GsdGaussianNaiveBayes,
    GsdRuleBased,
    GsdCnn1D,
)


ALGORITHM_CLASSES = {
    "svm": GsdSvm,
    "rf": GsdRandomForest,
    "knn": GsdKnn,
    "lr": GsdLogisticRegression,
    "gnb": GsdGaussianNaiveBayes,
    "rb": GsdRuleBased,
    "cnn": GsdCnn1D,
}


def select_algorithm_names(
    *,
    algorithm: str | None,
    algorithms: list[str] | None,
) -> list[str]:
    """
    Return the list of algorithm keys to run.

    --algorithms has priority over --algorithm.

    Examples:
        --algorithm cnn
        --algorithms cnn rb
        --algorithms svm rf cnn
        --algorithms all
    """

    if algorithms:
        selected = [name.lower() for name in algorithms]
    else:
        selected = [(algorithm or "all").lower()]

    if "all" in selected:
        return ["svm", "rf", "knn", "lr", "gnb", "rb", "cnn"]

    unknown = [name for name in selected if name not in ALGORITHM_CLASSES]

    if unknown:
        raise ValueError(
            f"Unknown algorithm(s): {unknown}. "
            f"Available: {sorted(ALGORITHM_CLASSES)} or all."
        )

    return selected


def resolve_optional_trial_path(path_text: str | None) -> Path | None:
    """Resolve a trial path and allow none/empty values to skip a trial."""

    if path_text is None:
        return None

    clean = str(path_text).strip().strip('"').strip("'")

    if clean == "" or clean.lower() in {"none", "null", "skip"}:
        return None

    return resolve_cli_path(clean)


def compute_window_quality_flags(
    window_detections: pd.DataFrame,
    *,
    sample_quality,
) -> pd.DataFrame:
    """
    Add quality metadata to a window-level detection table.

    This is used only when --exclude-low-quality-windows is enabled.

    A window is high quality only if all samples inside that window have
    sample_quality == True.
    """

    if sample_quality is None:
        out = window_detections.copy()
        out["window_quality_fraction"] = 1.0
        out["window_is_high_quality"] = True
        return out

    quality = np.asarray(sample_quality, dtype=bool)

    out = window_detections.copy()

    required_columns = ["window_start_sample", "window_end_sample"]
    missing = [col for col in required_columns if col not in out.columns]

    if missing:
        raise KeyError(
            "Cannot apply sample quality to window detections. "
            f"Missing columns: {missing}"
        )

    quality_fractions = []
    high_quality_flags = []

    for _, row in out.iterrows():
        start = int(row["window_start_sample"])
        end = int(row["window_end_sample"])

        if start < 0 or end > len(quality) or end <= start:
            quality_fractions.append(0.0)
            high_quality_flags.append(False)
            continue

        window_quality = quality[start:end]

        quality_fractions.append(float(np.mean(window_quality)))
        high_quality_flags.append(bool(np.all(window_quality)))

    out["window_quality_fraction"] = quality_fractions
    out["window_is_high_quality"] = high_quality_flags

    return out


def apply_quality_filter_to_algorithm(
    algorithm,
    *,
    sample_quality,
    min_sequence_duration_s: float,
    merge_gap_s: float,
):
    """
    Remove low-quality windows after detection and rebuild gs_list_.

    The algorithm is still run on the full preprocessed trial. This post-filter
    only removes unreliable windows from the exported result.
    """

    window_detections = compute_window_quality_flags(
        algorithm.window_detections_,
        sample_quality=sample_quality,
    )

    n_before = len(window_detections)

    high_quality_window_detections = window_detections[
        window_detections["window_is_high_quality"]
    ].copy()

    n_after = len(high_quality_window_detections)

    algorithm.window_detections_ = high_quality_window_detections.reset_index(
        drop=True
    )

    algorithm.gs_list_ = algorithm._build_gs_list(
        window_detections=algorithm.window_detections_,
        min_sequence_duration_s=min_sequence_duration_s,
        merge_gap_s=merge_gap_s,
    )

    if hasattr(algorithm, "result_"):
        algorithm.result_.window_detections = algorithm.window_detections_
        algorithm.result_.gs_list = algorithm.gs_list_

    print(
        "[INFO] quality filter: "
        f"kept {n_after}/{n_before} windows "
        f"({n_before - n_after} removed)"
    )

    return algorithm


def run_algorithms(
    trial_df: pd.DataFrame,
    *,
    algorithm_names: list[str],
    sampling_rate_hz: float,
    min_sequence_duration_s: float,
    merge_gap_s: float,
    exclude_low_quality_windows: bool,
) -> dict[str, object]:
    """Run selected GSD algorithms on one standardized trial."""

    results = {}

    for algorithm_name in algorithm_names:
        cls = ALGORITHM_CLASSES[algorithm_name]

        try:
            algorithm = cls()
        except ImportError as exc:
            print(f"[SKIP] {cls.__name__}: {exc}")
            continue
        except Exception as exc:
            print(f"[SKIP] {cls.__name__} could not be initialized: {exc}")
            continue

        print(f"[INFO] Running {algorithm.__class__.__name__}")

        algorithm.detect(
            data=trial_df,
            sampling_rate_hz=sampling_rate_hz,
            acc_columns=["acc_vt", "acc_ml", "acc_ap"],
            gyr_columns=["gyr_vt", "gyr_ml", "gyr_ap"],
            min_sequence_duration_s=min_sequence_duration_s,
            merge_gap_s=merge_gap_s,
        )

        if exclude_low_quality_windows:
            if "imu_quality" not in trial_df.columns:
                raise KeyError(
                    "--exclude-low-quality-windows was requested, but the "
                    "standardized trial does not contain an imu_quality column."
                )

            algorithm = apply_quality_filter_to_algorithm(
                algorithm,
                sample_quality=trial_df["imu_quality"].to_numpy(dtype=bool),
                min_sequence_duration_s=min_sequence_duration_s,
                merge_gap_s=merge_gap_s,
            )

        print(
            f"       windows={len(algorithm.window_detections_)} | "
            f"gs={len(algorithm.gs_list_)}"
        )

        results[algorithm.__class__.__name__] = algorithm

    return results


def build_reference_gs_list(
    trial_df: pd.DataFrame,
    *,
    reference_column: str,
    walking_labels: list[str],
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """
    Build an optional binary reference gs_list_ from a label column.

    The reference is used only for visualization. It is not needed by the GSD
    algorithms.
    """

    normalized_reference_column = normalize_column_name(reference_column)

    if normalized_reference_column not in trial_df.columns:
        raise KeyError(
            f"Reference column {reference_column!r} was not found after "
            f"normalization to {normalized_reference_column!r}."
        )

    walking_set = {
        str(label).strip().lower()
        for label in walking_labels
    }

    labels = (
        trial_df[normalized_reference_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    walking_mask = labels.isin(walking_set).to_numpy(dtype=bool)

    n_samples = len(walking_mask)

    if n_samples == 0 or not np.any(walking_mask):
        return pd.DataFrame(
            columns=["start_s", "end_s", "start_samples", "end_samples"]
        )

    padded = np.concatenate([
        np.array([False]),
        walking_mask,
        np.array([False]),
    ])

    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    rows = []

    time_s = trial_df["time_s"].to_numpy(dtype=float)

    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)

        if start >= end:
            continue

        if start < len(time_s):
            start_s = float(time_s[start])
        else:
            start_s = start / sampling_rate_hz

        if end < len(time_s):
            end_s = float(time_s[end])
        else:
            end_s = end / sampling_rate_hz

        rows.append({
            "start_s": start_s,
            "end_s": end_s,
            "start_samples": start,
            "end_samples": end,
        })

    return pd.DataFrame(rows)


def clip_gs_list_to_segment(
    gs_list: pd.DataFrame,
    *,
    start_s: float,
    end_s: float,
) -> pd.DataFrame:
    """Clip gait-sequence intervals to the plotted time segment."""

    if gs_list is None or gs_list.empty:
        return pd.DataFrame(
            columns=["start_s", "end_s", "start_samples", "end_samples"]
        )

    clipped_rows = []

    for _, row in gs_list.iterrows():
        seq_start = float(row["start_s"])
        seq_end = float(row["end_s"])

        clipped_start = max(seq_start, start_s)
        clipped_end = min(seq_end, end_s)

        if clipped_end <= clipped_start:
            continue

        clipped = row.to_dict()
        clipped["start_s"] = clipped_start
        clipped["end_s"] = clipped_end

        clipped_rows.append(clipped)

    return pd.DataFrame(clipped_rows)


def plot_single_trial(
    trial_df: pd.DataFrame,
    *,
    trial_name: str,
    algorithms: dict[str, object],
    reference_gs_list: pd.DataFrame | None,
    start_s: float,
    end_s: float,
    output_path: Path,
    max_bands: int,
) -> None:
    """
    Create a visual comparison plot for one trial.

    The top panel shows the standardized accelerometer signal.
    If a reference is available, walking-labelled reference intervals are shown
    as shaded regions on the signal.
    The lower panel shows only algorithm predictions.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    time_s = trial_df["time_s"].to_numpy(dtype=float)

    segment_mask = (time_s >= start_s) & (time_s <= end_s)

    if not np.any(segment_mask):
        raise ValueError(
            f"No samples found in requested plot segment [{start_s}, {end_s}] s."
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1 + 0.35 * max(len(algorithms), 1)]},
    )

    signal_axis = axes[0]
    bands_axis = axes[1]

    segment_time = time_s[segment_mask]

    if reference_gs_list is not None and not reference_gs_list.empty:
        clipped_ref = clip_gs_list_to_segment(
            reference_gs_list,
            start_s=start_s,
            end_s=end_s,
        )

        first_reference_band = True

        for _, row in clipped_ref.iterrows():
            label = "REF walking" if first_reference_band else None

            signal_axis.axvspan(
                float(row["start_s"]),
                float(row["end_s"]),
                alpha=0.15,
                label=label,
            )

            first_reference_band = False

    for col in ["acc_vt", "acc_ml", "acc_ap"]:
        signal_axis.plot(
            segment_time,
            trial_df.loc[segment_mask, col].to_numpy(dtype=float),
            linewidth=0.8,
            label=col,
        )

    signal_axis.set_ylabel("Acceleration")
    signal_axis.set_title(f"GSD single-trial visual check | {trial_name}")
    signal_axis.grid(True, alpha=0.3)
    signal_axis.legend(loc="upper right")

    row_names = list(algorithms.keys())

    if not row_names:
        row_names = ["No model output"]

    y_positions = np.arange(len(row_names))

    bands_axis.set_yticks(y_positions)
    bands_axis.set_yticklabels(row_names)
    bands_axis.set_xlabel("Time [s]")
    bands_axis.set_xlim(start_s, end_s)
    bands_axis.set_ylim(-0.75, len(row_names) - 0.25)
    bands_axis.grid(True, axis="x", alpha=0.3)

    row_index = 0

    for algorithm_name, algorithm in algorithms.items():
        clipped_gs = clip_gs_list_to_segment(
            algorithm.gs_list_,
            start_s=start_s,
            end_s=end_s,
        )

        for _, row in clipped_gs.head(max_bands).iterrows():
            bands_axis.broken_barh(
                [(float(row["start_s"]), float(row["end_s"] - row["start_s"]))],
                (row_index - 0.35, 0.7),
            )

        row_index += 1

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"[SAVED] {output_path}")


def save_algorithm_tables(
    algorithms: dict[str, object],
    *,
    output_dir: Path,
    trial_name: str,
) -> None:
    """Save window_detections_ and gs_list_ tables for each algorithm."""

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    safe_trial_name = trial_name.replace(" ", "_").replace("/", "_")

    for algorithm_name, algorithm in algorithms.items():
        window_path = tables_dir / f"{safe_trial_name}_{algorithm_name}_window_detections.csv"
        gs_path = tables_dir / f"{safe_trial_name}_{algorithm_name}_gs_list.csv"

        algorithm.window_detections_.to_csv(window_path, index=False)
        algorithm.gs_list_.to_csv(gs_path, index=False)

        print(f"[SAVED] {window_path}")
        print(f"[SAVED] {gs_path}")


def process_trial(
    *,
    trial_name: str,
    trial_csv: Path,
    args: argparse.Namespace,
    algorithm_names: list[str],
    output_dir: Path,
) -> None:
    """Load, standardize, run algorithms, optionally reference, plot and save."""

    print("\n" + "=" * 80)
    print(f"[TRIAL] {trial_name}")
    print(f"[PATH]  {trial_csv}")
    print("=" * 80)

    trial_df = load_trial_csv(trial_csv)
    trial_df = parse_time_column_to_seconds(
        trial_df,
        sampling_rate_hz=args.sampling_rate_hz,
    )

    trial_df, column_selection = standardize_single_trial_imu(
        trial_df,
        sampling_rate_hz=args.sampling_rate_hz,
        sensor_prefix=args.sensor_prefix,
        acc_columns=args.acc_columns,
        gyr_columns=args.gyr_columns,
        acc_input_unit=args.acc_input_unit,
        acc_output_unit=args.acc_output_unit,
        gyr_scale_factor=args.gyr_scale_factor,
        static_duration_s=args.static_duration_s,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
    )

    print(
        f"[INFO] Full trial samples={len(trial_df)} | "
        f"duration={trial_df['time_s'].max():.2f} s"
    )
    print(f"[INFO] selected IMU source: {column_selection.source}")

    algorithms = run_algorithms(
        trial_df,
        algorithm_names=algorithm_names,
        sampling_rate_hz=args.sampling_rate_hz,
        min_sequence_duration_s=args.min_sequence_duration_s,
        merge_gap_s=args.merge_gap_s,
        exclude_low_quality_windows=args.exclude_low_quality_windows,
    )

    reference_gs_list = None

    if args.show_reference:
        try:
            reference_gs_list = build_reference_gs_list(
                trial_df,
                reference_column=args.reference_column,
                walking_labels=args.reference_walking_labels,
                sampling_rate_hz=args.sampling_rate_hz,
            )

            print(
                "[INFO] Reference walking labels: "
                + ", ".join(sorted(args.reference_walking_labels))
            )
            print(f"[INFO] Reference walking sequences={len(reference_gs_list)}")

            reference_path = (
                output_dir
                / "tables"
                / f"{trial_name}_reference_gs_list.csv"
            )
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_gs_list.to_csv(reference_path, index=False)
            print(f"[SAVED] {reference_path}")

        except KeyError as exc:
            print(f"[WARNING] Reference not available: {exc}")
            reference_gs_list = None

    start_s = float(args.start_s)

    if args.duration_s is None:
        end_s = float(trial_df["time_s"].max())
    else:
        end_s = start_s + float(args.duration_s)

    plot_path = output_dir / f"{trial_name}_gsd_single_trial.png"

    plot_single_trial(
        trial_df,
        trial_name=trial_name,
        algorithms=algorithms,
        reference_gs_list=reference_gs_list,
        start_s=start_s,
        end_s=end_s,
        output_path=plot_path,
        max_bands=args.max_bands,
    )

    save_algorithm_tables(
        algorithms,
        output_dir=output_dir,
        trial_name=trial_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pd-trial-csv",
        type=str,
        default=str(DEFAULT_PD_TRIAL_CSV),
        help="PD/example trial CSV. Use 'none' to skip.",
    )

    parser.add_argument(
        "--control-trial-csv",
        type=str,
        default=str(DEFAULT_CONTROL_TRIAL_CSV),
        help="Control/example trial CSV. Use 'none' to skip.",
    )

    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=100.0,
        help="Sampling frequency of the input trial(s).",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["all", "svm", "rf", "knn", "lr", "gnb", "rb", "cnn"],
        help="GSD algorithm to run. Use 'all' to run every available algorithm.",
    )


    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=["all", "svm", "rf", "knn", "lr", "gnb", "rb", "cnn"],
        help=(
            "One or more GSD algorithms to run. This has priority over "
            "--algorithm. Examples: --algorithms cnn rb, --algorithms svm rf cnn."
        ),
    )


    parser.add_argument(
        "--sensor-prefix",
        type=str,
        default=None,
        help=(
            "Optional sensor prefix to select a specific IMU sensor, e.g. "
            "Forehead, Head, LeftEar. Use this when the file contains multiple sensors."
        ),
    )

    parser.add_argument(
        "--acc-columns",
        nargs=3,
        default=None,
        help="Explicit accelerometer columns in X Y Z order.",
    )

    parser.add_argument(
        "--gyr-columns",
        nargs=3,
        default=None,
        help="Explicit gyroscope columns in X Y Z order.",
    )

    parser.add_argument(
        "--acc-input-unit",
        type=str,
        default="auto",
        choices=["auto", "g", "m/s2", "m/s^2"],
        help="Input accelerometer unit passed to preprocessing.",
    )

    parser.add_argument(
        "--acc-output-unit",
        type=str,
        default="m/s2",
        choices=["g", "m/s2", "m/s^2"],
        help="Output accelerometer unit after preprocessing.",
    )

    parser.add_argument(
        "--gyr-scale-factor",
        type=float,
        default=1.0,
        help="Scale factor applied to gyroscope channels before preprocessing.",
    )

    parser.add_argument(
        "--static-duration-s",
        type=float,
        default=1.0,
        help="Initial static duration used for gravity alignment.",
    )

    parser.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        default=15.0,
        help="Final low-pass cutoff frequency.",
    )

    parser.add_argument(
        "--min-sequence-duration-s",
        type=float,
        default=0.0,
        help="Discard detected gait sequences shorter than this duration.",
    )

    parser.add_argument(
        "--merge-gap-s",
        type=float,
        default=0.0,
        help="Merge consecutive gait sequences separated by at most this gap.",
    )

    parser.add_argument(
        "--exclude-low-quality-windows",
        action="store_true",
        help=(
            "After detection, remove windows that contain at least one sample "
            "with imu_quality == False, then rebuild gs_list_."
        ),
    )

    parser.add_argument(
        "--start-s",
        type=float,
        default=0.0,
        help="Plot start time in seconds.",
    )

    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Plot duration in seconds. Use no value only by editing default.",
    )

    parser.add_argument(
        "--max-bands",
        type=int,
        default=10,
        help="Maximum number of gait-sequence bands shown per row.",
    )

    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="Display optional binary reference from the reference label column.",
    )

    parser.add_argument(
        "--reference-column",
        type=str,
        default="GeneralEvent",
        help="Column used to build the optional binary reference.",
    )

    parser.add_argument(
        "--reference-walking-labels",
        nargs="+",
        default=[
            "walk",
            "walking",
            "stair",
            "stairs",
            "ascend",
            "ascending",
            "descend",
            "descending",
            "upstairs",
            "downstairs",
            "turn",
            "turning",
        ],
        help="Labels considered walking-like in the optional binary reference.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/gait_sequence_detection/single_trial_visual_check",
        help="Output folder for plots and tables.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = resolve_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithm_names = select_algorithm_names(
        algorithm=args.algorithm,
        algorithms=args.algorithms,
    )

    trial_specs = [
        ("pd_freewalk", resolve_optional_trial_path(args.pd_trial_csv)),
        ("control_freewalk", resolve_optional_trial_path(args.control_trial_csv)),
    ]

    for trial_name, trial_csv in trial_specs:
        if trial_csv is None:
            continue

        process_trial(
            trial_name=trial_name,
            trial_csv=trial_csv,
            args=args,
            algorithm_names=algorithm_names,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
