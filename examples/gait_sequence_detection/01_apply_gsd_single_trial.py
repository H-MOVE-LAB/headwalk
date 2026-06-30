# Minimal single-trial Gait Sequence Detection example.
#
# This script demonstrates the raw inference workflow:
#
#     raw trial
#         -> generic IMU column selection
#         -> preprocessing
#         -> algorithm.detect(...)
#         -> algorithm.gs_list_
#
# The input adapter is schema-based, not dataset-name-based. If the raw file
# contains multiple IMU sensors, the user should specify the head-worn sensor
# with --sensor-prefix or with explicit --acc-columns / --gyr-columns.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
EXAMPLE_DIR = Path(__file__).resolve().parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


DEFAULT_TRIAL_CSV = (
    PROJECT_ROOT
    / "example_data"
    / "gait_sequence_detection"
    / "weargaitpd_pd_freewalk_nls192.csv"
)


from _single_trial_input import (  # noqa: E402
    load_trial_csv,
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


def select_algorithm_names(algorithm: str) -> list[str]:
    """Return the list of algorithm keys to run."""

    if algorithm == "all":
        return ["svm", "rf", "knn", "lr", "gnb", "rb", "cnn"]

    return [algorithm]


def compute_window_quality_flags(
    window_detections: pd.DataFrame,
    *,
    sample_quality,
) -> pd.DataFrame:
    """
    Add quality metadata to a window-level detection table.

    This function is used for raw inference when a sample-wise quality mask is
    available from preprocessing.

    It does not use labels and does not require a reference system.

    A window is marked as high quality only if every sample inside the window has
    sample_quality == True.
    """

    if sample_quality is None:
        out = window_detections.copy()
        out["window_quality_fraction"] = 1.0
        out["window_is_high_quality"] = True
        return out

    import numpy as np

    quality = np.asarray(sample_quality, dtype=bool)

    out = window_detections.copy()

    quality_fractions = []
    high_quality_flags = []

    required_columns = ["window_start_sample", "window_end_sample"]
    missing = [col for col in required_columns if col not in out.columns]

    if missing:
        raise KeyError(
            "Cannot apply sample quality to window detections. "
            f"Missing columns: {missing}"
        )

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
    Remove low-quality windows from an already executed GSD algorithm.

    The model is still evaluated on the complete preprocessed trial. This helper
    only removes unreliable windows from the final exported detection table and
    rebuilds gs_list_ from the remaining high-quality windows.

    This is useful for raw inference when preprocessing has identified original
    long IMU dropout regions.
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
    """Run selected algorithms and return fitted algorithm instances."""

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

        print("\n" + "=" * 80)
        print(f"[INFO] Running {algorithm.__class__.__name__}")
        print("=" * 80)

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
            f"[INFO] windows={len(algorithm.window_detections_)} | "
            f"gait_sequences={len(algorithm.gs_list_)}"
        )

        if algorithm.gs_list_.empty:
            print("[INFO] gs_list_ is empty.")
        else:
            print("[INFO] gs_list_:")
            print(algorithm.gs_list_.to_string(index=False))

        results[algorithm.__class__.__name__] = algorithm

    return results


def save_outputs(
    algorithms: dict[str, object],
    *,
    output_dir: Path,
    trial_name: str,
) -> None:
    """Save window-level detections and gs_list_ tables."""

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_trial_name = trial_name.replace(" ", "_").replace("/", "_")

    for algorithm_name, algorithm in algorithms.items():
        window_path = output_dir / f"{safe_trial_name}_{algorithm_name}_window_detections.csv"
        gs_path = output_dir / f"{safe_trial_name}_{algorithm_name}_gs_list.csv"

        algorithm.window_detections_.to_csv(window_path, index=False)
        algorithm.gs_list_.to_csv(gs_path, index=False)

        print(f"[SAVED] {window_path}")
        print(f"[SAVED] {gs_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trial-csv",
        type=str,
        default=str(DEFAULT_TRIAL_CSV),
        help="Input single-trial CSV. Defaults to the compact PD FreeWalk example trial.",
    )

    parser.add_argument(
        "--trial-name",
        type=str,
        default="single_trial",
        help="Name used when saving optional output tables.",
    )

    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        default=100.0,
        help="Sampling frequency of the input trial.",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="svm",
        choices=["all", "svm", "rf", "knn", "lr", "gnb", "rb", "cnn"],
        help="GSD algorithm to run. Use 'all' to run every available algorithm.",
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
        help=(
            "Explicit accelerometer columns in X Y Z order. "
            "Use with --gyr-columns for custom schemas."
        ),
    )

    parser.add_argument(
        "--gyr-columns",
        nargs=3,
        default=None,
        help=(
            "Explicit gyroscope columns in X Y Z order. "
            "Use with --acc-columns for custom schemas."
        ),
    )

    parser.add_argument(
        "--acc-input-unit",
        type=str,
        default="auto",
        choices=["auto", "g", "m/s2", "m/s^2"],
        help="Input accelerometer unit passed to the preprocessing function.",
    )

    parser.add_argument(
        "--acc-output-unit",
        type=str,
        default="m/s2",
        choices=["g", "m/s2", "m/s^2"],
        help="Output accelerometer unit used after preprocessing.",
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
            "with imu_quality == False, then rebuild gs_list_. This option uses "
            "only raw-signal quality information and does not require labels."
        ),
    )

    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save window_detections_ and gs_list_ tables.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/gait_sequence_detection/apply_single_trial",
        help="Output folder used only when --save-outputs is enabled.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trial_csv = resolve_cli_path(args.trial_csv)
    output_dir = resolve_cli_path(args.output_dir)

    print("\n" + "=" * 80)
    print("[GSD SINGLE-TRIAL APPLY EXAMPLE]")
    print("=" * 80)
    print(f"[INFO] trial_csv: {trial_csv}")
    print(f"[INFO] algorithm: {args.algorithm}")
    print(f"[INFO] sampling_rate_hz: {args.sampling_rate_hz}")

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
        f"[INFO] trial samples={len(trial_df)} | "
        f"duration={trial_df['time_s'].max():.2f} s"
    )
    print(f"[INFO] selected IMU source: {column_selection.source}")

    algorithm_names = select_algorithm_names(args.algorithm)

    algorithms = run_algorithms(
        trial_df,
        algorithm_names=algorithm_names,
        sampling_rate_hz=args.sampling_rate_hz,
        min_sequence_duration_s=args.min_sequence_duration_s,
        merge_gap_s=args.merge_gap_s,
        exclude_low_quality_windows=args.exclude_low_quality_windows,
    )

    if args.save_outputs:
        save_outputs(
            algorithms,
            output_dir=output_dir,
            trial_name=args.trial_name,
        )


if __name__ == "__main__":
    main()
