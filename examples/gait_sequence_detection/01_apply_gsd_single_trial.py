# Minimal single-trial Gait Sequence Detection example.
#
# This script demonstrates the raw inference workflow:
#
#     raw trial
#         -> preprocessing
#         -> algorithm.detect(...)
#         -> algorithm.gs_list_
#
# It does not build a reference and does not plot results. It is intended as the
# simplest example of how to call the GSD algorithms on a single trial.

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_TRIAL_CSV = (
    PROJECT_ROOT
    / "example_data"
    / "gait_sequence_detection"
    / "weargaitpd_pd_freewalk_nls192.csv"
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

from headwalk.gait_sequence_detection.preprocessing import (  # noqa: E402
    preprocess_head_imu_trial,
)


STANDARD_COLUMNS = [
    "acc_vt",
    "acc_ml",
    "acc_ap",
    "gyr_vt",
    "gyr_ml",
    "gyr_ap",
]


ALGORITHM_CLASSES = {
    "svm": GsdSvm,
    "rf": GsdRandomForest,
    "knn": GsdKnn,
    "lr": GsdLogisticRegression,
    "gnb": GsdGaussianNaiveBayes,
    "rb": GsdRuleBased,
    "cnn": GsdCnn1D,
}


def resolve_cli_path(path_text: str | None) -> Path | None:
    """Accept both Windows paths and Git-Bash-like /f/... paths."""

    if path_text is None:
        return None

    text = str(path_text).strip().strip('"').strip("'")

    if re.match(r"^/[A-Za-z]/", text):
        drive = text[1].upper()
        rest = text[2:]
        return Path(f"{drive}:{rest}")

    return Path(text)


def normalize_column_name(name: str) -> str:
    """Normalize column names to lowercase snake-case."""

    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .lower()
    )


def load_trial_csv(path: Path) -> pd.DataFrame:
    """Load a single-trial CSV and normalize column names."""

    if not path.exists():
        raise FileNotFoundError(f"Trial CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]

    return df


def parse_time_column_to_seconds(
    df: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Create a numeric time_s column when possible."""

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


def standardize_head_imu_columns(
    df: pd.DataFrame,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """
    Return a dataframe containing the standardized GSD IMU columns.

    If the dataframe already contains:
        acc_vt, acc_ml, acc_ap, gyr_vt, gyr_ml, gyr_ap

    these columns are used directly.

    Otherwise, the function expects compact WearGaitPD-style Forehead columns:
        Forehead_Acc_X/Y/Z
        Forehead_Gyr_X/Y/Z

    and applies the current head-IMU preprocessing before creating the
    standardized columns.
    """

    df = df.copy()

    if all(col in df.columns for col in STANDARD_COLUMNS):
        for col in STANDARD_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print("[INFO] Using existing standardized IMU columns.")
        return df

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
            "Cannot standardize the input trial. Expected either the "
            "standardized columns "
            f"{STANDARD_COLUMNS} or compact Forehead IMU columns. "
            f"Missing raw columns: {missing_sources}"
        )

    acc_raw = df[raw_acc_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    gyr_raw = df[raw_gyr_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    print("[INFO] Applying current head-IMU preprocessing.")
    print("[INFO] Raw Acc order: Forehead_Acc_X, Forehead_Acc_Y, Forehead_Acc_Z")
    print("[INFO] Raw Gyr order: Forehead_Gyr_X, Forehead_Gyr_Y, Forehead_Gyr_Z")
    print("[INFO] Acc unit kept as m/s^2 for the compact WearGaitPD example data.")

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

    df["acc_vt"] = X[:, 0]
    df["acc_ml"] = X[:, 1]
    df["acc_ap"] = X[:, 2]
    df["gyr_vt"] = X[:, 3]
    df["gyr_ml"] = X[:, 4]
    df["gyr_ap"] = X[:, 5]

    df["imu_quality"] = preprocessing_result.imu_quality

    debug = preprocessing_result.preprocessing_debug_info

    step_frequency = debug.get("step_frequency_hz")
    m3_cutoff = debug.get("m3_cutoff_hz")

    if step_frequency is not None and m3_cutoff is not None:
        print(
            "[INFO] Preprocessing completed | "
            f"step_frequency={step_frequency:.3f} Hz | "
            f"M3 cutoff={m3_cutoff:.3f} Hz"
        )
    else:
        print("[INFO] Preprocessing completed.")

    return df


def select_algorithm_names(algorithm: str) -> list[str]:
    """Return the list of algorithm keys to run."""

    if algorithm == "all":
        return ["svm", "rf", "knn", "lr", "gnb", "rb", "cnn"]

    return [algorithm]


def run_algorithms(
    trial_df: pd.DataFrame,
    *,
    algorithm_names: list[str],
    sampling_rate_hz: float,
    min_sequence_duration_s: float,
    merge_gap_s: float,
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
    trial_df = standardize_head_imu_columns(
        trial_df,
        sampling_rate_hz=args.sampling_rate_hz,
    )

    print(
        f"[INFO] trial samples={len(trial_df)} | "
        f"duration={trial_df['time_s'].max():.2f} s"
    )

    algorithm_names = select_algorithm_names(args.algorithm)

    algorithms = run_algorithms(
        trial_df,
        algorithm_names=algorithm_names,
        sampling_rate_hz=args.sampling_rate_hz,
        min_sequence_duration_s=args.min_sequence_duration_s,
        merge_gap_s=args.merge_gap_s,
    )

    if args.save_outputs:
        save_outputs(
            algorithms,
            output_dir=output_dir,
            trial_name=args.trial_name,
        )


if __name__ == "__main__":
    main()
