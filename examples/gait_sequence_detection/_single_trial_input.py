"""
Single-trial input utilities for Gait Sequence Detection examples.

This module converts a raw CSV trial into the standard six-channel schema used
by the current GSD algorithms:

    acc_vt, acc_ml, acc_ap, gyr_vt, gyr_ml, gyr_ap

The module is intentionally schema-based rather than dataset-name-based.

This means that it does not contain dataset-specific branches such as:

    if dataset == "WearGaitPD"
    if dataset == "Tobii"
    if dataset == "TOWalk"

Instead, it follows this logic:

1. if the standardized GSD columns already exist, use them directly;
2. otherwise, if explicit acc/gyr column names are provided, use them;
3. otherwise, if a sensor prefix is provided, use that sensor;
4. otherwise, try to infer a complete head-worn IMU sensor;
5. if the choice is ambiguous, raise a clear error and ask the user to specify
   --sensor-prefix or --acc-columns / --gyr-columns.

This keeps the example flexible while avoiding unsafe automatic choices when a
file contains multiple sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


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


HEAD_SENSOR_TOKENS = [
    "forehead",
    "head",
    "left_ear",
    "leftear",
    "right_ear",
    "rightear",
    "ear",
    "tobii",
    "glasses",
    "smart_glasses",
]


ACC_NAME_TOKENS = [
    "acc",
    "accelerometer",
    "acceleration",
]


GYR_NAME_TOKENS = [
    "gyr",
    "gyro",
    "gyroscope",
]


AXES = ["x", "y", "z"]


@dataclass(frozen=True)
class ImuColumnSelection:
    """Description of the selected raw IMU columns."""

    acc_columns: list[str]
    gyr_columns: list[str]
    source: str
    used_standard_columns: bool = False


def resolve_cli_path(path_text: str | None) -> Path | None:
    """
    Convert a CLI path into a pathlib Path.

    The function accepts both normal Windows paths and Git-Bash-like paths such
    as /f/headwalk/file.csv.
    """

    if path_text is None:
        return None

    text = str(path_text).strip().strip('"').strip("'")

    if re.match(r"^/[A-Za-z]/", text):
        drive = text[1].upper()
        rest = text[2:]
        return Path(f"{drive}:{rest}")

    return Path(text)


def normalize_column_name(name: str) -> str:
    """
    Normalize a column name to lowercase snake-case.

    Examples:
        Forehead_Acc_X       -> forehead_acc_x
        L Foot Contact       -> l_foot_contact
        Gyroscope-X          -> gyroscope_x
    """

    text = str(name).strip().lower()
    text = text.replace("(", "_").replace(")", "_")
    text = text.replace("[", "_").replace("]", "_")
    text = text.replace("/", "_").replace("\\", "_")
    text = text.replace("-", "_").replace(":", "_")
    text = text.replace(".", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")

    return text


def normalize_cli_columns(columns: Iterable[str] | None) -> list[str] | None:
    """Normalize an optional list of CLI column names."""

    if columns is None:
        return None

    return [normalize_column_name(col) for col in columns]


def load_trial_csv(path: Path) -> pd.DataFrame:
    """
    Load a single-trial CSV and normalize all column names.

    Original column names are not preserved in the returned DataFrame. The
    normalized names make matching robust across spaces, dashes and case
    differences.
    """

    if path is None:
        raise ValueError("trial CSV path is None.")

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
    """
    Ensure that the DataFrame contains a numeric time_s column.

    Priority:
    1. use an existing time_s column;
    2. parse a generic time column, including strings such as "12.34 sec";
    3. otherwise, reconstruct time from sample index and sampling frequency.
    """

    df = df.copy()

    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")

        if df["time_s"].notna().any():
            return df

    for candidate in ["time", "timestamp", "timestamps"]:
        if candidate in df.columns:
            extracted = (
                df[candidate]
                .astype(str)
                .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            )
            df["time_s"] = pd.to_numeric(extracted, errors="coerce")

            if df["time_s"].notna().any():
                return df

    df["time_s"] = np.arange(len(df), dtype=float) / float(sampling_rate_hz)

    return df


def _columns_exist(df: pd.DataFrame, columns: list[str]) -> bool:
    """Return True if all columns are present."""

    return all(col in df.columns for col in columns)


def _find_axis_column(
    columns: set[str],
    *,
    prefix: str | None,
    signal_tokens: list[str],
    axis: str,
) -> str | None:
    """
    Find one axis column for a given signal family and optional sensor prefix.

    Supported patterns include:
        prefix_acc_x
        prefix_acceleration_x
        prefix_accelerometer_x
        acc_x
        acceleration_x
        accelerometer_x
    """

    candidates = []

    if prefix:
        for signal_token in signal_tokens:
            candidates.append(f"{prefix}_{signal_token}_{axis}")
            candidates.append(f"{prefix}_{axis}_{signal_token}")

    for signal_token in signal_tokens:
        candidates.append(f"{signal_token}_{axis}")
        candidates.append(f"{axis}_{signal_token}")

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def _find_complete_sensor_candidate(
    columns: set[str],
    *,
    prefix: str | None,
) -> ImuColumnSelection | None:
    """Find a complete Acc XYZ + Gyr XYZ candidate for one prefix."""

    acc_columns = []
    gyr_columns = []

    for axis in AXES:
        acc_col = _find_axis_column(
            columns,
            prefix=prefix,
            signal_tokens=ACC_NAME_TOKENS,
            axis=axis,
        )

        gyr_col = _find_axis_column(
            columns,
            prefix=prefix,
            signal_tokens=GYR_NAME_TOKENS,
            axis=axis,
        )

        if acc_col is None or gyr_col is None:
            return None

        acc_columns.append(acc_col)
        gyr_columns.append(gyr_col)

    source = "unprefixed IMU columns" if prefix is None else f"sensor prefix '{prefix}'"

    return ImuColumnSelection(
        acc_columns=acc_columns,
        gyr_columns=gyr_columns,
        source=source,
        used_standard_columns=False,
    )


def _discover_prefixed_candidates(df: pd.DataFrame) -> list[ImuColumnSelection]:
    """
    Discover complete prefixed IMU candidates.

    A candidate is considered complete only when it has:
        Acc X/Y/Z and Gyr X/Y/Z.
    """

    columns = set(df.columns)
    possible_prefixes = set()

    pattern = re.compile(
        r"^(?P<prefix>.+?)_"
        r"(?P<signal>acc|accelerometer|acceleration|gyr|gyro|gyroscope)_"
        r"(?P<axis>x|y|z)$"
    )

    for col in df.columns:
        match = pattern.match(col)
        if match is not None:
            possible_prefixes.add(match.group("prefix"))

    candidates = []

    for prefix in sorted(possible_prefixes):
        candidate = _find_complete_sensor_candidate(columns, prefix=prefix)

        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _is_head_like_source(source: str) -> bool:
    """Return True if a candidate source looks like a head-worn sensor."""

    source_norm = normalize_column_name(source)

    return any(token in source_norm for token in HEAD_SENSOR_TOKENS)


def describe_imu_candidates(df: pd.DataFrame) -> str:
    """Return a human-readable summary of complete IMU candidates."""

    candidates = _discover_prefixed_candidates(df)

    unprefixed = _find_complete_sensor_candidate(
        set(df.columns),
        prefix=None,
    )

    if unprefixed is not None:
        candidates = [unprefixed] + candidates

    if not candidates:
        return "No complete Acc XYZ + Gyr XYZ candidate was found."

    lines = ["Complete IMU candidates found:"]

    for candidate in candidates:
        lines.append(
            "  - "
            f"{candidate.source} | "
            f"acc={candidate.acc_columns} | "
            f"gyr={candidate.gyr_columns}"
        )

    return "\n".join(lines)


def select_imu_columns(
    df: pd.DataFrame,
    *,
    sensor_prefix: str | None = None,
    acc_columns: list[str] | None = None,
    gyr_columns: list[str] | None = None,
) -> ImuColumnSelection:
    """
    Select the IMU columns to be used by the GSD preprocessing.

    Selection priority:
    1. explicit --acc-columns and --gyr-columns;
    2. existing standardized columns;
    3. explicit --sensor-prefix;
    4. exactly one head-like complete IMU candidate;
    5. exactly one complete IMU candidate;
    6. otherwise, raise a clear ambiguity error.
    """

    normalized_acc_columns = normalize_cli_columns(acc_columns)
    normalized_gyr_columns = normalize_cli_columns(gyr_columns)

    if normalized_acc_columns is not None or normalized_gyr_columns is not None:
        if normalized_acc_columns is None or normalized_gyr_columns is None:
            raise ValueError(
                "Both --acc-columns and --gyr-columns must be provided together."
            )

        if len(normalized_acc_columns) != 3 or len(normalized_gyr_columns) != 3:
            raise ValueError(
                "--acc-columns and --gyr-columns must each contain exactly 3 columns."
            )

        missing = [
            col for col in normalized_acc_columns + normalized_gyr_columns
            if col not in df.columns
        ]

        if missing:
            raise KeyError(
                "Explicit IMU columns were provided, but some columns are "
                f"missing after normalization: {missing}"
            )

        return ImuColumnSelection(
            acc_columns=normalized_acc_columns,
            gyr_columns=normalized_gyr_columns,
            source="explicit CLI columns",
            used_standard_columns=False,
        )

    if _columns_exist(df, STANDARD_COLUMNS):
        return ImuColumnSelection(
            acc_columns=STANDARD_COLUMNS[:3],
            gyr_columns=STANDARD_COLUMNS[3:],
            source="existing standardized GSD columns",
            used_standard_columns=True,
        )

    columns = set(df.columns)

    if sensor_prefix:
        prefix = normalize_column_name(sensor_prefix)
        candidate = _find_complete_sensor_candidate(columns, prefix=prefix)

        if candidate is None:
            raise KeyError(
                f"No complete Acc XYZ + Gyr XYZ columns found for "
                f"--sensor-prefix {sensor_prefix!r} after normalization to {prefix!r}.\n"
                + describe_imu_candidates(df)
            )

        return candidate

    candidates = _discover_prefixed_candidates(df)

    head_like_candidates = [
        candidate for candidate in candidates
        if _is_head_like_source(candidate.source)
    ]

    if len(head_like_candidates) == 1:
        return head_like_candidates[0]

    if len(head_like_candidates) > 1:
        raise ValueError(
            "Multiple complete head-like IMU candidates were found. "
            "Please choose one with --sensor-prefix or provide explicit "
            "--acc-columns / --gyr-columns.\n"
            + describe_imu_candidates(df)
        )

    unprefixed = _find_complete_sensor_candidate(columns, prefix=None)

    all_candidates = []
    if unprefixed is not None:
        all_candidates.append(unprefixed)

    all_candidates.extend(candidates)

    if len(all_candidates) == 1:
        return all_candidates[0]

    if len(all_candidates) == 0:
        raise KeyError(
            "No complete Acc XYZ + Gyr XYZ IMU sensor could be inferred.\n"
            "Provide --sensor-prefix or explicit --acc-columns / --gyr-columns.\n"
            + describe_imu_candidates(df)
        )

    raise ValueError(
        "Multiple complete IMU candidates were found, but none could be "
        "selected safely as the head-worn sensor. Please use --sensor-prefix "
        "or explicit --acc-columns / --gyr-columns.\n"
        + describe_imu_candidates(df)
    )


def standardize_single_trial_imu(
    df: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    sensor_prefix: str | None = None,
    acc_columns: list[str] | None = None,
    gyr_columns: list[str] | None = None,
    acc_input_unit: str = "auto",
    acc_output_unit: str = "m/s2",
    gyr_scale_factor: float = 1.0,
    static_duration_s: float = 1.0,
    lowpass_cutoff_hz: float = 15.0,
    print_schema: bool = True,
) -> tuple[pd.DataFrame, ImuColumnSelection]:
    """
    Convert a raw trial DataFrame into the standardized GSD IMU schema.

    Returns:
        standardized_df, column_selection
    """

    df = df.copy()

    selection = select_imu_columns(
        df,
        sensor_prefix=sensor_prefix,
        acc_columns=acc_columns,
        gyr_columns=gyr_columns,
    )

    if print_schema:
        print("[INFO] IMU column selection:")
        print(f"       source: {selection.source}")
        print(f"       acc: {selection.acc_columns}")
        print(f"       gyr: {selection.gyr_columns}")

    if selection.used_standard_columns:
        for col in STANDARD_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print("[INFO] Input already contains standardized GSD IMU columns.")
        return df, selection

    acc_raw = df[selection.acc_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    gyr_raw = df[selection.gyr_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    print("[INFO] Applying current head-IMU preprocessing.")
    print(f"[INFO] acc_input_unit={acc_input_unit} | acc_output_unit={acc_output_unit}")
    print(f"[INFO] gyr_scale_factor={gyr_scale_factor}")

    preprocessing_result = preprocess_head_imu_trial(
        acc_raw=acc_raw,
        gyr_raw=gyr_raw,
        fs=sampling_rate_hz,
        static_duration_s=static_duration_s,
        static_reference_mask=None,
        fixed_axis_transform=None,
        acc_input_unit=acc_input_unit,
        acc_output_unit=acc_output_unit,
        gyr_scale_factor=gyr_scale_factor,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
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

    return df, selection
