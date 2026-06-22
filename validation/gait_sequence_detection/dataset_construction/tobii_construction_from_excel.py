"""
Construction script for the Tobii / VIXIONE raw IMU dataset using manual Excel labels.

Purpose:
- read raw Tobii IMU streams from each raw recording folder (imudata.gz or extracted imudata file);
- read manual temporal annotations from the Excel workbook;
- assign sample-wise GSD labels and metadata;
- optionally resample IMU signals to a fixed sampling frequency, default 100 Hz;
- create trial-level and window-level datasets compatible with the WearGait-style pipeline.

Important:
- The Excel workbook remains the only manual labeling source.
- The script does not require Tobii Pro Lab CSV exports.
- The raw timestamp folder name is kept as the stable recording identifier.
"""

import argparse
import gzip
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by moving upward from the current script.

    This avoids hard-coded user-specific Windows paths and allows the same file
    to run on different PCs, provided that the project contains src/ and data/.
    """

    current = start_path.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "data").exists():
            return parent

    raise FileNotFoundError(
        "Project root not found. Expected a parent folder containing src/ and data/."
    )


PROJECT_ROOT = find_project_root(Path(__file__))
RESULTS_ROOT = PROJECT_ROOT / "results"
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.rotations import align_imu_to_gravity


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "dataset_VIXIONE_working",
    PROJECT_ROOT / "data" / "dataset_VIXIONE",
    PROJECT_ROOT / "data" / "Tobii_dataset",
]

DEFAULT_RESULTS_DIR = RESULTS_ROOT / "Tobii_dataset"
RAW_RECORDINGS_FOLDER_NAME = "File grezzi con cartelle complete"

TARGET_FS = 100.0
WINDOW_LABEL_MIN_FRACTION = 0.80
PATH_TYPE_MIN_FRACTION = 0.80

STATIC_CONTEXT_MIN_FRACTION = 0.10
GAIT_TRANSIENT_DURATION_S = 1.0

# Minimum fraction required for both static and walking samples to consider a
# Tobii window as a transition window.
# This is specific to manually labelled Tobii data, where transition windows
# are detected from the temporal order of valid labels inside the window.
TRANSITION_MIN_CLASS_FRACTION = 0.10

# Human-readable description of how Tobii transition metadata are built.
# The window is marked as Static2Walking or Walking2Static only if both static
# and walking cover at least 10% of the window and the first/last valid labels
# indicate a change of state.
TRANSITION_DEFINITION = (
    "Boundary-aware first/last valid label with >=10% static and >=10% walking"
)

WINDOW_CONFIGS = [
    ("1s_50p_overlap", 100, 50),
    ("2s_50p_overlap", 200, 100),
    ("5s_50p_overlap", 500, 250),
    ("10s_50p_overlap", 1000, 500),
]


# =============================================================================
# LABEL MAPS
# =============================================================================

GSD_LABEL_MAP = {
    "none": -1,
    "static": 0,
    "walking": 1,
}

ACTIVITY_DETAIL_MAP = {
    "none": -1,
    "static": 0,
    "walking": 1,
    "ascending": 2,
    "descending": 3,
}

STATIC_TYPE_MAP = {
    "none": -1,
    "static": 0,
    "standing": 1,
    "sitting": 2,
    "both": 3,
}

PATH_TYPE_MAP = {
    None: "none",
    0: "straight",
    1: "curved",
}

PATH_TYPE_NAME_TO_CODE = {
    "none": None,
    "straight": 0,
    "curved": 1,
}

INCLINE_MAP = {
    "none": -1,
    "level": 0,
    "up": 1,
    "down": 2,
    "mixed": 3,
}

GAIT_PHASE_MAP = {
    "none": -1,
    "gait_initiation": 0,
    "steady_state": 1,
    "gait_termination": 2,
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def get_map_name(map_dict, value):
    """
    Convert a numeric code into its readable map key.
    """

    for key, val in map_dict.items():
        if val == value:
            return key

    return "unknown"


def make_project_relative_path(path: Path) -> str:
    """
    Store paths relative to the project root to keep metadata portable.
    """

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_dataset_root(explicit_root=None) -> Path:
    """
    Resolve the Tobii dataset root using either an explicit path or candidates.
    """

    if explicit_root is not None:
        dataset_root = Path(explicit_root).expanduser().resolve()
        if dataset_root.exists():
            return dataset_root
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Tobii dataset folder not found. Checked:\n" + checked
    )


def resolve_excel_path(dataset_root: Path, explicit_excel=None) -> Path:
    """
    Resolve the Excel annotation workbook.
    """

    if explicit_excel is not None:
        excel_path = Path(explicit_excel).expanduser().resolve()
        if excel_path.exists():
            return excel_path
        raise FileNotFoundError(f"Excel annotation file not found: {excel_path}")

    candidates = [
        dataset_root / "Tobii_recording_review_template.xlsx",
        PROJECT_ROOT / "data" / "Tobii_recording_review_template.xlsx",
        PROJECT_ROOT / "Tobii_recording_review_template.xlsx",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Excel annotation workbook not found. Checked:\n" + checked
    )


def normalize_text(value, default="none") -> str:
    """
    Normalize Excel text values while preserving human-readable details elsewhere.
    """

    if pd.isna(value):
        return default

    text = str(value).strip()

    if text == "":
        return default

    if text.lower() in ["nan", "none", "null"]:
        return default

    return text

def parse_excel_time_seconds(value, allow_end_marker=False):
    """
    Convert Excel manual time annotations to numeric seconds.

    This parser is intentionally more tolerant than pd.to_numeric because the
    workbook may be edited with Italian Excel, where decimal commas can be
    written as text values such as "12,35". Decimal points such as "12.35" are
    also accepted.

    Accepted examples:
    - 12.35
    - "12.35"
    - "12,35"
    - "12.35 s"
    - "00:01:12.35"
    - "end" only for end_time_s when allow_end_marker=True

    The "end" marker is converted to +inf here and then replaced later, during
    sample labeling, by the actual end of the current IMU recording.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if text == "":
        return np.nan

    text_lower = text.lower()

    # Allow explicit textual markers often used during manual video annotation.
    # This is useful when the first interval of a recording is written as
    # "start" or "inizio" instead of 0.0 in Excel.
    if text_lower in ["start", "begin", "beginning", "inizio"]:
        return 0.0

    # Allow the last interval to end at the actual end of the recording without
    # manually writing the exact video/IMU duration in Excel.
    if allow_end_marker and text_lower in ["end", "fine", "finish", "last", "ultimo"]:
        return np.inf

    if text_lower in ["nan", "none", "null", "na", "n/a"]:
        return np.nan

    # Italian decimal-comma support.
    text = text.replace(",", ".")

    # Optional time format support: HH:MM:SS(.sss) or MM:SS(.sss).
    if ":" in text:
        parts = text.split(":")

        try:
            parts = [float(part.strip()) for part in parts]
        except ValueError:
            return np.nan

        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]

        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]

        return np.nan

    # Remove common textual suffixes while keeping digits, sign and decimal point.
    text = text.replace("seconds", "").replace("second", "").replace("sec", "").replace("s", "")
    text = text.strip()

    try:
        return float(text)
    except ValueError:
        return np.nan


def remove_parenthetical_explanations(value, default="none") -> str:
    """
    Remove short explanatory notes written between parentheses in Excel.

    Example:
    - "static_head_yaw (no)" becomes "static_head_yaw"
    - "straight_walking_head_pitch (yes)" becomes "straight_walking_head_pitch"

    This lets the Excel file remain readable for manual work while keeping the
    constructed dataset clean and programmatically stable.
    """

    text = normalize_text(value, default=default)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.strip()

    if text == "":
        return default

    return text


def normalize_basic_category(value, default="none") -> str:
    """
    Normalize simple categorical values to lowercase underscore format.

    Parenthetical explanations used only for Excel readability are removed.
    """

    text = remove_parenthetical_explanations(value, default=default).lower().strip()
    text = text.replace(" ", "_").replace("-", "_")

    return text


# =============================================================================
# EXCEL LABEL NORMALIZATION
# =============================================================================

def normalize_label_name(value) -> str:
    """
    Normalize the main GSD label from Excel.
    """

    label = normalize_basic_category(value, default="none")

    if label in GSD_LABEL_MAP:
        return label

    raise ValueError(f"Unknown label_name value: {value}")


def normalize_static_type(value, label_name=None) -> str:
    """
    Convert Excel static_type detail into the canonical static-type class.

    Examples accepted from Excel:
    - standing static -> standing
    - standing yaw (no) -> standing
    - sitting -> sitting
    - none -> none
    """

    text = remove_parenthetical_explanations(value, default="none").lower()

    if label_name == "walking":
        return "none"

    if "sitting" in text or "sit" in text:
        return "sitting"

    if "standing" in text or "stand" in text:
        return "standing"

    if "both" in text:
        return "both"

    if text in ["static", "generic_static"]:
        return "static"

    if text in ["none", "nan", ""]:
        return "none"

    # Fallback: if the main label is static, keep it as generic static.
    if label_name == "static":
        return "static"

    return "none"


def normalize_path_type(value, label_name=None) -> str:
    """
    Convert Excel path_type detail into the canonical path-type class.

    Examples accepted from Excel:
    - straight simple -> straight
    - straight yaw (no) -> straight
    - straight pitch (yes) -> straight
    - curved, turn, u-turn -> curved
    - none -> none
    """

    text = remove_parenthetical_explanations(value, default="none").lower()

    if label_name == "static":
        return "none"

    if "curved" in text or "turn" in text or "u_turn" in text or "u-turn" in text:
        return "curved"

    if "straight" in text:
        return "straight"

    if text in ["none", "nan", ""]:
        return "none"

    return "none"


def normalize_incline(value, label_name=None) -> str:
    """
    Normalize incline metadata from Excel.
    """

    text = normalize_basic_category(value, default="none")

    if label_name == "static":
        return "none"

    if text in INCLINE_MAP:
        return text

    raise ValueError(f"Unknown incline value: {value}")


def normalize_gait_phase(value) -> str:
    """
    Normalize gait_stationarity values from Excel.
    """

    text = normalize_basic_category(value, default="none")

    aliases = {
        "initiation": "gait_initiation",
        "gait_initiation": "gait_initiation",
        "steady": "steady_state",
        "steady_state": "steady_state",
        "termination": "gait_termination",
        "gait_termination": "gait_termination",
        "none": "none",
    }

    return aliases.get(text, "none")


def normalize_transition_type(value) -> str:
    """
    Normalize transition-type values from Excel.
    """

    text = normalize_text(value, default="None").strip()

    if text in ["Static2Walking", "Walking2Static", "None"]:
        return text

    if text.lower() in ["none", "nan", ""]:
        return "None"

    return text


def infer_activity_detail(label_name: str, incline_name: str) -> str:
    """
    Convert main label and incline into activity detail.
    """

    if label_name == "none":
        return "none"

    if label_name == "static":
        return "static"

    if label_name == "walking" and incline_name == "up":
        return "ascending"

    if label_name == "walking" and incline_name == "down":
        return "descending"

    return "walking"


# =============================================================================
# EXCEL LOADING AND VALIDATION
# =============================================================================

def load_excel_annotations(excel_path: Path):
    """
    Load Recording_Review and Manual_Intervals from the Excel workbook.
    """

    recording_df = pd.read_excel(excel_path, sheet_name="Recording_Review")
    intervals_df = pd.read_excel(excel_path, sheet_name="Manual_Intervals")

    recording_df.columns = [str(col).strip() for col in recording_df.columns]
    intervals_df.columns = [str(col).strip() for col in intervals_df.columns]

    return recording_df, intervals_df


def prepare_intervals(intervals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize Manual_Intervals for programmatic labeling.
    """

    required = [
        "raw_recording_folder",
        "start_time_s",
        "end_time_s",
        "label_name",
        "static_type",
        "path_type",
        "incline",
    ]

    missing = [col for col in required if col not in intervals_df.columns]

    if missing:
        raise ValueError(f"Manual_Intervals is missing required columns: {missing}")

    df = intervals_df.copy()

    # Keep only rows that actually define an interval.
    df = df[df["raw_recording_folder"].notna()].copy()
    df = df[df["start_time_s"].notna() & df["end_time_s"].notna()].copy()

    if len(df) == 0:
        raise ValueError(
            "Manual_Intervals contains no valid intervals. Fill the Excel sheet first."
        )

    df["raw_recording_folder"] = df["raw_recording_folder"].astype(str).str.strip()

    # Preserve the original Excel values for debugging before conversion.
    # This is useful when a time value is written as text, for example with an
    # Italian decimal comma such as "12,35".
    df["start_time_s_raw_excel"] = df["start_time_s"]
    df["end_time_s_raw_excel"] = df["end_time_s"]

    df["start_time_s"] = df["start_time_s"].apply(
        lambda value: parse_excel_time_seconds(value, allow_end_marker=False)
    )
    df["end_time_s"] = df["end_time_s"].apply(
        lambda value: parse_excel_time_seconds(value, allow_end_marker=True)
    )

    normalized_rows = []

    for excel_row, row in df.iterrows():
        label_name = normalize_label_name(row.get("label_name"))
        static_type_name = normalize_static_type(row.get("static_type"), label_name=label_name)
        path_type_name = normalize_path_type(row.get("path_type"), label_name=label_name)
        incline_name = normalize_incline(row.get("incline"), label_name=label_name)
        activity_detail_name = infer_activity_detail(label_name, incline_name)

        normalized_rows.append({
            **row.to_dict(),
            "excel_row": int(excel_row) + 2,
            "label_name_canonical": label_name,
            "label_code": GSD_LABEL_MAP[label_name],
            "static_type_detail": normalize_text(row.get("static_type"), default="none"),
            "static_type_canonical": static_type_name,
            "static_type_code": STATIC_TYPE_MAP[static_type_name],
            "path_type_detail": normalize_text(row.get("path_type"), default="none"),
            "path_type_canonical": path_type_name,
            "path_type_code": PATH_TYPE_NAME_TO_CODE[path_type_name],
            "incline_canonical": incline_name,
            "incline_code": INCLINE_MAP[incline_name],
            "activity_detail_name": activity_detail_name,
            "activity_detail_code": ACTIVITY_DETAIL_MAP[activity_detail_name],
            "transition_type_canonical": normalize_transition_type(row.get("transition_type", "None")),
            "gait_phase_canonical": normalize_gait_phase(row.get("gait_stationarity", "none")),
        })

    normalized_df = pd.DataFrame(normalized_rows)

    return normalized_df


def validate_intervals(intervals_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Validate temporal and semantic consistency of manual intervals.

    Gaps are reported as warnings, not errors, because the user may intentionally
    exclude sit-to-stand, stand-to-sit or unclear transition portions.
    """

    rows = []

    for _, row in intervals_df.iterrows():
        start_is_finite = np.isfinite(row["start_time_s"])
        # end_time_s may be +inf when the Excel cell contains the explicit
        # marker "end". This is valid and will be resolved to the actual
        # recording end during sample labeling.
        end_is_valid = np.isfinite(row["end_time_s"]) or np.isposinf(row["end_time_s"])

        if not start_is_finite or not end_is_valid:
            rows.append({
                "severity": "ERROR",
                "raw_recording_folder": row.get("raw_recording_folder"),
                "excel_row": row.get("excel_row"),
                "issue": "start_time_s or end_time_s is not numeric",
                "raw_start_time_s": row.get("start_time_s_raw_excel"),
                "raw_end_time_s": row.get("end_time_s_raw_excel"),
            })

        elif row["end_time_s"] <= row["start_time_s"]:
            rows.append({
                "severity": "ERROR",
                "raw_recording_folder": row.get("raw_recording_folder"),
                "excel_row": row.get("excel_row"),
                "issue": "end_time_s must be greater than start_time_s",
            })

        if row["label_name_canonical"] == "walking" and row["static_type_canonical"] != "none":
            rows.append({
                "severity": "WARNING",
                "raw_recording_folder": row.get("raw_recording_folder"),
                "excel_row": row.get("excel_row"),
                "issue": "walking interval has non-none static_type; code will force static_type=none",
            })

        if row["label_name_canonical"] == "static" and row["path_type_canonical"] != "none":
            rows.append({
                "severity": "WARNING",
                "raw_recording_folder": row.get("raw_recording_folder"),
                "excel_row": row.get("excel_row"),
                "issue": "static interval has non-none path_type; code will force path_type=none",
            })

    for folder, folder_df in intervals_df.groupby("raw_recording_folder"):
        folder_df = folder_df.sort_values(["start_time_s", "end_time_s"])
        prev_end = None
        prev_row = None

        for _, row in folder_df.iterrows():
            if prev_end is not None:
                gap = float(row["start_time_s"] - prev_end)

                if gap < -1e-9:
                    rows.append({
                        "severity": "ERROR",
                        "raw_recording_folder": folder,
                        "excel_row": row.get("excel_row"),
                        "previous_excel_row": prev_row,
                        "issue": f"interval overlaps previous interval by {-gap:.3f} s",
                    })

                elif gap > 1.0:
                    rows.append({
                        "severity": "WARNING",
                        "raw_recording_folder": folder,
                        "excel_row": row.get("excel_row"),
                        "previous_excel_row": prev_row,
                        "issue": f"gap from previous interval is {gap:.3f} s; acceptable only if intentionally excluded",
                    })

            prev_end = float(row["end_time_s"])
            prev_row = row.get("excel_row")

    report_df = pd.DataFrame(rows)
    report_path = output_dir / "manual_interval_validation_report.csv"
    report_df.to_csv(report_path, index=False)

    if len(report_df) > 0:
        print(f"[WARNING] Manual interval validation report saved: {report_path}")
        print(report_df["severity"].value_counts(dropna=False).to_string())
    else:
        print(f"[OK] No manual interval validation issues found. Empty report saved: {report_path}")

    return report_df


def prepare_recording_metadata(recording_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Recording_Review and keep one row per raw recording folder.
    """

    if "raw_recording_folder" not in recording_df.columns:
        raise ValueError("Recording_Review is missing raw_recording_folder column.")

    df = recording_df.copy()
    df = df[df["raw_recording_folder"].notna()].copy()
    df["raw_recording_folder"] = df["raw_recording_folder"].astype(str).str.strip()

    # Keep the last occurrence if accidental duplicates exist.
    df = df.drop_duplicates(subset=["raw_recording_folder"], keep="last")

    return df.set_index("raw_recording_folder")


# =============================================================================
# RAW IMU LOADING
# =============================================================================

def find_imu_file(raw_folder: Path) -> Path:
    """
    Find the IMU file inside a raw Tobii recording folder.

    The function supports both compressed and manually extracted files.
    Priority is given to extracted files only if they are clearly named imudata.
    """

    candidates = [
        raw_folder / "imudata",
        raw_folder / "imudata.jsonl",
        raw_folder / "imudata.txt",
        raw_folder / "imudata.gz",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    gz_candidates = sorted(raw_folder.glob("*imu*.gz"))
    if gz_candidates:
        return gz_candidates[0]

    extracted_candidates = [
        path for path in sorted(raw_folder.glob("*imu*"))
        if path.is_file() and path.suffix.lower() != ".gz"
    ]

    if extracted_candidates:
        return extracted_candidates[0]

    raise FileNotFoundError(f"No IMU file found inside: {raw_folder}")


def open_maybe_gzip(path: Path):
    """
    Open either a .gz file or a plain extracted JSON-lines file.
    """

    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")

    return open(path, "rt", encoding="utf-8")


def read_tobii_raw_imu(imu_path: Path) -> pd.DataFrame:
    """
    Read Tobii raw IMU JSON-lines into a DataFrame.

    Expected line example:
    {"type":"imu","timestamp":0.004049,
     "data":{"accelerometer":[...],"gyroscope":[...]}}
    """

    rows = []

    with open_maybe_gzip(imu_path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "imu":
                continue

            data = obj.get("data", {})
            acc = data.get("accelerometer", [np.nan, np.nan, np.nan])
            gyr = data.get("gyroscope", [np.nan, np.nan, np.nan])

            if len(acc) != 3 or len(gyr) != 3:
                continue

            rows.append({
                "time_s_raw": float(obj.get("timestamp", np.nan)),
                "acc_x": acc[0],
                "acc_y": acc[1],
                "acc_z": acc[2],
                "gyro_x": gyr[0],
                "gyro_y": gyr[1],
                "gyro_z": gyr[2],
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError(f"No IMU rows read from: {imu_path}")

    df = df.sort_values("time_s_raw").drop_duplicates(subset=["time_s_raw"], keep="first")
    df = df.reset_index(drop=True)

    # Shift to zero to align with video annotation time.
    df["time_s"] = df["time_s_raw"] - df["time_s_raw"].iloc[0]

    return df


def estimate_fs_from_time(time_s: np.ndarray) -> float:
    """
    Estimate sampling frequency from timestamps.
    """

    if len(time_s) < 3:
        return np.nan

    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        return np.nan

    return float(1.0 / np.median(dt))


def resample_imu_to_uniform_grid(imu_df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
    """
    Interpolate raw IMU samples onto a fixed-frequency time grid.

    This step is useful because Tobii raw IMU is close to 119 Hz, while the
    existing GSD pipeline and window definitions are usually based on 100 Hz.
    """

    time_raw = imu_df["time_s"].to_numpy(dtype=float)
    t_start = 0.0
    t_end = float(np.nanmax(time_raw))

    if not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("Invalid IMU time vector; cannot resample.")

    time_grid = np.arange(t_start, t_end, 1.0 / target_fs)

    out = pd.DataFrame({"time_s": time_grid})

    signal_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    for col in signal_cols:
        values = pd.to_numeric(imu_df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(time_raw) & np.isfinite(values)

        if np.sum(valid) < 2:
            out[col] = np.nan
        else:
            out[col] = np.interp(time_grid, time_raw[valid], values[valid])

    return out


# =============================================================================
# GRAVITY ALIGNMENT
# =============================================================================

def find_first_valid_static_reference_window(
    sample_df: pd.DataFrame,
    fs: float,
    static_duration_s: float = 1.0,
):
    """
    Find the first continuous static interval usable as gravity reference.

    The reference must satisfy two conditions:
    1. sample-wise GSD label equal to static;
    2. finite accelerometer and gyroscope values for at least static_duration_s.

    This implements the Tobii-specific version of the same idea used in the
    other construction scripts: gravity alignment is estimated from a static
    portion of the current recording, but here the static portion is selected
    from the manual Excel labels rather than assumed to be at the very beginning.
    """

    n_static = int(round(static_duration_s * fs))

    if n_static < 1:
        raise ValueError("Static duration too short for the selected sampling frequency.")

    signal_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    labels = sample_df["label"].to_numpy(dtype=np.int32)
    signal = sample_df[signal_cols].to_numpy(dtype=float)

    finite_signal_mask = np.isfinite(signal).all(axis=1)
    valid_static_mask = (labels == GSD_LABEL_MAP["static"]) & finite_signal_mask

    static_indices = np.where(valid_static_mask)[0]

    if len(static_indices) < n_static:
        raise ValueError(
            "No valid static portion long enough for gravity alignment. "
            f"Required at least {n_static} consecutive samples."
        )

    split_points = np.where(np.diff(static_indices) > 1)[0] + 1
    static_blocks = np.split(static_indices, split_points)

    for block in static_blocks:
        if len(block) >= n_static:
            reference_start = int(block[0])
            reference_end = int(block[0] + n_static)
            return reference_start, reference_end

    raise ValueError(
        "Static samples exist, but no continuous static block is long enough "
        "for gravity alignment."
    )


def align_sample_df_to_gravity_from_first_static_label(
    sample_df: pd.DataFrame,
    fs: float,
    static_duration_s: float = 1.0,
    recording_id: str = "unknown",
) -> pd.DataFrame:
    """
    Align Tobii IMU signals to gravity using the first valid static label.

    The imported align_imu_to_gravity() function estimates the rotation from the
    first static_duration_s seconds of the input signal. To use the first labelled
    static portion as reference without cutting the recording, the function is
    called on the signal starting at the selected static block. The estimated
    rotation matrix is then applied to the full recording.

    No additional metadata columns are added. Only the six IMU signal columns are
    replaced with their gravity-aligned version.
    """

    signal_cols_acc = ["acc_x", "acc_y", "acc_z"]
    signal_cols_gyr = ["gyro_x", "gyro_y", "gyro_z"]

    reference_start, reference_end = find_first_valid_static_reference_window(
        sample_df=sample_df,
        fs=fs,
        static_duration_s=static_duration_s,
    )

    acc = sample_df[signal_cols_acc].to_numpy(dtype=float)
    gyr = sample_df[signal_cols_gyr].to_numpy(dtype=float)

    # Estimate the rotation using the first valid static-labelled block.
    # The returned aligned cropped signals are not used; only R is needed so that
    # the same rotation can be applied to the complete recording.
    _, _, R = align_imu_to_gravity(
        acc=acc[reference_start:],
        gyr=gyr[reference_start:],
        sampling_rate_hz=fs,
        static_duration_s=static_duration_s,
        gravity_ideal=np.array([1.0, 0.0, 0.0]),
    )

    acc_aligned = (R @ acc.T).T
    gyr_aligned = (R @ gyr.T).T

    mean_static_acc = np.mean(acc_aligned[reference_start:reference_end], axis=0)
    mean_static_acc_norm = mean_static_acc / np.linalg.norm(mean_static_acc)

    print(f"[ALIGNMENT CHECK] Recording {recording_id}")
    print("Alignment reference: first_valid_static_label_1s")
    print(
        "Reference interval: "
        f"{sample_df['time_s'].iloc[reference_start]:.3f}-"
        f"{sample_df['time_s'].iloc[reference_end - 1]:.3f} s"
    )
    print("Normalized mean aligned static acceleration:", mean_static_acc_norm)

    aligned_df = sample_df.copy()
    aligned_df.loc[:, signal_cols_acc] = acc_aligned
    aligned_df.loc[:, signal_cols_gyr] = gyr_aligned

    return aligned_df


# =============================================================================
# SAMPLE LABELING
# =============================================================================

def derive_gait_phase(labels: np.ndarray, fs: float, transient_duration_s: float) -> np.ndarray:
    """
    Derive gait initiation / steady-state / termination from walking bouts.
    """

    gait_phase = np.full(len(labels), GAIT_PHASE_MAP["none"], dtype=np.int32)
    walking_indices = np.where(labels == GSD_LABEL_MAP["walking"])[0]

    if len(walking_indices) == 0:
        return gait_phase

    split_points = np.where(np.diff(walking_indices) > 1)[0] + 1
    walking_blocks = np.split(walking_indices, split_points)
    transient_samples = int(round(transient_duration_s * fs))

    for block in walking_blocks:
        if len(block) == 0:
            continue

        if len(block) <= 2 * transient_samples:
            midpoint = len(block) // 2
            gait_phase[block[:midpoint]] = GAIT_PHASE_MAP["gait_initiation"]
            gait_phase[block[midpoint:]] = GAIT_PHASE_MAP["gait_termination"]
        else:
            gait_phase[block[:transient_samples]] = GAIT_PHASE_MAP["gait_initiation"]
            gait_phase[block[transient_samples:-transient_samples]] = GAIT_PHASE_MAP["steady_state"]
            gait_phase[block[-transient_samples:]] = GAIT_PHASE_MAP["gait_termination"]

    return gait_phase


def build_sample_walking_bouts(labels: np.ndarray) -> np.ndarray:
    """
    Build local walking-bout identifiers from contiguous walking labels.
    """

    walking_bouts = np.full(len(labels), -1, dtype=np.int32)
    walking_indices = np.where(labels == GSD_LABEL_MAP["walking"])[0]

    if len(walking_indices) == 0:
        return walking_bouts

    split_points = np.where(np.diff(walking_indices) > 1)[0] + 1
    walking_blocks = np.split(walking_indices, split_points)

    for bout_id, block in enumerate(walking_blocks):
        walking_bouts[block] = bout_id

    return walking_bouts


def apply_manual_intervals_to_imu(
    imu_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    recording_metadata: dict,
    fs: float,
) -> pd.DataFrame:
    """
    Apply Excel temporal intervals to the IMU samples of one recording.
    """

    n = len(imu_df)
    sample_df = imu_df.copy()

    sample_df["label"] = GSD_LABEL_MAP["none"]
    sample_df["label_name"] = "none"
    sample_df["activity_detail"] = ACTIVITY_DETAIL_MAP["none"]
    sample_df["activity_detail_name"] = "none"
    sample_df["static_type"] = STATIC_TYPE_MAP["none"]
    sample_df["static_type_name"] = "none"
    sample_df["path_type"] = None
    sample_df["path_type_name"] = "none"
    sample_df["incline"] = INCLINE_MAP["none"]
    sample_df["incline_name"] = "none"
    sample_df["transition_type"] = "None"
    sample_df["gait_phase"] = GAIT_PHASE_MAP["none"]
    sample_df["gait_phase_name"] = "none"
    sample_df["excel_static_type_detail"] = "none"
    sample_df["excel_path_type_detail"] = "none"
    sample_df["excel_row"] = -1

    for _, interval in intervals_df.sort_values("start_time_s").iterrows():
        start = float(interval["start_time_s"])
        end = float(interval["end_time_s"])

        # The Excel annotation may use "end" in end_time_s. In prepare_intervals
        # this is represented as +inf, then resolved here using the actual
        # resampled IMU timeline of the current recording. Adding one sample
        # period keeps the last IMU sample inside the half-open interval.
        if np.isposinf(end):
            end = float(sample_df["time_s"].max()) + (1.0 / fs)

        # Half-open interval prevents double assignment at exact boundaries.
        mask = (sample_df["time_s"] >= start) & (sample_df["time_s"] < end)

        label_code = int(interval["label_code"])
        label_name = interval["label_name_canonical"]
        static_code = int(interval["static_type_code"])
        static_name = interval["static_type_canonical"]
        path_code = interval["path_type_code"]
        path_name = interval["path_type_canonical"]
        incline_code = int(interval["incline_code"])
        incline_name = interval["incline_canonical"]
        activity_code = int(interval["activity_detail_code"])
        activity_name = interval["activity_detail_name"]
        gait_phase_name = interval["gait_phase_canonical"]
        gait_phase_code = GAIT_PHASE_MAP[gait_phase_name]

        sample_df.loc[mask, "label"] = label_code
        sample_df.loc[mask, "label_name"] = label_name
        sample_df.loc[mask, "activity_detail"] = activity_code
        sample_df.loc[mask, "activity_detail_name"] = activity_name
        sample_df.loc[mask, "static_type"] = static_code
        sample_df.loc[mask, "static_type_name"] = static_name
        sample_df.loc[mask, "path_type"] = path_code
        sample_df.loc[mask, "path_type_name"] = path_name
        sample_df.loc[mask, "incline"] = incline_code
        sample_df.loc[mask, "incline_name"] = incline_name
        sample_df.loc[mask, "transition_type"] = interval["transition_type_canonical"]
        sample_df.loc[mask, "gait_phase"] = gait_phase_code
        sample_df.loc[mask, "gait_phase_name"] = gait_phase_name
        sample_df.loc[mask, "excel_static_type_detail"] = interval["static_type_detail"]
        sample_df.loc[mask, "excel_path_type_detail"] = interval["path_type_detail"]
        sample_df.loc[mask, "excel_row"] = int(interval["excel_row"])

    # Fill gait phase automatically where Excel did not provide it.
    derived_phase = derive_gait_phase(
        labels=sample_df["label"].to_numpy(dtype=np.int32),
        fs=fs,
        transient_duration_s=GAIT_TRANSIENT_DURATION_S,
    )

    missing_phase_mask = sample_df["gait_phase"].to_numpy(dtype=np.int32) == GAIT_PHASE_MAP["none"]
    walking_mask = sample_df["label"].to_numpy(dtype=np.int32) == GSD_LABEL_MAP["walking"]
    fill_mask = missing_phase_mask & walking_mask

    sample_df.loc[fill_mask, "gait_phase"] = derived_phase[fill_mask]
    sample_df.loc[fill_mask, "gait_phase_name"] = [
        get_map_name(GAIT_PHASE_MAP, value) for value in derived_phase[fill_mask]
    ]

    walking_bouts = build_sample_walking_bouts(sample_df["label"].to_numpy(dtype=np.int32))
    sample_df["walking_bout"] = walking_bouts
    sample_df["SampleWalkingBouts"] = walking_bouts

    # Recording-level metadata copied to every sample.
    sample_df["dataset"] = "Tobii_VIXIONE"
    sample_df["raw_recording_folder"] = recording_metadata.get("raw_recording_folder", "unknown")
    sample_df["subject_id"] = recording_metadata.get("subject_id", "tobii_subject")
    # Store transition_type as a clean string instead of NaN/empty values.
    sample_df["transition_type"] = (
        sample_df["transition_type"]
        .fillna("None")
        .astype(str)
        .replace({"nan": "None", "": "None"})
    )

    sample_df["task"] = recording_metadata.get("task_group", "unknown")
    sample_df["test_type"] = recording_metadata.get("task_group", "unknown")
    sample_df["task_group"] = recording_metadata.get("task_group", "unknown")
    sample_df["task_name_manual"] = recording_metadata.get("task_name_manual", "unknown")
    sample_df["scenevideo_relative_path"] = recording_metadata.get("scenevideo_relative_path", "")
    sample_df["rename_alias"] = recording_metadata.get("rename_alias_do_not_rename_folder", "")
    sample_df["fs"] = fs

    return sample_df


# =============================================================================
# WINDOWING
# =============================================================================

def assign_transition_type_boundary_aware(y_win: np.ndarray, min_class_fraction: float = 0.10) -> str:
    """
    Assign window-level transition type using the first and last valid labels.

    This is less conservative than first-half / second-half majority voting.
    It is designed for manually labeled Tobii data, where a transition window is
    expected to contain both static and walking samples and where the temporal
    order of the first and last valid labels is informative.

    A transition is assigned only if both static and walking cover at least
    min_class_fraction of the full window. This avoids marking windows as
    transitions because of a single isolated sample.
    """

    valid = y_win[y_win != GSD_LABEL_MAP["none"]]

    if len(valid) < 2:
        return "None"

    n_static = int(np.sum(y_win == GSD_LABEL_MAP["static"]))
    n_walking = int(np.sum(y_win == GSD_LABEL_MAP["walking"]))

    static_fraction = n_static / len(y_win)
    walking_fraction = n_walking / len(y_win)

    if static_fraction < min_class_fraction or walking_fraction < min_class_fraction:
        return "None"

    start_label = int(valid[0])
    end_label = int(valid[-1])

    if start_label == GSD_LABEL_MAP["static"] and end_label == GSD_LABEL_MAP["walking"]:
        return "Static2Walking"

    if start_label == GSD_LABEL_MAP["walking"] and end_label == GSD_LABEL_MAP["static"]:
        return "Walking2Static"

    return "None"


def summarize_window_path_type(path_win: np.ndarray, window_size: int) -> dict:
    """
    Summarize straight/curved/none path-type composition inside a window.
    """

    path_values = list(path_win)
    n_straight = int(sum(value == 0 for value in path_values))
    n_curved = int(sum(value == 1 for value in path_values))
    n_none = int(sum(value is None or str(value) == "nan" for value in path_values))
    n_other = int(window_size - n_straight - n_curved - n_none)

    straight_fraction = n_straight / window_size
    curved_fraction = n_curved / window_size
    none_fraction = n_none / window_size

    if straight_fraction >= PATH_TYPE_MIN_FRACTION:
        final_path_type = 0
        is_path_type_robust = True
    elif curved_fraction >= PATH_TYPE_MIN_FRACTION:
        final_path_type = 1
        is_path_type_robust = True
    else:
        final_path_type = None
        is_path_type_robust = False

    return {
        "final_path_type": final_path_type,
        "n_straight_path_samples": n_straight,
        "n_curved_path_samples": n_curved,
        "n_none_path_samples": n_none,
        "n_other_path_samples": n_other,
        "straight_path_fraction": straight_fraction,
        "curved_path_fraction": curved_fraction,
        "none_path_fraction": none_fraction,
        "path_type_confidence": max(straight_fraction, curved_fraction),
        "is_path_type_robust": is_path_type_robust,
    }


def summarize_window_static_context(static_type_win: np.ndarray, window_size: int) -> dict:
    """
    Preserve internal static-subtype composition inside one window.
    """

    counts = {
        "none": int(np.sum(static_type_win == STATIC_TYPE_MAP["none"])),
        "static": int(np.sum(static_type_win == STATIC_TYPE_MAP["static"])),
        "standing": int(np.sum(static_type_win == STATIC_TYPE_MAP["standing"])),
        "sitting": int(np.sum(static_type_win == STATIC_TYPE_MAP["sitting"])),
        "both": int(np.sum(static_type_win == STATIC_TYPE_MAP["both"])),
    }

    fractions = {key: value / window_size for key, value in counts.items()}
    candidate_contexts = {
        STATIC_TYPE_MAP["static"]: fractions["static"],
        STATIC_TYPE_MAP["standing"]: fractions["standing"],
        STATIC_TYPE_MAP["sitting"]: fractions["sitting"],
        STATIC_TYPE_MAP["both"]: fractions["both"],
    }

    static_context_type = max(candidate_contexts, key=candidate_contexts.get)
    static_context_fraction = candidate_contexts[static_context_type]

    if static_context_fraction < STATIC_CONTEXT_MIN_FRACTION:
        static_context_type = STATIC_TYPE_MAP["none"]
        static_context_fraction = 0.0
        has_static_context = False
    else:
        has_static_context = True

    return {
        "n_none_static_type_samples": counts["none"],
        "n_generic_static_type_samples": counts["static"],
        "n_standing_static_type_samples": counts["standing"],
        "n_sitting_static_type_samples": counts["sitting"],
        "n_both_static_type_samples": counts["both"],
        "none_static_type_fraction": fractions["none"],
        "generic_static_type_fraction": fractions["static"],
        "standing_static_type_fraction": fractions["standing"],
        "sitting_static_type_fraction": fractions["sitting"],
        "both_static_type_fraction": fractions["both"],
        "static_context_min_fraction": STATIC_CONTEXT_MIN_FRACTION,
        "static_context_type": static_context_type,
        "static_context_type_name": get_map_name(STATIC_TYPE_MAP, static_context_type),
        "static_context_fraction": static_context_fraction,
        "has_static_context": has_static_context,
        "has_standing_context": fractions["standing"] >= STATIC_CONTEXT_MIN_FRACTION,
        "has_sitting_context": fractions["sitting"] >= STATIC_CONTEXT_MIN_FRACTION,
        "has_both_context": fractions["both"] >= STATIC_CONTEXT_MIN_FRACTION,
    }


def create_window_dataset_from_trials(trial_dataset: list, window_size: int, step_size: int, output_path: Path):
    """
    Create a fixed-window dataset from trial-level Tobii data.
    """

    X_windows = []
    Y_windows = []
    metadata_windows = []

    for trial in trial_dataset:
        X = trial["X"]
        sample_metadata = trial["sample_metadata"]
        labels = trial["labels"]
        activity_detail = trial["activity_detail"]
        static_type = trial["static_type"]
        path_types = trial["path_types"]
        gait_phase = trial["gait_phase"]
        walking_bouts = trial["walking_bouts"]

        for i in range(0, len(X) - window_size + 1, step_size):
            x_win = X[i:i + window_size]
            y_win = labels[i:i + window_size]

            if np.isnan(x_win).any():
                continue

            valid_labels = y_win[y_win != GSD_LABEL_MAP["none"]]

            if len(valid_labels) == 0:
                continue

            unique_vals, counts = np.unique(valid_labels, return_counts=True)
            final_label = int(unique_vals[np.argmax(counts)])

            n_static_samples = int(np.sum(y_win == GSD_LABEL_MAP["static"]))
            n_walking_samples = int(np.sum(y_win == GSD_LABEL_MAP["walking"]))
            n_none_label_samples = int(np.sum(y_win == GSD_LABEL_MAP["none"]))

            static_fraction = n_static_samples / window_size
            walking_fraction = n_walking_samples / window_size
            none_label_fraction = n_none_label_samples / window_size
            valid_label_fraction = (n_static_samples + n_walking_samples) / window_size

            if final_label == GSD_LABEL_MAP["static"]:
                final_label_fraction = static_fraction
            elif final_label == GSD_LABEL_MAP["walking"]:
                final_label_fraction = walking_fraction
            else:
                final_label_fraction = 0.0

            is_robust_window = final_label_fraction >= WINDOW_LABEL_MIN_FRACTION
            walking_static_ratio = (n_walking_samples + 1e-6) / (n_static_samples + 1e-6)

            transition_type = assign_transition_type_boundary_aware(
                y_win,
                min_class_fraction=TRANSITION_MIN_CLASS_FRACTION
            )
            path_summary = summarize_window_path_type(path_types[i:i + window_size], window_size)
            static_context_summary = summarize_window_static_context(static_type[i:i + window_size], window_size)

            activity_detail_win = activity_detail[i:i + window_size]
            static_type_win = static_type[i:i + window_size]
            gait_phase_win = gait_phase[i:i + window_size]
            walking_bouts_win = walking_bouts[i:i + window_size]

            if final_label == GSD_LABEL_MAP["walking"]:
                final_static_type = STATIC_TYPE_MAP["none"]
                final_path_type = path_summary["final_path_type"]

                valid_activity = activity_detail_win[
                    (activity_detail_win == ACTIVITY_DETAIL_MAP["walking"])
                    | (activity_detail_win == ACTIVITY_DETAIL_MAP["ascending"])
                    | (activity_detail_win == ACTIVITY_DETAIL_MAP["descending"])
                ]

                if len(valid_activity) > 0:
                    vals, cnts = np.unique(valid_activity, return_counts=True)
                    final_activity_detail = int(vals[np.argmax(cnts)])
                else:
                    final_activity_detail = ACTIVITY_DETAIL_MAP["walking"]

                valid_phase = gait_phase_win[gait_phase_win != GAIT_PHASE_MAP["none"]]
                if len(valid_phase) > 0:
                    vals, cnts = np.unique(valid_phase, return_counts=True)
                    final_gait_phase = int(vals[np.argmax(cnts)])
                else:
                    final_gait_phase = GAIT_PHASE_MAP["none"]

                valid_bouts = walking_bouts_win[walking_bouts_win >= 0]
                if len(valid_bouts) > 0:
                    vals, cnts = np.unique(valid_bouts, return_counts=True)
                    final_walking_bout = int(vals[np.argmax(cnts)])
                else:
                    final_walking_bout = -1

            else:
                final_path_type = None
                final_activity_detail = ACTIVITY_DETAIL_MAP["static"]
                final_gait_phase = GAIT_PHASE_MAP["none"]
                final_walking_bout = -1

                valid_static = static_type_win[static_type_win != STATIC_TYPE_MAP["none"]]
                if len(valid_static) > 0:
                    vals, cnts = np.unique(valid_static, return_counts=True)
                    final_static_type = int(vals[np.argmax(cnts)])
                else:
                    final_static_type = STATIC_TYPE_MAP["static"]

            first_sample_row = sample_metadata.iloc[i]

            X_windows.append(x_win)
            Y_windows.append(final_label)

            metadata_windows.append({
                "dataset": trial["dataset"],
                "subject_id": trial["subject_id"],
                "task": trial["task"],
                "test_type": trial["test_type"],
                "task_group": trial["task_group"],
                "task_name_manual": trial["task_name_manual"],
                "raw_recording_folder": trial["raw_recording_folder"],
                "scenevideo_relative_path": trial["scenevideo_relative_path"],
                "relative_file_path": trial["relative_file_path"],
                "file_name": trial["file_name"],
                "window_start_sample": i,
                "window_end_sample": i + window_size,
                "window_start_time_s": float(first_sample_row["time_s"]),
                "window_duration_s": window_size / trial["fs"],
                "n_static_samples": n_static_samples,
                "n_walking_samples": n_walking_samples,
                "n_none_label_samples": n_none_label_samples,
                "static_fraction": float(static_fraction),
                "walking_fraction": float(walking_fraction),
                "none_label_fraction": float(none_label_fraction),
                "valid_label_fraction": float(valid_label_fraction),
                "final_label_fraction": float(final_label_fraction),
                "label_min_fraction": float(WINDOW_LABEL_MIN_FRACTION),
                "walking_static_ratio": float(walking_static_ratio),
                "contains_static_walking_mix": bool(n_static_samples > 0 and n_walking_samples > 0),
                "is_transition": bool(transition_type != "None"),
                "transition_definition": TRANSITION_DEFINITION,
                "is_robust_window": bool(is_robust_window),
                "label": int(final_label),
                "label_name": get_map_name(GSD_LABEL_MAP, final_label),
                "activity_detail": int(final_activity_detail),
                "activity_detail_name": get_map_name(ACTIVITY_DETAIL_MAP, final_activity_detail),
                "path_type": final_path_type,
                "path_type_name": PATH_TYPE_MAP.get(final_path_type, "none"),
                "path_type_min_fraction": float(PATH_TYPE_MIN_FRACTION),
                **path_summary,
                "static_type": int(final_static_type),
                "static_type_name": get_map_name(STATIC_TYPE_MAP, final_static_type),
                **static_context_summary,
                "transition_type": transition_type,
                "walking_bout": int(final_walking_bout),
                "gait_phase": int(final_gait_phase),
                "gait_phase_name": get_map_name(GAIT_PHASE_MAP, final_gait_phase),
                "group": trial["group"],
            })

    X_windows = np.array(X_windows, dtype=np.float32)
    Y_windows = np.array(Y_windows, dtype=np.int32)
    metadata_windows = np.array(metadata_windows, dtype=object)

    window_bundle = {
        "DatasetName": "Tobii_VIXIONE",
        "WindowSize": window_size,
        "StepSize": step_size,
        "SamplingFrequencyHz": TARGET_FS,
        "label_map": GSD_LABEL_MAP,
        "gsd_label_map": GSD_LABEL_MAP,
        "activity_detail_map": ACTIVITY_DETAIL_MAP,
        "static_type_map": STATIC_TYPE_MAP,
        "path_type_map": PATH_TYPE_MAP,
        "gait_phase_map": GAIT_PHASE_MAP,
        "TransitionDefinition": TRANSITION_DEFINITION,
        "X": X_windows,
        "Y": Y_windows,
        "metadata": metadata_windows,
    }

    with open(output_path, "wb") as f:
        pickle.dump(window_bundle, f)

    metadata_df = pd.DataFrame(list(metadata_windows))
    metadata_csv_path = output_path.with_suffix(".csv")

    # ------------------------------------------------------------
    # Semantic consistency checks for window-level metadata
    # ------------------------------------------------------------
    # These checks verify that final window-level metadata remain coherent with the
    # final binary GSD label.
    #
    # A walking window must not have a final static_type different from none,
    # because static_type describes the final static condition of the window.
    #
    # A static window must not have a final path_type different from none,
    # because path_type describes locomotor geometry and is meaningful only for
    # final walking windows.
    #
    # Static-context metadata are allowed in both cases because they describe the
    # internal composition of the window, not the final class.
    walking_with_static_type = (
            (metadata_df["label_name"] == "walking")
            &
            (metadata_df["static_type_name"] != "none")
    )

    static_with_path_type = (
            (metadata_df["label_name"] == "static")
            &
            (metadata_df["path_type_name"] != "none")
    )

    if walking_with_static_type.any() or static_with_path_type.any():
        debug_inconsistency_path = (
            metadata_csv_path.with_name(
                metadata_csv_path.stem + "_SEMANTIC_INCONSISTENCIES.csv"
            )
        )

        metadata_df[
            walking_with_static_type | static_with_path_type
            ].to_csv(
            debug_inconsistency_path,
            index=False
        )

        raise RuntimeError(
            "Semantic inconsistency detected in Tobii window-level metadata. "
            "A debug CSV was saved to: "
            f"{debug_inconsistency_path}"
        )

    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"[SUCCESS] Window dataset saved: {output_path}")
    print(f"[SUCCESS] Window metadata saved: {metadata_csv_path}")
    print(f"          X shape: {X_windows.shape} | Y shape: {Y_windows.shape}")


# =============================================================================
# MAIN CONSTRUCTION
# =============================================================================

def build_trial_dataset(dataset_root: Path, excel_path: Path, output_dir: Path, target_fs: float):
    """
    Build labeled trial-level data from raw IMU files and Excel annotations.
    """

    recording_df, intervals_raw_df = load_excel_annotations(excel_path)
    recording_meta_df = prepare_recording_metadata(recording_df)
    intervals_df = prepare_intervals(intervals_raw_df)

    validation_report = validate_intervals(intervals_df, output_dir)

    if len(validation_report) > 0 and (validation_report["severity"] == "ERROR").any():
        raise RuntimeError(
            "Manual_Intervals contains ERROR-level issues. Fix Excel before construction."
        )

    raw_root = dataset_root / RAW_RECORDINGS_FOLDER_NAME

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw recordings folder not found: {raw_root}")

    trial_dataset = []
    all_sample_metadata = []

    folders_to_process = sorted(intervals_df["raw_recording_folder"].unique().tolist())

    print(f"[INFO] Recordings with manual intervals: {len(folders_to_process)}")

    for idx, folder_name in enumerate(folders_to_process, start=1):
        print(f"\n[{idx}/{len(folders_to_process)}] Processing {folder_name}")

        raw_folder = raw_root / folder_name

        if not raw_folder.exists():
            print(f"[WARNING] Raw folder not found, skipped: {raw_folder}")
            continue

        folder_intervals = intervals_df[intervals_df["raw_recording_folder"] == folder_name].copy()

        if folder_name in recording_meta_df.index:
            meta = recording_meta_df.loc[folder_name].to_dict()
        else:
            meta = {}

        meta["raw_recording_folder"] = folder_name
        meta["subject_id"] = "tobii_subject"
        # Keep task identifiers clean in the constructed dataset.
        # Excel may contain readable notes such as "(yes)" or "(no)", but these
        # should not become category values in metadata files.
        meta["task_group"] = normalize_basic_category(meta.get("task_group"), default="unknown")
        meta["task_name_manual"] = normalize_basic_category(meta.get("task_name_manual"), default=meta["task_group"])
        meta["scenevideo_relative_path"] = normalize_text(meta.get("scenevideo_relative_path"), default="")

        try:
            imu_path = find_imu_file(raw_folder)
            imu_raw_df = read_tobii_raw_imu(imu_path)
            fs_raw = estimate_fs_from_time(imu_raw_df["time_s"].to_numpy(dtype=float))

            imu_df = resample_imu_to_uniform_grid(imu_raw_df, target_fs=target_fs)

        except Exception as exc:
            print(f"[WARNING] Skipped {folder_name}. Reason: {exc}")
            continue

        sample_df = apply_manual_intervals_to_imu(
            imu_df=imu_df,
            intervals_df=folder_intervals,
            recording_metadata=meta,
            fs=target_fs,
        )

        try:
            sample_df = align_sample_df_to_gravity_from_first_static_label(
                sample_df=sample_df,
                fs=target_fs,
                static_duration_s=1.0,
                recording_id=folder_name,
            )

        except Exception as exc:
            print(f"[WARNING] Skipped {folder_name}. Reason: gravity alignment failed. {exc}")
            continue

        # Keep only recordings with at least some valid manual labels.
        valid_fraction = float(np.mean(sample_df["label"].to_numpy(dtype=np.int32) != GSD_LABEL_MAP["none"]))

        print(f"[INFO] Raw Fs estimated: {fs_raw:.3f} Hz | Resampled Fs: {target_fs:.1f} Hz")
        print(f"[INFO] Samples: {len(sample_df)} | Valid labeled fraction: {valid_fraction:.3f}")

        signal_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        X = sample_df[signal_cols].to_numpy(dtype=float)

        labels = sample_df["label"].to_numpy(dtype=np.int32)
        activity_detail = sample_df["activity_detail"].to_numpy(dtype=np.int32)
        static_type = sample_df["static_type"].to_numpy(dtype=np.int32)
        gait_phase = sample_df["gait_phase"].to_numpy(dtype=np.int32)
        walking_bouts = sample_df["walking_bout"].to_numpy(dtype=np.int32)
        incline = sample_df["incline"].to_numpy(dtype=np.int32)
        path_types = sample_df["path_type"].to_numpy(dtype=object)

        relative_imu_path = make_project_relative_path(imu_path)

        trial_dataset.append({
            "dataset": "Tobii_VIXIONE",
            "subject_id": meta["subject_id"],
            "task": meta["task_group"],
            "test_type": meta["task_group"],
            "task_group": meta["task_group"],
            "task_name_manual": meta["task_name_manual"],
            "raw_recording_folder": folder_name,
            "scenevideo_relative_path": meta["scenevideo_relative_path"],
            "file_name": imu_path.name,
            "file_path": str(imu_path),
            "relative_file_path": relative_imu_path,
            "fs": target_fs,
            "fs_raw_estimated": fs_raw,
            "X": X,
            "labels": labels,
            "activity_detail": activity_detail,
            "static_type": static_type,
            "gait_phase": gait_phase,
            "walking_bouts": walking_bouts,
            "SampleWalkingBouts": walking_bouts,
            "path_types": path_types,
            "incline": incline,
            "label_map": GSD_LABEL_MAP,
            "gsd_label_map": GSD_LABEL_MAP,
            "activity_detail_map": ACTIVITY_DETAIL_MAP,
            "static_type_map": STATIC_TYPE_MAP,
            "path_type_map": PATH_TYPE_MAP,
            "incline_map": INCLINE_MAP,
            "gait_phase_map": GAIT_PHASE_MAP,
            "sample_metadata": sample_df.drop(columns=signal_cols).copy(),
            "group": "EXTERNAL_TOBII",
        })

        all_sample_metadata.append(sample_df.drop(columns=signal_cols).copy())

    if len(trial_dataset) == 0:
        raise RuntimeError("No trial was constructed. Check Excel intervals and raw folders.")

    trial_output_path = output_dir / "Tobii_trial_dataset.pkl"
    with open(trial_output_path, "wb") as f:
        pickle.dump(trial_dataset, f)

    sample_metadata_df = pd.concat(all_sample_metadata, ignore_index=True)
    sample_metadata_path = output_dir / "Tobii_sample_metadata.csv"
    sample_metadata_df.to_csv(sample_metadata_path, index=False)

    print(f"\n[SUCCESS] Trial dataset saved: {trial_output_path}")
    print(f"[SUCCESS] Sample metadata saved: {sample_metadata_path}")

    return trial_dataset


def main():
    """
    Execute Tobii construction from raw IMU streams and Excel manual labels.
    """

    parser = argparse.ArgumentParser(
        description="Build Tobii IMU sample/window datasets from raw imudata and Excel labels."
    )

    parser.add_argument("--root-path", type=str, default=None, help="Optional Tobii dataset root.")
    parser.add_argument("--excel-path", type=str, default=None, help="Optional Excel annotation workbook path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument("--target-fs", type=float, default=TARGET_FS, help="Target uniform sampling frequency.")
    parser.add_argument("--skip-windowing", action="store_true", help="Save trial/sample data only.")

    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.root_path)
    excel_path = resolve_excel_path(dataset_root, args.excel_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Project root:")
    print(PROJECT_ROOT)
    print("\n[INFO] Dataset root:")
    print(dataset_root)
    print("\n[INFO] Excel annotations:")
    print(excel_path)
    print("\n[INFO] Output directory:")
    print(output_dir)

    trial_dataset = build_trial_dataset(
        dataset_root=dataset_root,
        excel_path=excel_path,
        output_dir=output_dir,
        target_fs=args.target_fs,
    )

    if not args.skip_windowing:
        for config_name, window_size, step_size in WINDOW_CONFIGS:
            output_path = output_dir / f"Tobii_window_dataset_{config_name}.pkl"
            create_window_dataset_from_trials(
                trial_dataset=trial_dataset,
                window_size=window_size,
                step_size=step_size,
                output_path=output_path,
            )

    print("\n[DONE] Tobii construction completed.")


if __name__ == "__main__":
    main()
