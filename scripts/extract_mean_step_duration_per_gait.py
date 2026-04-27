from pathlib import Path
import re
import pandas as pd


# Paths (same structure as your validation script)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EXCEL_PATH = DATA_DIR / "Paolo_step_data.xlsx"
OUT_CSV_PATH = DATA_DIR / "mean_step_duration_per_recording.csv"


def extract_timepoint(first_name: str) -> str | None:
    """
    Extract timepoint (F2, F3, or F4) from First_name.
    Returns None if not found.
    """
    if pd.isna(first_name):
        return None

    match = re.search(r"(F[234])", str(first_name).upper())
    return match.group(1) if match else None


def main():
    print("Loading GaitRite metrics Excel file...")
    df = pd.read_excel(EXCEL_PATH)

    # Required columns
    required_columns = ["First_name", "Group", "WalkTask", "Stp_time"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in Excel file: {missing_columns}")

    # Keep only needed columns
    df = df[required_columns].copy()

    # Keep only rows containing F2, F3, or F4 in First_name
    df["First_name"] = df["First_name"].astype(str).str.strip()
    df = df[df["First_name"].str.contains(r"F[234]", case=False, na=False)].copy()

    # Extract timepoint
    df["Timepoint"] = df["First_name"].apply(extract_timepoint)

    # Drop rows with missing values
    df = df.dropna(subset=["First_name", "Group", "WalkTask", "Timepoint", "Stp_time"])

    # Ensure numeric step time
    df["Stp_time"] = pd.to_numeric(df["Stp_time"], errors="coerce")
    df = df.dropna(subset=["Stp_time"])

    # Compute mean step duration per recording
    result = (
        df.groupby(["First_name", "Group", "WalkTask", "Timepoint"], as_index=False)
        .agg(
            mean_step_duration=("Stp_time", "mean"),
            n_steps=("Stp_time", "count"),
        )
        .sort_values(by=["Group", "First_name", "WalkTask", "Timepoint"])
    )

    # Save CSV
    result.to_csv(OUT_CSV_PATH, index=False)

    print(f"CSV successfully saved to: {OUT_CSV_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()