"""
Example usage of load_icicle_trial_with_gait_events
"""

from pathlib import Path
import numpy as np

from load_icicle_trial_with_gait_events import load_icicle_trial_with_gait_events

PROJECT_ROOT = Path(__file__).resolve().parents[3]

h5_file = PROJECT_ROOT / "data" / "ICICLE-Gait_F2_SC" / "Controls" / \
          "20140616-104931-INGC116F2_SC.h5"
h5_file = PROJECT_ROOT / "data" / "ICICLE-Gait_F2_SC" / "PDs" / \
          "20140430-142918-INGP065F2_SC.h5"
gaitrite_excel = PROJECT_ROOT / "metadata" / "Paolo_step_data.xlsx"

# Load WT accelerometer (z-axis only), all Gait_Id
time_sec, data, events = load_icicle_trial_with_gait_events(
    h5_file=h5_file,
    gaitrite_excel=gaitrite_excel,
    sensor="HD",
    signal="acc",
    gait_id=None,
    plot=False
)

# Save for future ML / DL usage
out_file = "INGP065_F2_SC_HD_acc.npz"
print(f"Saving...")

np.savez(
    out_file,
    time=time_sec,
    data=data,
    events=events
)

print(f"Saved processed data to {out_file}")
