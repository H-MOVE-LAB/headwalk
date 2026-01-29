"""
Example: apply multiple ICD algorithms to a single H-IMU trial
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from gait_events_detection.icd import (
    IcdFang,
    IcdHwang,
    IcdHwangImproved,
    IcdJarchi
)
from gait_events_detection.data import load_npz_trial

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
npz_file = DATA_DIR / "INGC116_F2_SC_HD_acc.npz"

# --------------------------------------------------
# Load data
# --------------------------------------------------
time, imu_data, events = load_npz_trial(npz_file)

# Body frame convention (Mobilise-D):
# x -> inferosuperior (is)
# y -> mediolateral (ml)
# z -> anteroposterior (ap)
imu_data[:, 0] = -imu_data[:, 0]
imu_data[:, 2] = -imu_data[:, 2]

# --------------------------------------------------
# Resample to 128 Hz
# --------------------------------------------------
target_fs = 128.0

time_rs = np.arange(time[0], time[-1], 1 / target_fs)
interp_fun = interp1d(
    time,
    imu_data,
    axis=0,
    bounds_error=False,
    fill_value="extrapolate",
)
imu_rs = interp_fun(time_rs)

df = pd.DataFrame(
    imu_rs,
    columns=["acc_is", "acc_ml", "acc_ap"],
)

# --------------------------------------------------
# Apply algorithms
# --------------------------------------------------
algorithms = {
    "Fang": IcdFang(),
    "Hwang": IcdHwang(),
    "HwangImp": IcdHwangImproved(),
    "Jarchi": IcdJarchi()
}

results = {}

for name, alg in algorithms.items():
    alg.detect(df, sampling_rate_hz=target_fs)
    results[name] = alg

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(time_rs, df["acc_is"], color="black", label="Vertical Acc")

# GaitRite (seconds)
# plt.scatter(events[:, 0], np.zeros_like(events[:, 0]),
#             color="red", marker="o", label="GaitRite IC")
# plt.scatter(events[:, 1], np.zeros_like(events[:, 1]),
#             color="red", marker="x", label="GaitRite FC")
for i, t in enumerate(events[:, 0]):
    plt.axvline(x=t, color="red", linestyle="--", linewidth=2, label="GaitRite IC" if i == 0 else None)
for i, t in enumerate(events[:, 1]):
    plt.axvline(x=t, color="red", linestyle=":", linewidth=2, label="GaitRite FC" if i == 0 else None)
# ICD results
for name, alg in results.items():
    ic_sec = alg.ic_list_["ic"].to_numpy() / target_fs
    plt.scatter(
        ic_sec,
        np.interp(ic_sec, time_rs, df["acc_is"]),
        marker="o",
        label=f"{name} IC",
    )

plt.xlabel("Time [s]")
plt.ylabel("Vertical Acceleration [m/s²]")
plt.legend()
plt.title("IC Detection Comparison (H-IMU)")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Done!")