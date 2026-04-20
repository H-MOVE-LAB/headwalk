"""
Example: apply McCamley IC detector to a single trial
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from src.headwalk.gait_events_detection import IcdFang
from scripts import load_npz_trial

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
npz_file = DATA_DIR / "INGC116_F2_SC_HD_acc.npz"

# --------------------------------------------------
# Load data
# --------------------------------------------------
time, imu_data, events = load_npz_trial(npz_file)
# From ICICLE to Mobilise-D local frame convention
imu_data[:, 0] = -imu_data[:, 0]
imu_data[:, 2] = -imu_data[:, 2]

# --------------------------------------------------
# RESAMPLE IMU DATA TO 128 Hz
# --------------------------------------------------
target_fs = 128.0  # Hz

# time_sec: (N,)
# data: (N, C)  -> e.g. accelerometer or gyroscope channels

t_start = time[0]
t_end = time[-1]

# New uniform time base
time_resampled = np.arange(t_start, t_end, 1.0 / target_fs)

# Interpolator (works for 1 or multiple channels)
interp_fun = interp1d(
    time,
    imu_data,
    axis=0,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate",
)

# Resampled signal
imu_data_rs = interp_fun(time_resampled)

df = pd.DataFrame(
    imu_data_rs,
    columns=["acc_is", "acc_ml", "acc_ap"]
)

# --------------------------------------------------
# Apply IC detector
# --------------------------------------------------
icd = IcdFang()
icd.detect(df, sampling_rate_hz=target_fs)

import matplotlib.pyplot as plt

# Convert Fang IC/FC from samples to seconds
fang_ic_sec = icd.ic_list_['ic'].to_numpy() / target_fs
fang_fc_sec = icd.fc_list_['fc'].to_numpy() / target_fs

# GaitRite IC/FC
gait_ic_sec = events[:, 0]  # column 0 = IC
gait_fc_sec = events[:, 1]  # column 1 = FC

# Vertical acceleration (resampled)
acc_vert = df['acc_is'].to_numpy()

plt.figure(figsize=(14, 5))
plt.plot(time_resampled, acc_vert, label="Vertical Acceleration", color="black")

# Plot GaitRite events
plt.scatter(gait_ic_sec, np.interp(gait_ic_sec, time_resampled, acc_vert),
            color='red', marker='o', s=50, label='GaitRite IC')
plt.scatter(gait_fc_sec, np.interp(gait_fc_sec, time_resampled, acc_vert),
            color='red', marker='x', s=50, label='GaitRite FC')

# Plot Fang-detected events
plt.scatter(fang_ic_sec, np.interp(fang_ic_sec, time_resampled, acc_vert),
            color='blue', marker='o', s=50, label='Fang IC')
plt.scatter(fang_fc_sec, np.interp(fang_fc_sec, time_resampled, acc_vert),
            color='blue', marker='x', s=50, label='Fang FC')

plt.xlabel("Time [s]")
plt.ylabel("Vertical Acceleration [m/s²]")
plt.title("Vertical Acceleration with Gait Events (IC/FC)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
