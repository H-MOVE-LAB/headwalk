"""
Example: apply multiple ICD algorithms to a single H-IMU trial
"""

from pathlib import Path
import numpy as np
from scripts import load_npz_trial
from src.headwalk.utils import align_imu_to_gravity
# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
acc_file = DATA_DIR / "INGC116_F2_SC_HD_acc.npz"
gyr_file = DATA_DIR / "INGC116_F2_SC_HD_gyr.npz"

# --------------------------------------------------
# Load data
# --------------------------------------------------
_, acc, _ = load_npz_trial(acc_file)
t, gyr, events_gyr = load_npz_trial(gyr_file)
acc[:,[0, 2]] = -acc[:,[0, 2]]
gyr[:,[0, 2]] = -gyr[:,[0, 2]]

# --------------------------------------------------
# Combine acc and gyr data
# --------------------------------------------------

imu_data = np.concatenate((acc, gyr), axis=1)
# --------------------------------------------------
# Align to gravity
# --------------------------------------------------
sampling_rate_hz = 128
static_duration_s = 1
acc_aligned, gyr_aligned, R = align_imu_to_gravity(
        acc=acc,
        gyr=gyr,
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=static_duration_s,
        gravity_ideal=np.array([1.0, 0.0, 0.0])  # gravity along x-axis
    )

import matplotlib.pyplot as plt
# Create two subplots and unpack the output array immediately
s = 0
e = -1#5*sampling_rate_hz
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, dpi = 300)
ax1.plot(t[s:e], acc[s:e, :])
ax2.plot(t[s:e], acc_aligned[s:e, :])
plt.show()
print("Done!")
