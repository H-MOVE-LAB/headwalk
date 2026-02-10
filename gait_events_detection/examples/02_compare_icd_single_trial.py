"""
Example: apply multiple ICD algorithms to a single H-IMU trial
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from utils import align_imu_to_gravity

from gait_events_detection.icd import (
    IcdFang,
    IcdHwang,
    IcdHwangImproved,
    IcdJarchi,
    IcdDiao,
    IcdTasca,
    IcdTomc,
    IcdTcn
)
from gait_events_detection.data import load_npz_trial

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
acc_file = DATA_DIR / "INGC116_F2_SC_HD_acc.npz"
gyr_file = DATA_DIR / "INGC116_F2_SC_HD_gyr.npz"
# Path to trained TCN model
MODEL_PATH = Path(__file__).resolve().parents[1] /"models/lastTrainedModel"
# --------------------------------------------------
# Load data
# --------------------------------------------------
_, acc, _ = load_npz_trial(acc_file)
t, gyr, events = load_npz_trial(gyr_file)
# Body frame convention (Mobilise-D):
# x -> inferosuperior (is)
# y -> mediolateral (ml)
# z -> anteroposterior (ap)
acc[:,[0, 2]] = -acc[:,[0, 2]]
gyr[:,[0, 2]] = -gyr[:,[0, 2]]

# Combine acc and gyr data
imu_data = np.concatenate((acc, gyr), axis=1)

# --------------------------------------------------
# Resample to 128 Hz
# --------------------------------------------------
target_fs = 128.0

time_rs = np.arange(t[0], t[-1], 1 / target_fs)
interp_fun = interp1d(
    t,
    imu_data,
    axis=0,
    bounds_error=False,
    fill_value="extrapolate",
)
imu_rs = interp_fun(time_rs)

# --------------------------------------------------
# Align to ideal vector [1, 0, 0]
# --------------------------------------------------
static_duration_s = 1
acc_aligned, gyr_aligned, R = align_imu_to_gravity(
        acc=acc,
        gyr=gyr,
        sampling_rate_hz=target_fs,
        static_duration_s=static_duration_s,
        gravity_ideal=np.array([1.0, 0.0, 0.0])  # gravity along x-axis
    )

df = pd.DataFrame(
    imu_rs,
    columns=["acc_is", "acc_ml", "acc_ap", "gyr_is", "gyr_ml", "gyr_ap"],
)

# --------------------------------------------------
# Apply algorithms
# --------------------------------------------------
algorithms = {
    "Fang": IcdFang(),
    "Hwang": IcdHwang(),
    "HwangImp": IcdHwangImproved(),
    "Jarchi": IcdJarchi(),
    "Diao": IcdDiao(),
    # "IcdTasca": IcdTasca()
    "Tomc": IcdTomc(),
    "TCN": IcdTcn(model_path=MODEL_PATH)
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

for i, t in enumerate(events[:, 0]):
    plt.axvline(x=t, color="red", linestyle="--", linewidth=2, label="GaitRite IC" if i == 0 else None)
# for i, t in enumerate(events[:, 1]):
#     plt.axvline(x=t, color="red", linestyle=":", linewidth=2, label="GaitRite FC" if i == 0 else None)
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