"""
Example: apply multiple ICD algorithms to a single H-IMU trial
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.headwalk.utils import align_imu_to_gravity

from src.headwalk.gait_events_detection import (
    GedDiaoComplete
)
from scripts import load_npz_trial

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
acc_file = DATA_DIR / "INGC116_F2_SC_HD_acc.npz"
gyr_file = DATA_DIR / "INGC116_F2_SC_HD_gyr.npz"
# Path to trained TCN model
MODEL_PATH = Path(__file__).resolve().parents[1] /"models/bestModel8"
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

# Combine acc and gyr data
imu_data = np.concatenate((acc_aligned, gyr_aligned), axis=1)

df = pd.DataFrame(
    imu_data,
    columns=["acc_is", "acc_ml", "acc_ap", "gyr_is", "gyr_ml", "gyr_ap"],
)

# --------------------------------------------------
# Apply algorithms
# --------------------------------------------------
algorithms = {
    # "Fang": IcdFang(),
    # "Hwang": IcdHwang(),
    # "Jarchi": IcdJarchi(),
    "Diao": GedDiaoComplete(iterative_filtering=True),
    # "Tomc": IcdTomc(),
    # "TCN": IcdTcn(),
    # "CNN": IcdCnn(),
    # "Seifer": IcdSeifer(),
    # "Fawden": IcdFawden(),
    # "Jiang": IcdJiang(),
    # 'Transformer': IcdTransformer(),
}

results = {}

for name, alg in algorithms.items():
    alg.detect(df, sampling_rate_hz=target_fs, plot_debug= True)
    results[name] = alg

# --------------------------------------------------
# Plot
# --------------------------------------------------
ax_ = "acc_is"
plt.figure(figsize=(14, 5))
plt.plot(time_rs, df[ax_], color="black", label="Vertical Acc", linewidth = 3, zorder=1)

for i, t in enumerate(events[:, 0]):
    plt.axvline(x=t, color="red", linestyle="--", linewidth=2, label="GaitRite IC" if i == 0 else None)
for i, t in enumerate(events[:, 1]):
    plt.axvline(x=t, color="red", linestyle=":", linewidth=2, label="GaitRite FC" if i == 0 else None)
# ICD results
for name, alg in results.items():
    ic_sec = alg.ic_list_["ic"].to_numpy() / target_fs
    plt.scatter(
        ic_sec,
        np.interp(ic_sec, time_rs, df[ax_]),
        marker="o",
        label=f"{name} IC",
        s=100,
    )
# FCD results
for name, alg in results.items():
    fc_sec = alg.fc_list_["fc"].to_numpy() / target_fs
    if fc_sec.size != 0:
        plt.scatter(
            fc_sec,
            np.interp(fc_sec, time_rs, df[ax_]),
            marker="o",
            label=f"{name} FC",
            s=100,
        )
plt.xlabel("Time (s)", fontsize = 30)
plt.ylabel("Vertical Acceleration (m/s²)", fontsize = 30)
plt.legend(fontsize = 30)
plt.title("IC Detection Comparison (H-IMU)", fontsize = 30)
plt.grid(True)
plt.tight_layout()
plt.show()
print("Done!")