"""
Explore a single ICICLE raw HDF5 file and overlay GaitRite Initial Contacts.

- Reads APDM ICICLE .h5 file
- Extracts Head (HD) IMU data
- Loads corresponding GaitRite Excel annotations
- Plots AP acceleration (z-axis) with ICs overlaid

Author: ---
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
h5_file = PROJECT_ROOT / "data" / "ICICLE-Gait_F2_SC" / "Controls" / "20140616-104931-INGC116F2_SC.h5"
gaitrite_excel = PROJECT_ROOT / "metadata" / "Paolo_step_data.xlsx"

sensor_key = "HD"       # Head sensor
acc_axis = 2            # z-axis (0=x,1=y,2=z)

# --------------------------------------------------
# 1. PARSE SUBJECT / TIMEPOINT / TEST FROM FILENAME
# --------------------------------------------------
fname = Path(h5_file).name
# Example: 20140430-142918-INGP065F2_SC.h5
id_part = fname[16:]                 # INGP065F2_SC.h5
subject_tp = id_part.split("_")[0]   # INGP065F2
subject = subject_tp[:7]             # INGP065
timepoint = subject_tp[7:]           # F2
test_opal = id_part.split("_")[1][:2]  # SC

print(f"Subject: {subject}")
print(f"Timepoint: {timepoint}")
print(f"Test: {test_opal}")

# Source - https://stackoverflow.com/a
# Posted by Alex44, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-20, License - CC BY-SA 4.0

# import h5py

# filename_hdf = 'data.hdf5'

# def h5_tree(val, pre=''):
#     items = len(val)
#     for key, val in val.items():
#         items -= 1
#         if items == 0:
#             # the last item
#             if type(val) == h5py._hl.group.Group:
#                 print(pre + '└── ' + key)
#                 h5_tree(val, pre+'    ')
#             else:
#                 try:
#                     print(pre + '└── ' + key + ' (%d)' % len(val))
#                 except TypeError:
#                     print(pre + '└── ' + key + ' (scalar)')
#         else:
#             if type(val) == h5py._hl.group.Group:
#                 print(pre + '├── ' + key)
#                 h5_tree(val, pre+'│   ')
#             else:
#                 try:
#                     print(pre + '├── ' + key + ' (%d)' % len(val))
#                 except TypeError:
#                     print(pre + '├── ' + key + ' (scalar)')

# with h5py.File(h5_file, 'r') as hf:
#     # print(hf)
#     # h5_tree(hf)
#     xxx = hf['SI-000925/Calibrated/Accelerometers'][()]
#     lista_nomi = hf.attrs['MonitorLabelList']
# --------------------------------------------------
# 2. READ RAW H5 DATA (FINAL, APDM-CORRECT)
# --------------------------------------------------
with h5py.File(h5_file, "r") as f:

    monitor_labels = f.attrs["MonitorLabelList"]
    case_ids = f.attrs["CaseIdList"]

    sensor_idx = None
    for i, lbl in enumerate(monitor_labels):
        if lbl.decode("utf-8") == sensor_key:
            sensor_idx = i
            break

    if sensor_idx is None:
        raise RuntimeError(f"Sensor {sensor_key} not found")

    sensor_group = case_ids[sensor_idx].decode("utf-8")
    base_path = f"/{sensor_group}"

    # Read calibrated signals
    acc = f[f"{base_path}/Calibrated/Accelerometers"][:].T
    gyr = f[f"{base_path}/Calibrated/Gyroscopes"][:].T

    # Read time vector (already in seconds)
    time = f[f"{base_path}/Time"][:]
    time_sec = (time - time[0])/1000000
acc_ap = acc[acc_axis, :]

# --------------------------------------------------
# 3. LOAD GAITRITE ICs FROM EXCEL
# --------------------------------------------------
T = pd.read_excel(gaitrite_excel)

T["First_name"] = T["First_name"].astype(str)
T["WalkTask"] = T["WalkTask"].astype(str)

key = subject + timepoint

subset = T[T["First_name"] == key]

# Map SC/SI ↔ Cont/Interm
if test_opal == "SC":
    subset = subset[subset["WalkTask"] == "Cont"]
elif test_opal == "SI":
    subset = subset[subset["WalkTask"] == "Interm"]

if subset.empty:
    raise RuntimeError("No GaitRite data found")

IC_times = np.sort(subset["FirstContact"].values)

# Convert IC times → nearest IMU samples
IC_idx = np.array([np.argmin(np.abs(time_sec - ic)) for ic in IC_times])

# --------------------------------------------------
# 4. PLOT
# --------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(time_sec, -acc_ap, label="AP acceleration (HD)", linewidth=1.2)

plt.plot(
    time_sec[IC_idx],
    -acc_ap[IC_idx],
    'ro',
    ms = 10,
    label="Initial Contacts (GaitRite)",
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title(f"{subject} {timepoint} {test_opal} – Head IMU")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

