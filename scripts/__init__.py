import numpy as np
from pathlib import Path

def load_npz_trial(npz_path: str | Path):
    """
    Load a processed ICICLE trial saved as .npz.

    Returns
    -------
    time : np.ndarray
        Time vector in seconds
    data : np.ndarray
        IMU data (N x C)
    events : np.ndarray
        Gait events array [IC, FC]
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    return data["time"], data["data"], data["events"]

__all__ = ["load_npz_trial"]
