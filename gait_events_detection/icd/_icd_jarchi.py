from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pyts.decomposition import SingularSpectrumAnalysis
from typing_extensions import Self, Unpack

from .base import BaseIcDetector

class IcdJarchi(BaseIcDetector):
    """
    Initial Contact (IC) and Final Contact (FC) detection based on
    ear-worn IMU signals using SSA and extrema detection
    (inspired by Jarchi et al., 2014).
    """
    #TODO: align to gravity prior to gait events detection
    def __init__(
        self,
        window_length: int = 200,
        min_peak_distance_s: float = 0.4,
    ) -> None:
        self.window_length = window_length
        self.min_peak_distance_s = min_peak_distance_s
        self.ssa = SingularSpectrumAnalysis(
            window_size=window_length,
            groups=[[0], [1, 2], np.arange(3, window_length, 1)]
        )

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_columns: Dict[str, str] = {"si": "acc_is", "ml": "acc_ml", "pa": "acc_ap"},
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect IC and FC events using Jarchi algorithm logic.

        Parameters
        ----------
        data : pd.DataFrame
            IMU data with columns for sagittal (si), medial-lateral (ml),
            and posterior-anterior (pa) axes.
        sampling_rate_hz : float
            Sampling rate of the data.
        acc_columns : dict
            Dictionary mapping axis names to dataframe columns.

        Returns
        -------
        self
        """
        data[acc_columns["si"]] = -data[acc_columns["si"]]
        self.data = data
        fs = sampling_rate_hz
        min_dist_samples = int(self.min_peak_distance_s * fs)

        # --------------------------------------------------
        # 1. Apply SSA to each axis
        # --------------------------------------------------
        acc_ssa = {}
        for axis in acc_columns.values():
            acc_ssa[axis] = self.ssa.fit_transform(data[axis].to_numpy().reshape(1, -1))

        # --------------------------------------------------
        # 2. Detect IC events on dominant AP component
        # --------------------------------------------------
        pa_dom = acc_ssa[acc_columns["pa"]][1]  # dominant component
        mins, _ = find_peaks(-pa_dom, height=0, distance=min_dist_samples)

        # Remove trend from all axes (sum of first three components + mean)
        acc_wo_trend = {}
        for axis in acc_columns.values():
            comp = acc_ssa[axis]
            acc_wo_trend[axis] = comp[1] + comp[2] + np.mean(data[axis])

        # IC events refined with min search in interval
        t1 = int(0.05 * fs)
        ic_idx = np.array([
            self._find_min_in_interval(acc_wo_trend[acc_columns["pa"]],
                                       -acc_wo_trend[acc_columns["si"]],
                                       peak, t1)
            for peak in mins
        ])

        # Determine side (ipsilateral / contralateral) based on ML axis
        ic_side = []
        for i in range(len(ic_idx) - 2):
            mean1 = np.mean(acc_wo_trend[acc_columns["ml"]][ic_idx[i]:ic_idx[i+1]])
            mean2 = np.mean(acc_wo_trend[acc_columns["ml"]][ic_idx[i+1]:ic_idx[i+2]])
            ic_side.append("ipsilateral" if mean1 < mean2 else "contralateral")
        ic_side = (
            ic_side + ["ipsilateral", "contralateral"]
            if ic_side[-1] == "contralateral"
            else ic_side + ["contralateral", "ipsilateral"]
        )

        # --------------------------------------------------
        # 3. Detect FC events using ML axis extrema
        # --------------------------------------------------
        ml_trend = acc_wo_trend[acc_columns["ml"]]
        maximas, _ = find_peaks(ml_trend, width=int(0.1 * fs))
        minimas, _ = find_peaks(-ml_trend, width=int(0.1 * fs))

        fc_ipsi, fc_contra = [], []
        for idx, side in zip(ic_idx, ic_side):
            if side == "ipsilateral":
                # first local minimum after contralateral IC
                potential = minimas[minimas > idx]
                fc_ipsi.append(potential[0] if len(potential) else np.nan)
            else:
                # last local maximum before contralateral IC
                potential = maximas[maximas < idx]
                fc_contra.append(potential[-1] if len(potential) else np.nan)

        # Build dataframes
        ic_df = pd.DataFrame({"ic": ic_idx, "side": ic_side}, index=pd.RangeIndex(len(ic_idx)))
        fc_idx = np.concatenate([fc_ipsi, fc_contra])
        fc_side = ["ipsilateral"]*len(fc_ipsi) + ["contralateral"]*len(fc_contra)
        fc_df = pd.DataFrame({"fc": fc_idx, "side": fc_side}, index=pd.RangeIndex(len(fc_idx)))

        # Store results
        self.ic_list_ = ic_df
        self.fc_list_ = fc_df

        return self

    @staticmethod
    def _find_min_in_interval(signal_pa, signal_si, peak, t):
        """Find local minimum in ±t window using product of PA and SI axes."""
        if peak < t:
            t = peak
        elif peak + t > len(signal_pa):
            t = len(signal_pa) - peak
        interval = signal_pa[peak - t:peak + t] * signal_si[peak - t:peak + t]
        return np.argmin(interval) + peak - t
