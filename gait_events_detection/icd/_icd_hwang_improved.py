from typing import Any
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdHwangImproved(BaseIcDetector):
    """
    Initial Contact (HS) and Final Contact (TO) detection based on
    head vertical acceleration peak detection (Hwang et al., 2018).
    This version imnplements an optimized versione of the abovementioned method (which was originally devised for real-time implementation)
    """

    def __init__(
        self,
        lowpass_hz: float = 6.0,
        butter_order: int = 4,
        min_peak_distance_s: float = 0.4,
        peak_threshold: float | None = None,
    ) -> None:
        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order
        self.min_peak_distance_s = min_peak_distance_s
        self.peak_threshold = peak_threshold

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_column: str = "acc_is",
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect HS (Initial Contact) and TO (Final Contact) events.
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        fs = sampling_rate_hz
        acc = data[acc_column].to_numpy()

        # --------------------------------------------------
        # 1. LOW-PASS FILTERING
        # --------------------------------------------------
        nyq = 0.5 * fs
        b, a = butter(
            self.butter_order,
            self.lowpass_hz / nyq,
            btype="low",
        )
        acc_filt = filtfilt(b, a, acc)

        # --------------------------------------------------
        # 2. PEAK DETECTION (HS)
        # --------------------------------------------------
        min_dist_samples = int(self.min_peak_distance_s * fs)

        if self.peak_threshold is None:
            threshold = 0.3 * np.std(acc_filt)
        else:
            threshold = self.peak_threshold

        hs_idx, properties = find_peaks(
            acc_filt,
            height=threshold,
            distance=min_dist_samples,
        )

        # --------------------------------------------------
        # 3. TO DETECTION (3rd peak after HS)
        # --------------------------------------------------
        to_idx = []

        for i in range(len(hs_idx) - 1):
            start = hs_idx[i]
            end = hs_idx[i + 1]

            peaks_between, _ = find_peaks(
                acc_filt[start:end],
                height=threshold / 2,
            )

            if len(peaks_between) >= 3:
                to_idx.append(start + peaks_between[2])

        # --------------------------------------------------
        # 4. STORE RESULTS
        # --------------------------------------------------
        self.ic_list_ = pd.DataFrame(
            {"ic": hs_idx},
            index=pd.RangeIndex(len(hs_idx), name="step_id"),
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": np.asarray(to_idx, dtype=int)},
            index=pd.RangeIndex(len(to_idx), name="step_id"),
        )

        return self
