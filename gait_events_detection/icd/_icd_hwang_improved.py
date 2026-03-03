import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from typing import Any, Optional
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdHwangImproved(BaseIcDetector):
    """
    Initial Contact (IC) and Final Contact (FC) detector based on
    head vertical acceleration peak detection (Hwang et al., 2018).

    Description:
    ------------
    This is an optimized offline version of the Hwang algorithm. Instead of 
    sliding window FFT smoothing, it uses a zero-phase Butterworth low-pass 
    filter. ICs (Heel Strikes) are detected as primary peaks, while FCs 
    (Toe Offs) are identified as the third local peak following each IC.

    Algorithm Pipeline:
    1. Low-pass Butterworth filtering (default 6.0 Hz).
    2. Primary Peak Detection: ICs are peaks above a dynamic threshold 
       (default 0.3 * STD).
    3. Peak Counting Heuristic: For each interval between two ICs, local 
       peaks are identified. The 3rd peak in this sequence is labeled as the FC.

    Author: Paolo Tasca
    Date: 18 February 2026

    Reference:
    ----------
    Adapted from: Hwang et al. (2018). "Real-Time Gait Analysis Using a Single 
    Head-Worn Inertial Measurement Unit," IEEE Transactions on Consumer 
    Electronics, 64(2), pp. 240-248.
    """

    def __init__(
            self,
            lowpass_hz: float = 6.0,
            butter_order: int = 4,
            min_peak_distance_s: float = 0.4,
            peak_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the detector with optimized hyperparameters.

        Args:
            lowpass_hz: Cutoff frequency for the Butterworth low-pass filter.
            butter_order: Order of the Butterworth filter.
            min_peak_distance_s: Minimum time between consecutive ICs.
            peak_threshold: Fixed threshold for peak detection. If None, 
                            defaults to 0.3 * signal standard deviation.
        """
        super().__init__()
        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order
        self.min_peak_distance_s = min_peak_distance_s
        self.peak_threshold = peak_threshold

        # Public output attributes
        self.ic_list_: Optional[pd.DataFrame] = None
        self.ic_side_: Optional[pd.Series] = None
        self.fc_list_: Optional[pd.DataFrame] = None
        self.fc_side_: Optional[pd.Series] = None

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_column: str = "acc_is",
            plot_debug: bool = False,
            **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect Initial Contacts (IC) and Final Contacts (FC).
        """
        fs = sampling_rate_hz
        acc = data[acc_column].to_numpy()

        # 1. Low-pass Filtering
        nyq = 0.5 * fs
        b, a = butter(
            self.butter_order,
            self.lowpass_hz / nyq,
            btype="low",
        )
        acc_filt = filtfilt(b, a, acc)

        # 2. Peak Detection (Initial Contact)
        min_dist_samples = int(self.min_peak_distance_s * fs)

        if self.peak_threshold is None:
            threshold = 0.3 * np.std(acc_filt)
        else:
            threshold = self.peak_threshold

        ic_idx, _ = find_peaks(
            acc_filt,
            height=threshold,
            distance=min_dist_samples,
        )

        # 3. Final Contact Detection (3rd peak after IC)
        # ------------------------------------------------------------------
        # This heuristic assumes the vertical acceleration profile contains 
        # multiple sub-peaks per step, with the 3rd often representing Toe-Off.
        # ------------------------------------------------------------------
        fc_idx = []

        for i in range(len(ic_idx) - 1):
            start = ic_idx[i]
            end = ic_idx[i + 1]

            # Search for sub-peaks between current and next IC
            # Threshold is lowered (half of IC threshold) to capture smaller events
            peaks_between, _ = find_peaks(
                acc_filt[start:end],
                height=threshold / 2,
            )

            if len(peaks_between) >= 3:
                # Select the 3rd sub-peak
                fc_idx.append(start + peaks_between[2])

        # 4. Standardized Output Storage
        self.ic_list_ = pd.DataFrame(
            {"ic": ic_idx},
            index=pd.RangeIndex(len(ic_idx), name="step_id"),
        )
        self.ic_side_ = pd.Series(['U'] * len(ic_idx), index=self.ic_list_.index, name="side")

        self.fc_list_ = pd.DataFrame(
            {"fc": np.asarray(fc_idx, dtype=int)},
            index=pd.RangeIndex(len(fc_idx), name="step_id"),
        )
        self.fc_side_ = pd.Series(['U'] * len(fc_idx), index=self.fc_list_.index, name="side")

        if plot_debug:
            self._plot_debug(acc_filt, ic_idx, fc_idx)

        return self

    def _plot_debug(self, acc_filt: np.ndarray, ic_idx: np.ndarray, fc_idx: list):
        """Standardized plotting for visual verification."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(acc_filt, label='Low-pass Filtered Acc', color='tab:green', linewidth=1.5)

        ax.scatter(ic_idx, acc_filt[ic_idx], color='red', marker='v', s=80, label='IC (Primary Peak)')

        if fc_idx:
            ax.scatter(fc_idx, acc_filt[fc_idx], color='blue', marker='^', s=80, label='FC (3rd Peak)')

        ax.set_title("Hwang Improved: Peak Counting Heuristic")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.tight_layout()
        plt.show()