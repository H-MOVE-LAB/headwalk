import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from typing import Any, Optional
from typing_extensions import Self, Unpack

from .base import BaseGeDetector


class GedHwang(BaseGeDetector):
    """
    Hwang Initial and Final Contact detector (Original FFT-based).

    Description:
    ------------
    This algorithm implements the method proposed by Hwang et al. (2016, 2018)
    for head-worn IMUs. It uses a two-stage refinement process: a coarse peak
    detection followed by a fine-tuning step using a frequency-domain squared
    low-pass filter on a sliding window.

    Algorithm Pipeline:
    1. (Optional) Gravity removal via high-pass filtering.
    2. Coarse IC detection: Local maxima in vertical acceleration above a
       dynamic threshold (0.3 * STD).
    3. Fine IC refinement:
       - Extract a window of approx. 0.27s (16 samples at 60Hz) ending at the coarse peak.
       - Apply a frequency-domain squared low-pass filter (FFT -> Mask -> IFFT).
       - The maximum of this smoothed window is the refined IC.
    4. FC detection: The absolute minimum in the raw acceleration within a 0.5s
       window following the refined IC.

    Author: Paolo Tasca
    Date: 18 February 2026

    References:
    -----------
    [1] T.-H. Hwang et al. (2016). "Real-time gait event detection using a single
        head-worn inertial measurement unit," ICCE-Berlin, pp. 28-32.
        doi: 10.1109/ICCE-Berlin.2016.7684709

    [2] T.-H. Hwang et al. (2018). "Real-Time Gait Analysis Using a Single
        Head-Worn Inertial Measurement Unit," IEEE Transactions on Consumer
        Electronics, 64(2), pp. 240-248.
        doi: 10.1109/TCE.2018.2843289
    """

    def __init__(
            self,
            *,
            fft_window_s: float = 0.2667,  # 16 samples @ 60Hz (from 2018 paper)
            peak_threshold_std: float = 0.3,
            min_peak_distance_s: float = 0.4,
            max_ic_to_fc_s: float = 0.5
    ) -> None:
        """
        Initialize the detector with default hyperparameters.

        Args:
            fft_window_s: Temporal window size for FFT-based smoothing.
            peak_threshold_std: Threshold for coarse peak detection (ratio of signal std).
            min_peak_distance_s: Minimum time between coarse peaks.
            max_ic_to_fc_s: Maximum duration after IC to search for FC (minimum value).
        """
        super().__init__()
        self.fft_window_s = fft_window_s
        self.peak_threshold_std = peak_threshold_std
        self.min_peak_distance_s = min_peak_distance_s
        self.max_ic_to_fc_s = max_ic_to_fc_s

        # Public output attributes
        self.ic_list_: Optional[pd.DataFrame] = None
        self.ic_side_: Optional[pd.Series] = None
        self.fc_list_: Optional[pd.DataFrame] = None
        self.fc_side_: Optional[pd.Series] = None

    def _fft_lowpass(self, segment: np.ndarray) -> np.ndarray:
        """
        Squared low-pass filter in frequency domain (Hwang et al., 2018).
        Passband: |ω| < π/8.
        """
        N = len(segment)
        if N == 0:
            return segment

        fft_sig = np.fft.fft(segment)

        # The mask corresponds to |ω| < π/8, which is N/16 for real signals
        # but the original implementation used N/8 bins for 60Hz.
        cutoff = max(1, N // 8)
        mask = np.zeros(N)
        mask[:cutoff] = 1
        mask[-cutoff:] = 1

        return np.real(np.fft.ifft(fft_sig * mask))

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_column: str = "acc_is",
            plot_debug: bool = True,
            **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect Initial Contacts (IC) and Final Contacts (FC).
        """
        fs = sampling_rate_hz
        acc = data[acc_column].to_numpy()

        # 1. Coarse Peak Detection
        threshold = np.mean(acc) + self.peak_threshold_std * np.std(acc)
        min_dist_samples = int(self.min_peak_distance_s * fs)
        fft_win_samples = int(self.fft_window_s * fs)

        ic_candidates, _ = find_peaks(
            acc,
            height=threshold,
            distance=min_dist_samples,
        )

        ic_idx_list: list[int] = []
        fc_idx_list: list[int] = []

        # 3. FFT Refinement & FC Search
        for ic in ic_candidates:
            start = ic - fft_win_samples + 1
            end = ic + 1

            if start < 0:
                continue

            segment = acc[start:end]
            if len(segment) < 2:
                continue

            # Apply FFT-based smoothing
            smooth_seg = self._fft_lowpass(segment)

            # Refined IC is the maximum in the smoothed window
            local_ic = np.argmax(smooth_seg)
            ic_refined = start + local_ic
            ic_idx_list.append(ic_refined)

            # FC Search: Minimum value in raw signal after refined IC
            search_end = min(
                ic_refined + int(self.max_ic_to_fc_s * fs),
                len(acc)
            )
            post_seg = acc[ic_refined:search_end]

            if len(post_seg) > 0:
                fc_rel = np.argmin(post_seg)
                fc_idx_list.append(ic_refined + fc_rel)

        # 4. Standardized Output Storage
        self.ic_list_ = pd.DataFrame(
            {"ic": np.asarray(ic_idx_list, dtype=int)},
            index=pd.RangeIndex(len(ic_idx_list), name="step_id"),
        )
        self.ic_side_ = pd.Series(['U'] * len(ic_idx_list), index=self.ic_list_.index, name="side")

        self.fc_list_ = pd.DataFrame(
            {"fc": np.asarray(fc_idx_list, dtype=int)},
            index=pd.RangeIndex(len(fc_idx_list), name="step_id"),
        )
        self.fc_side_ = pd.Series(['U'] * len(fc_idx_list), index=self.fc_list_.index, name="side")

        if plot_debug:
            self._plot_debug(acc, ic_idx_list, fc_idx_list)

        return self

    def _plot_debug(self, acc, ic_idx, fc_idx):
        """Standardized plotting of detection steps."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(acc, label='Acceleration (Processed)', color='tab:blue', alpha=0.7)

        ax.scatter(ic_idx, acc[ic_idx], color='red', marker='v', s=100, label='IC (Refined)')
        ax.scatter(fc_idx, acc[fc_idx], color='black', marker='^', s=100, label='FC (Min search)')

        ax.set_title("Hwang (Original) Gait Event Detection")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.show()