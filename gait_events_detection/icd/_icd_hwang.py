from typing import Any
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdHwang(BaseIcDetector):
    """
    Initial Contact (IC) and Final Contact (FC) detection based on
    head vertical acceleration peak detection (Hwang et al., 2018).

    This class implements the ORIGINAL algorithm, including
    FFT/IFFT-based smoothing on short sliding windows.
    """

    def __init__(
        self,
        *,
        remove_gravity: bool = False,
        highpass_hz: float = 0.3,
        butter_order: int = 2,
        fft_window: int = 16,
        peak_threshold_std: float = 0.3,
        min_peak_distance_s: float = 0.4,
    ) -> None:
        self.remove_gravity = remove_gravity
        self.highpass_hz = highpass_hz
        self.butter_order = butter_order
        self.fft_window = fft_window
        self.peak_threshold_std = peak_threshold_std
        self.min_peak_distance_s = min_peak_distance_s

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _highpass_filter(self, signal: np.ndarray, fs: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(
            self.butter_order,
            self.highpass_hz / nyq,
            btype="high",
        )
        return filtfilt(b, a, signal)

    def _fft_lowpass(self, segment: np.ndarray) -> np.ndarray:
        """
        Squared low-pass filter in frequency domain.
        Passband: |ω| < π/8 (Hwang et al.)
        """
        N = len(segment)
        fft_sig = np.fft.fft(segment)

        cutoff = N // 8
        mask = np.zeros(N)
        mask[:cutoff] = 1
        mask[-cutoff:] = 1

        return np.real(np.fft.ifft(fft_sig * mask))

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_column: str = "acc_is",
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect Initial Contacts (IC) and Final Contacts (FC).
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        fs = sampling_rate_hz
        acc = data[acc_column].to_numpy()

        # --------------------------------------------------
        # 1. Optional gravity removal (high-pass)
        # --------------------------------------------------
        if self.remove_gravity:
            acc = self._highpass_filter(acc, fs)

        # --------------------------------------------------
        # 2. Peak detection (IC candidates)
        # --------------------------------------------------
        threshold = self.peak_threshold_std * np.std(acc)
        min_dist_samples = int(self.min_peak_distance_s * fs)

        ic_candidates, _ = find_peaks(
            acc,
            height=threshold,
            distance=min_dist_samples,
        )

        ic_idx: list[int] = []
        fc_idx: list[int] = []

        # --------------------------------------------------
        # 3. FFT/IFFT refinement + FC detection
        # --------------------------------------------------
        for ic in ic_candidates:
            start = ic - self.fft_window + 1
            end = ic + 1

            if start < 0:
                continue

            segment = acc[start:end]
            if len(segment) != self.fft_window:
                continue

            # FFT-based smoothing
            smooth_seg = self._fft_lowpass(segment)

            # IC = maximum in smoothed window
            local_ic = np.argmax(smooth_seg)
            ic_refined = start + local_ic
            ic_idx.append(ic_refined)

            # FC = first minimum after IC
            search_end = min(
                ic_refined + int(0.5 * fs),
                len(acc),
            )
            post_seg = acc[ic_refined:search_end]

            if len(post_seg) == 0:
                continue

            fc_rel = np.argmin(post_seg)
            fc_idx.append(ic_refined + fc_rel)

        # --------------------------------------------------
        # 4. Store results
        # --------------------------------------------------
        self.ic_list_ = pd.DataFrame(
            {"ic": np.asarray(ic_idx, dtype=int)},
            index=pd.RangeIndex(len(ic_idx), name="step_id"),
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": np.asarray(fc_idx, dtype=int)},
            index=pd.RangeIndex(len(fc_idx), name="step_id"),
        )

        return self
