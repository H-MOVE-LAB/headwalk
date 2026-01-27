"""
IC detection algorithm following DelDin / McCamley original implementation (vertical acceleration).
Used by Fang et al., (2024) for step detection in people with Parkinson's Disease

Author: Paolo Tasca
Date: 26-Jan-2026
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from pywt import cwt
from tpcp import cf

from .base import BaseIcDetector
from mobgap.data_transform import EpflDedriftedGaitFilter, EpflGaitFilter, Resample
from mobgap.data_transform.base import BaseFilter
from mobgap.initial_contacts._utils import find_zero_crossings


class IcdMcCamley(BaseIcDetector):
    """IC detection using vertical acceleration of lower-back / trunk (here applied to head data as described by Fang et al., (2024)) """

    pre_filter: BaseFilter
    cwt_width: float
    _internal_sampling_hz: int = 40  # internal downsampling

    def __init__(
        self, *, pre_filter: BaseFilter = cf(EpflDedriftedGaitFilter()), cwt_width: float = 9.0
    ) -> None:
        self.pre_filter = pre_filter
        self.cwt_width = cwt_width

    def detect(
        self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Any
    ) -> "IcdMcCamley":
        """
        Detect initial contacts using McCamley method:
            1. Resample to 40 Hz
            2. Band-pass filter (pre-filter)
            3. Cumulative integral
            4. Continuous Wavelet Transform (Ricker/Gaussian)
            5. Zero-crossing detection
            6. Detect minima between zero crossings (negative peaks = ICs)
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # 0. Select vertical acceleration column (assume 'acc_is' exists)
        acc_v = data[["acc_is"]].to_numpy()

        # 1. Resample
        downsampled = (
            Resample(self._internal_sampling_hz)
            .transform(acc_v, sampling_rate_hz=sampling_rate_hz)
            .transformed_data_.squeeze()
        )

        # 2. Pre-filter with padding
        n_coeffs = len(EpflGaitFilter().coefficients[0])
        pad_len = 4 * n_coeffs
        padded_signal = np.pad(downsampled, (pad_len, pad_len), "wrap")
        filtered = (
            self.pre_filter.clone()
            .filter(padded_signal, sampling_rate_hz=self._internal_sampling_hz)
            .filtered_data_.squeeze()
        )
        filtered = filtered[pad_len - 1 : -pad_len]  # remove padding

        # 3. Cumulative integral
        integral = cumulative_trapezoid(filtered, initial=0) / self._internal_sampling_hz

        # 4. Continuous Wavelet Transform (gaussian)
        cwt_signal, _ = cwt(
            integral.squeeze(),
            [self.cwt_width],
            "gaus2",
            sampling_period=1 / self._internal_sampling_hz,
        )
        cwt_signal = cwt_signal.squeeze() - cwt_signal.mean()

        # 5. Detect ICs (minima between zero crossings)
        ic_indices = self._find_minima_between_zero_crossings(cwt_signal)

        # 6. Convert to original sampling rate
        detected = pd.DataFrame({"ic": ic_indices})
        detected_unsampled = (
            (detected * sampling_rate_hz / self._internal_sampling_hz).round().astype("int64")
        )
        self.ic_list_ = detected_unsampled

        return self

    @staticmethod
    def _find_minima_between_zero_crossings(signal: np.ndarray) -> np.ndarray:
        """Helper: find minima between zero crossings."""
        zeros = find_zero_crossings(signal, "both", refine=False).astype("int64")
        if len(zeros) == 0:
            return np.array([])

        pos_to_neg = zeros[signal[zeros] >= 0] + 1
        neg_to_pos = zeros[signal[zeros] < 0] + 1
        if not (signal[zeros][0] >= 0):
            neg_to_pos = neg_to_pos[1:]

        minima = np.array([np.argmin(signal[s:e]) + s for s, e in zip(pos_to_neg, neg_to_pos)]).astype("int64")
        return minima
