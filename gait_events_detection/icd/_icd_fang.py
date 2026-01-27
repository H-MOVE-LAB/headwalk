"""
Fang / Del Din based Initial Contact (HS) and Final Contact (TO) detection.

Implementation following:
- Del Din et al. (2016, 2020)
- Fang et al. (2022)
- Pham et al. (2017) for wavelet scale selection

Pipeline:
1. Linear detrending
2. Low-pass Butterworth filtering (3.2 Hz, 10th order)
3. Signal integration (cumtrapz)
4. First differentiation via Gaussian CWT (gaus1)
5. IC (HS): local minima of first derivative
6. Second differentiation
7. FC (TO): local maxima of second derivative

Author: ---
Date: 26 January 2026
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.integrate import cumulative_trapezoid
import pywt
from scipy.signal import detrend
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdFang(BaseIcDetector):
    """
    Fang / Del Din Initial and Final Contact detector.

    Expected input:
    - Single gait sequence
    - Vertical acceleration (e.g. acc_z)

    Outputs:
    - ic_list_: heel strikes (HS)
    - fc_list_: toe offs (TO)
    """

    def __init__(
        self,
        lowpass_hz: float = 3.2,
        butter_order: int = 10,
        min_step_duration_s: float = 0.44,
    ) -> None:
        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order
        self.min_step_duration_s = min_step_duration_s

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_column: str = "acc_is",
            **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect Initial Contacts (Heel Strikes, HS) and Final Contacts (Toe-Offs, TO)
        using Fang / Del Din method with vertical acceleration.

        Parameters
        ----------
        data : pd.DataFrame
            IMU data containing vertical acceleration.
        sampling_rate_hz : float
            Sampling rate in Hz.
        acc_column : str
            Column name of vertical acceleration.

        Returns
        -------
        self
        """


        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        acc = data[acc_column].to_numpy()
        fs = sampling_rate_hz
        dt = 1.0 / fs

        # --------------------------------------------------
        # 1. LINEAR DETRENDING
        # --------------------------------------------------
        acc_dt = detrend(acc, type="linear")

        # --------------------------------------------------
        # 2. LOW-PASS FILTERING (Butterworth)
        # --------------------------------------------------
        acc_lp = _lowpass_filter(
            acc_dt,
            fs=fs,
            cutoff=self.lowpass_hz,
            order=self.butter_order,
        )

        # --------------------------------------------------
        # 3. INTEGRATION
        # --------------------------------------------------
        acc_int = cumulative_trapezoid(acc_lp, initial=0) * dt

        # --------------------------------------------------
        # 4. FIRST DIFFERENTIATION (Gaussian CWT)
        # --------------------------------------------------
        # Estimate dominant frequency
        fa = _dominant_frequency(acc, fs)
        fc = pywt.central_frequency("gaus2")
        scale = fc / (fa * dt)

        # Continuous wavelet transform
        cwt_coeffs, _ = pywt.cwt(
            acc_int,
            scales=[scale],
            wavelet="gaus2",
            sampling_period=dt,
        )
        d1 = cwt_coeffs[0].squeeze()
        d1 -= d1.mean()

        # --------------------------------------------------
        # 5. INITIAL CONTACTS (HS) - local minima
        # --------------------------------------------------
        min_dist_samples = int(self.min_step_duration_s * fs)
        ic_idx, _ = find_peaks(-d1, distance=min_dist_samples)

        # --------------------------------------------------
        # 6. SECOND DIFFERENTIATION (CWT again)
        # --------------------------------------------------
        cwt_coeffs2, _ = pywt.cwt(
            d1,
            scales=[scale],
            wavelet="gaus1",
            sampling_period=dt,
        )
        d2 = cwt_coeffs2[0].squeeze()
        d2 -= d2.mean()

        # --------------------------------------------------
        # 7. FINAL CONTACTS (TO) - local maxima
        # --------------------------------------------------
        fc_idx, _ = find_peaks(d2, distance=min_dist_samples)

        # --------------------------------------------------
        # 8. STORE RESULTS
        # --------------------------------------------------
        self.ic_list_ = pd.DataFrame(
            {"ic": ic_idx}, index=pd.RangeIndex(len(ic_idx), name="step_id")
        )
        self.fc_list_ = pd.DataFrame(
            {"fc": fc_idx}, index=pd.RangeIndex(len(fc_idx), name="step_id")
        )

        return self


# ======================================================
# UTILITIES
# ======================================================
def _lowpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff: float,
    order: int,
) -> np.ndarray:
    nyq = 0.5 * fs
    norm_cut = cutoff / nyq
    b, a = butter(order, norm_cut, btype="low")
    return filtfilt(b, a, signal)


def _dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """
    Estimate dominant frequency using FFT peak.
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum[0] = 0  # remove DC
    return freqs[np.argmax(spectrum)]
