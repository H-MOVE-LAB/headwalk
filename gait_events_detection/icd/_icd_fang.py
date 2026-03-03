import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.integrate import cumulative_trapezoid
from typing import Optional, Tuple
from .base import BaseIcDetector


class IcdFang(BaseIcDetector):
    """
    Fang / Del Din Initial and Final Contact detector with McCamley Laterality.

    Description:
    ------------
    This algorithm detects Initial Contacts (IC) and Final Contacts (FC) using
    head-worn or lower-trunk vertical acceleration. It employs a signal processing
    pipeline involving integration and Continuous Wavelet Transform (CWT) to
    identify gait events.

    Additionally, this implementation estimates the laterality (Left/Right) of the
    Initial Contacts using the vertical angular velocity, as proposed by McCamley
    et al. (2012). This specific laterality check was part of the original
    McCamley/Del Din frameworks but was not explicitly detailed in the Fang (2024)
    validity study.

    Algorithm Pipeline:
    1. Linear detrending of the vertical acceleration.
    2. Low-pass Butterworth filtering (default 3.2 Hz, 10th order).
    3. Signal integration using the cumulative trapezoidal rule.
    4. First differentiation via Gaussian CWT (gaus1) to identify ICs (local minima).
    5. Second differentiation via Gaussian CWT to identify FCs (local maxima).
    6. (Laterality) Low-pass filtering of vertical angular velocity (2 Hz, 4th order).
       Sign of filtered signal at IC determines side (Pos=Left, Neg=Right).

    Author: Paolo Tasca
    Date: 18 February 2026

    References:
    -----------
    [1] Fang et al. (2024). Examining the validity of smart glasses in measuring
        spatiotemporal parameters of gait among people with Parkinson’s disease.
        Gait & Posture, 113, 139-144.
        https://doi.org/10.1016/j.gaitpost.2024.06.001

    [2] S. Del Din, A. Godfrey and L. Rochester (2016). Validation of an Accelerometer
        to Quantify a Comprehensive Battery of Gait Characteristics...
        IEEE Journal of Biomedical and Health Informatics, 20(3), 838-847.
        doi: 10.1109/JBHI.2015.2419317

    [3] John McCamley et al. (2012). An enhanced estimate of initial contact and
        final contact instants of time using lower trunk inertial sensor data.
        Gait & Posture, 36(2), 316-318.
        https://doi.org/10.1016/j.gaitpost.2012.02.019
    """

    def __init__(
            self,
            lowpass_hz: float = 3.0, #optimized: 3, #original: 3.2,
            butter_order: int = 10,
            min_step_duration_s: float = 0.44,
            wavelet_type: str = "gaus1", #optimized: "gaus1", #original: "gaus1",
            lat_lp_hz: float = 2.0,
            lat_lp_order: int = 4
    ) -> None:
        """
        Initialize the detector with default hyperparameters.

        Args:
            lowpass_hz: Cutoff frequency for the main acceleration low-pass filter.
            butter_order: Order of the main acceleration Butterworth filter.
            min_step_duration_s: Minimum time between consecutive gait events.
            wavelet_type: Type of wavelet used for differentiation.
            lat_lp_hz: Cutoff frequency for the laterality gyro filter (McCamley).
            lat_lp_order: Order of the laterality gyro filter.
        """
        super().__init__()
        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order
        self.min_step_duration_s = min_step_duration_s
        self.wavelet_type = wavelet_type

        # Laterality parameters (McCamley)
        self.lat_lp_hz = lat_lp_hz
        self.lat_lp_order = lat_lp_order

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
            gyr_column: str = "gyr_is",  # Vertical axis angular velocity
            plot_debug: bool = True,
            **kwargs
    ) -> "IcdFang":
        """
        Detect Initial and Final Contacts from acceleration data and determine laterality.

        Args:
            data: DataFrame containing the gait signals.
            sampling_rate_hz: Sampling frequency of the data.
            acc_column: The column name representing vertical (Superior-Inferior) acceleration.
            gyr_column: The column name representing vertical (Superior-Inferior) angular velocity.
            plot_debug: If True, generates a plot of the signal processing steps.

        Returns:
            self: The instance with populated ic_list_, ic_side_, fc_list_, and fc_side_.
        """
        acc = data[acc_column].to_numpy()
        gyr = -data[gyr_column].to_numpy() if gyr_column in data.columns else np.zeros_like(acc)

        fs = sampling_rate_hz
        dt = 1.0 / fs

        # --- 1. Signal Processing for Timing (Fang/Del Din) ---

        # Linear Detrending
        acc_dt = detrend(acc, type="linear")

        # Low-pass Filtering (Acceleration)
        nyq = 0.5 * fs
        norm_cut = self.lowpass_hz / nyq
        b, a = butter(self.butter_order, norm_cut, btype="low")
        acc_lp = filtfilt(b, a, acc_dt)

        # Integration
        acc_int = cumulative_trapezoid(acc_lp, initial=0) * dt

        # Scale selection (IC)
        spectrum = np.abs(np.fft.rfft(acc_lp))
        freqs = np.fft.rfftfreq(len(acc_lp), d=dt)
        # Skip DC component (index 0)
        dom_freq1 = freqs[np.argmax(spectrum[1:]) + 1]

        fc_wave = pywt.central_frequency(self.wavelet_type)
        scale1 = fc_wave / (dom_freq1 * dt)

        # First Derivative (IC detection) - CWT
        cwt_coeffs, _ = pywt.cwt(acc_int, scales=[scale1], wavelet=self.wavelet_type, sampling_period=dt)
        d1 = cwt_coeffs[0].squeeze()
        d1 -= d1.mean()

        min_dist = int(self.min_step_duration_s * fs)
        ic_idx, _ = find_peaks(-d1, distance=min_dist)

        # Scale selection (FC) - based on d1 spectrum
        spectrum_d1 = np.abs(np.fft.rfft(d1))
        dom_freq2 = freqs[np.argmax(spectrum_d1[1:]) + 1]
        scale2 = fc_wave / (dom_freq2 * dt)

        # Second Derivative (FC detection) - CWT
        cwt_coeffs2, _ = pywt.cwt(d1, scales=[scale2], wavelet=self.wavelet_type, sampling_period=dt)
        d2 = cwt_coeffs2[0].squeeze()
        d2 -= d2.mean()

        fc_idx, _ = find_peaks(d2, distance=min_dist)

        # --- 2. Laterality Detection (McCamley) ---

        # Filter vertical gyro (2 Hz, 4th order)
        norm_cut_lat = self.lat_lp_hz / nyq
        b_lat, a_lat = butter(self.lat_lp_order, norm_cut_lat, btype="low")
        gyr_lp = filtfilt(b_lat, a_lat, gyr)

        # Assign side for ICs
        # McCamley (2012): "Positive or negative sign... indicates left and right ICs"
        ic_sides = []
        for idx in ic_idx:
            val = gyr_lp[idx]
            if val > 0:
                ic_sides.append('L')
            else:
                ic_sides.append('R')

        # --- Standardized Output Formatting ---

        # Store ICs
        self.ic_list_ = pd.DataFrame({"ic": ic_idx}, index=pd.RangeIndex(len(ic_idx), name="step_id"))
        self.ic_side_ = pd.Series(ic_sides, index=self.ic_list_.index, name="side")

        # Store FCs
        # Note: McCamley logic is explicitly for ICs. FC side is left as Unknown ('U')
        # unless a specific logic for FC-to-foot mapping is added.
        # Added logic to determine FC laterality (opposite laterality of preceding IC)
        self.fc_list_ = pd.DataFrame({"fc": fc_idx}, index=pd.RangeIndex(len(fc_idx), name="step_id"))

        # --- NEW: assign FC side as opposite of preceding IC, with one-to-one constraint ---
        fc_sides = []
        used_fc = set()

        ic_idx_sorted = np.asarray(ic_idx, dtype=int)
        ic_sides_arr = np.asarray(ic_sides, dtype=str)

        for f in np.asarray(fc_idx, dtype=int):
            # find the last IC strictly before this FC
            prev_ic_pos = np.searchsorted(ic_idx_sorted, f, side="left") - 1

            if prev_ic_pos < 0:
                # no preceding IC
                fc_sides.append("U")
                continue

            if f in used_fc:
                # already associated to another IC -> unknown
                fc_sides.append("U")
                continue

            ic_side = ic_sides_arr[prev_ic_pos]
            fc_side = "R" if ic_side == "L" else "L"

            fc_sides.append(fc_side)
            used_fc.add(f)

        self.fc_side_ = pd.Series(fc_sides, index=self.fc_list_.index, name="side")
        if plot_debug:
            self._plot_debug(acc, acc_lp, acc_int, d1, d2, ic_idx, fc_idx, gyr_lp, ic_sides)

        return self

    def _plot_debug(self, acc, acc_lp, acc_int, d1, d2, ic_idx, fc_idx, gyr_lp, ic_sides):
        """Internal helper for debugging signal processing steps."""
        fig, axs = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

        axs[0].plot(acc, label='Raw Acc')
        axs[0].plot(acc_lp, label='Filtered Acc')
        axs[0].set_title('Vertical Acceleration')
        axs[0].legend(loc='upper right')

        axs[1].plot(acc_int)
        axs[1].set_title('Integrated Acceleration')

        axs[2].plot(d1)
        axs[2].plot(ic_idx, d1[ic_idx], 'or', label='IC')
        axs[2].set_title('CWT 1st Derivative (IC detection)')

        axs[3].plot(d2)
        axs[3].plot(fc_idx, d2[fc_idx], 'ob', label='FC')
        axs[3].set_title('CWT 2nd Derivative (FC detection)')

        # Laterality Plot
        axs[4].plot(gyr_lp, color='purple', label='Filt. Gyro (2Hz)')
        axs[4].axhline(0, color='k', linestyle='--', linewidth=0.8)

        # Plot markers for L/R
        # Filter indices by side for plotting
        left_ics = [idx for i, idx in enumerate(ic_idx) if ic_sides[i] == 'L']
        right_ics = [idx for i, idx in enumerate(ic_idx) if ic_sides[i] == 'R']

        if left_ics:
            axs[4].plot(left_ics, gyr_lp[left_ics], '^g', markersize=8, label='Left IC')
        if right_ics:
            axs[4].plot(right_ics, gyr_lp[right_ics], 'vm', markersize=8, label='Right IC')

        axs[4].set_title('Laterality (Vertical Angular Velocity)')
        axs[4].legend()

        axs[5].plot(acc_lp, color='gray', alpha=0.5)
        for idx, side in zip(ic_idx, ic_sides):
            color = 'g' if side == 'L' else 'm'
            axs[5].axvline(idx, color=color, linestyle='--', alpha=0.6)
            axs[5].text(idx, np.max(acc_lp), side, color=color, ha='center')
        axs[5].set_title('Final Output: ICs with Side')

        plt.tight_layout()
        plt.show()