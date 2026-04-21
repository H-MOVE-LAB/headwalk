from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema, firwin, kaiserord, filtfilt
from pyts.decomposition import SingularSpectrumAnalysis
from .base import BaseGeDetector


class GedSeifer(BaseGeDetector):
    """
    Initial Contact (IC) and Final Contact (FC) detection based on Seifer et al. (2023)[cite: 8].

    This is an improved version of the Diao et al. (2020) algorithm.

    Key Improvements (Seifer et al.):
    ---------------------------------
    1. Laterality Determination: Instead of using the trend-removed ML signal (which contains
       higher frequency components), this method uses the FIRST dominant oscillation
       (SSA Component 1) of the ML axis to determine the side (Left vs Right)[cite: 220].
    2. Robustness: This approach reduces misclassification of steps caused by signal noise
       or irregular gait patterns[cite: 29].

    Algorithm Steps:
    ----------------
    1. Low-pass filtering (FIR Kaiser).
    2. Singular Spectrum Analysis (SSA) decomposition.
    3. IC Detection: Minima of the dominant SSA component (Index 1) of the SI axis[cite: 251].
    4. Laterality: Determined by the slope of the dominant SSA component (Index 1) of the ML axis[cite: 252].
    5. FC Detection: Local extrema of the detrended ML signal (SSA components 1+2+3)[cite: 254].
    """

    def __init__(
            self,
            window_length: int | None = None,
            cutoff_hz: float = 3.4, #optimized: 3.4, #original: 3.0,
            ripple_db: float = 60.0,
    ) -> None:
        self.window_length = window_length
        self.cutoff_hz = cutoff_hz
        self.ripple_db = ripple_db

        self.sampling_rate_hz: float | None = None
        self.ssa = None
        self.filter_taps = None

        # Results storage
        self.ic_list_ = None
        self.fc_list_ = None
        self.ic_side_ = None
        self.fc_side_ = None

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_si_col: str = "acc_is",
            acc_ml_col: str = "acc_ml",
            plot_debug: bool = True,
            **_,
    ):
        """
        Execute the Seifer gait event detection pipeline.
        """
        self.sampling_rate_hz = sampling_rate_hz
        if self.window_length is None:
            self.window_length = int(self.sampling_rate_hz)

        # Initialize helper objects
        self._initialize_filter()
        self._initialize_ssa()

        # 1. Pre-processing
        # Get raw data
        acc_si_raw = data[acc_si_col].to_numpy()
        acc_ml_raw = data[acc_ml_col].to_numpy()

        # Apply Low-Pass Filter
        # Note: Inverting signals to match Diao/EarGait conventions where peaks/valleys align
        acc_si_filt = -self._filter_signal(acc_si_raw)
        acc_ml_filt = -self._filter_signal(acc_ml_raw)

        # 2. SSA Decomposition
        # We reshape to (1, n_samples) because pyts expects 2D input for a single time series
        acc_si_ssa = self.ssa.fit_transform(acc_si_filt.reshape(1, -1))
        acc_ml_ssa = self.ssa.fit_transform(acc_ml_filt.reshape(1, -1))

        # 3. Component Extraction
        # SI Axis: Use Component 1 (Dominant Oscillation) for IC detection [cite: 251]
        acc_si_dom = acc_si_ssa[0, 1, :]

        # ML Axis: Use Component 1 (Dominant Oscillation) for SIDE determination [cite: 252]
        acc_ml_dom = acc_ml_ssa[0, 1, :]

        # ML Axis: Use Components 1+2+3 (Detrended) for FC/TC detection [cite: 254]
        # This preserves more signal detail than just the dominant oscillation
        acc_ml_det = acc_ml_ssa[0, 1, :] + acc_ml_ssa[0, 2, :] + acc_ml_ssa[0, 3, :]

        # 4. Event Detection
        # Detect Initial Contacts and assign side (L/R)
        ic_idx, ic_side = self._detect_ic_with_side(acc_si_dom, acc_ml_dom)

        # Detect Final Contacts based on ICs
        fc_idx, fc_side = self._detect_fc_with_side(acc_ml_det, ic_idx, ic_side)

        # 5. Store Results
        # Adjust indices by the dataframe offset
        offset = data.index[0]
        self._store_results(ic_idx + offset, ic_side, fc_idx + offset, fc_side)

        # 6. Optional Debug Plotting
        if plot_debug:
            self._plot_debug(-acc_si_raw, -acc_ml_raw, acc_si_dom, acc_ml_det)

        return self

    def _detect_ic_with_side(self, si_dom: np.ndarray, ml_dom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect Initial Contacts and determine laterality using Seifer's logic.

        Parameters
        ----------
        si_dom : np.ndarray
            Dominant SSA component of SI axis (for peak detection).
        ml_dom : np.ndarray
            Dominant SSA component of ML axis (for side determination).
        """
        # Find peaks in the inverted SI dominant signal (corresponds to minima in original)
        peaks, _ = find_peaks(-si_dom, height=0.2, distance=int(0.2 * self.sampling_rate_hz))

        sides = []
        for p in peaks:
            # Seifer Improvement: Determine side based on the slope of the
            # FIRST dominant oscillation of ML axis.

            # Check slope: (value at peak) vs (value at peak + 1)
            # Boundary check included for the last sample
            next_idx = min(p + 1, len(ml_dom) - 1)
            slope = ml_dom[next_idx] - ml_dom[p]

            # Rule: Increasing ML -> Right | Decreasing ML -> Left
            sides.append("R" if slope > 0 else "L")

        return peaks, np.array(sides)

    def _detect_fc_with_side(self, ml_det: np.ndarray, ic_idx: np.ndarray, ic_side: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Detect Final Contacts (TC) using cross-lateral logic on detrended ML signal.
        """
        # Find local extrema in the ML detrended signal
        # [0] is used to extract the array from the tuple returned by argrelextrema
        maxima = argrelextrema(ml_det, np.greater, order=2)[0]
        minima = argrelextrema(ml_det, np.less, order=2)[0]

        fc_idx = []
        fc_side = []

        for ic, side in zip(ic_idx, ic_side):
            # Seifer/EarGait logic:
            # If current Step is Right (Side=R), we look for the next event which corresponds
            # to the contralateral side (Left FC).

            candidates = []
            if side == "R":
                # Look for maxima after the IC
                candidates = maxima[maxima > ic]
            else:
                # Look for minima after the IC
                candidates = minima[minima > ic]

            if len(candidates) > 0:
                # The first extrema after IC is the TC
                fc_idx.append(candidates[0])

                # Cross-lateral assignment:
                # Right IC -> leads to Left FC
                # Left IC -> leads to Right FC
                fc_side.append("L" if side == "R" else "R")

        return np.array(fc_idx), np.array(fc_side)

    def _initialize_filter(self):
        """Initialize the Low-Pass FIR filter using Kaiser window."""
        nyq = self.sampling_rate_hz / 2.0
        # Calculate filter order and beta parameter
        n, beta = kaiserord(self.ripple_db, 2.0 / nyq)
        self.filter_taps = firwin(n, self.cutoff_hz / nyq, window=("kaiser", beta))

    def _filter_signal(self, x: np.ndarray) -> np.ndarray:
        """Apply zero-phase filtering."""
        return filtfilt(self.filter_taps, [1.0], x)

    def _initialize_ssa(self):
        """
        Initialize SSA with groups specific to Seifer/EarGait implementation.
        Group [1] is isolated for robust laterality detection.
        """
        self.ssa = SingularSpectrumAnalysis(
            window_size=int(self.window_length),
            groups=[
                [0],  # Trend
                [1],  # Dominant Oscillation (Used for IC & Side) [cite: 251-252]
                [2],  # Second component (Detail)
                [3],  # Third component (Detail)
                # Remaining components are treated as noise/high freq
            ]
        )

    def _store_results(self, ic, ic_side, fc, fc_side):
        """Sort and store events in pandas DataFrames."""
        # Ensure events are sorted by time (index)
        idx_i = np.argsort(ic)
        idx_f = np.argsort(fc)

        self.ic_list_ = pd.DataFrame(
            {"ic": ic[idx_i], "side": ic_side[idx_i]},
            index=pd.RangeIndex(len(ic), name="step_id")
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": fc[idx_f], "side": fc_side[idx_f]},
            index=pd.RangeIndex(len(fc), name="step_id")
        )

        # Keep Series for backward compatibility with base class
        self.ic_side_ = self.ic_list_["side"]
        self.fc_side_ = self.fc_list_["side"]

    def _plot_debug(self, si_raw, ml_raw, si_dom, ml_det):
        """
        Plot debug visualization showing raw signals and SSA components
        used for detection.
        """

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: SI Axis (IC Detection)
        ax1.plot(si_raw, color='gray', alpha=0.3, label='Raw SI')
        ax1.plot(si_dom, color='blue', label='SSA Dom (IC Search)')

        for i, s in zip(self.ic_list_['ic'], self.ic_side_):
            color = 'red' if s == 'R' else 'orange'
            label = f'IC {s}' if f'IC {s}' not in ax1.get_legend_handles_labels()[1] else ""
            ax1.axvline(i, color=color, linestyle='--', label=label)

        ax1.set_title("IC Detection (SI Axis)")
        ax1.set_ylabel("Acceleration ($m/s^2$)")
        ax1.legend(loc='upper right', fontsize='small')

        # Plot 2: ML Axis (FC Detection)
        ax2.plot(ml_raw, color='gray', alpha=0.3, label='Raw ML')
        ax2.plot(ml_det, color='green', label='SSA Detrended (FC Search)')

        for i, s in zip(self.fc_list_['fc'], self.fc_side_):
            color = 'darkred' if s == 'R' else 'gold'
            label = f'TC {s}' if f'TC {s}' not in ax2.get_legend_handles_labels()[1] else ""
            ax2.axvline(i, color=color, linestyle=':', label=label)

        ax2.set_title("FC/TC Detection (ML Axis)")
        ax2.set_ylabel("Acceleration ($m/s^2$)")
        ax2.set_xlabel("Samples")
        ax2.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        plt.show()