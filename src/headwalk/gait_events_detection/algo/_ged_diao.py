from typing import Any
import numpy as np
import pandas as pd

from scipy.signal import (
    find_peaks,
    argrelextrema,
    firwin,
    kaiserord,
    lfilter,
)
from pyts.decomposition import SingularSpectrumAnalysis

from .base import BaseGeDetector


class GedDiao(BaseGeDetector):
    """
    Initial Contact detection based on Diao et al., 2020.

    IC:
        minima of dominant SSA component of SI acceleration
    FC:
        mapped from toe-off (TC) events detected on ML acceleration
    """

    def __init__(
        self,
        window_length: int | None = None,
        cutoff_hz: float = 3.0,
        ripple_db: float = 60.0,
    ) -> None:
        self.window_length = window_length
        self.cutoff_hz = cutoff_hz
        self.ripple_db = ripple_db

        self.sampling_rate_hz: float | None = None
        self.ssa = None
        self.filter_order_n = None
        self.filter_taps = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_si_col: str = "acc_is",
            acc_ml_col: str = "acc_ml",
            **_,
    ):
        self.sampling_rate_hz = sampling_rate_hz

        """
        Detect IC and FC events using Diao algorithm.
        """

        if self.window_length is None:
            self.window_length = int(self.sampling_rate_hz)

        assert (
            self.window_length == int(self.sampling_rate_hz)
        ), "window_length must equal sampling_rate_hz (1 s)"

        self._initialize_filter()
        self._initialize_ssa()

        # --------------------------------------------------
        # FILTERING
        # --------------------------------------------------
        acc_si = -self._filter_signal(data[acc_si_col].to_numpy())
        acc_ml = self._filter_signal(data[acc_ml_col].to_numpy())   # sign flip as in eargait

        assert (
            acc_si.shape[0] > 3 * self.sampling_rate_hz
        ), "Walking bout must be longer than 3 seconds."

        # --------------------------------------------------
        # SSA DECOMPOSITION
        # --------------------------------------------------
        acc_si_ssa = self.ssa.fit_transform(acc_si.reshape(1, -1))
        acc_ml_ssa = self.ssa.fit_transform(acc_ml.reshape(1, -1))

        acc_si_dom = acc_si_ssa[0,1,:]
        acc_ml_wo_trend = acc_ml_ssa[0,1,:] + acc_ml_ssa[0,2,:]

        # --------------------------------------------------
        # EVENT DETECTION
        # --------------------------------------------------
        ic_idx = self._detect_ic(acc_si_dom)
        fc_idx = self._detect_fc(acc_ml_wo_trend, ic_idx)

        # shift to absolute indices
        offset = data.index[0]
        ic_idx = ic_idx + offset
        fc_idx = fc_idx + offset

        # --------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------
        self.ic_list_ = pd.DataFrame(
            {"ic": ic_idx},
            index=pd.RangeIndex(len(ic_idx), name="step_id"),
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": fc_idx},
            index=pd.RangeIndex(len(fc_idx), name="step_id"),
        )

        return self

    # ------------------------------------------------------------------
    # IC / FC DETECTION
    # ------------------------------------------------------------------
    def _detect_ic(self, acc_si_dominant: np.ndarray) -> np.ndarray:
        """
        IC = minima of dominant SSA component on SI axis
        """
        peaks, _ = find_peaks(
            -acc_si_dominant,
            height=0.2,
            distance=int(0.2 * self.sampling_rate_hz),
        )
        return peaks.astype(int)

    def _detect_fc(
        self,
        acc_ml_wo_trend: np.ndarray,
        ic_idx: np.ndarray,
    ) -> np.ndarray:
        """
        FC mapped from TC (toe-off) detection in Diao.
        """

        mins = argrelextrema(acc_ml_wo_trend, np.less, order=2)[0]
        maxs = argrelextrema(acc_ml_wo_trend, np.greater, order=2)[0]

        fc = []

        for ic in ic_idx:
            # first extremum after IC
            candidates = np.concatenate([mins, maxs])
            candidates = candidates[candidates > ic]
            if len(candidates) > 0:
                fc.append(int(candidates.min()))

        return np.unique(fc)

    # ------------------------------------------------------------------
    # FILTERING
    # ------------------------------------------------------------------
    def _initialize_filter(self):
        nyq = self.sampling_rate_hz / 2.0
        width = 2.0 / nyq

        self.filter_order_n, beta = kaiserord(self.ripple_db, width)
        self.filter_taps = firwin(
            self.filter_order_n,
            self.cutoff_hz / nyq,
            window=("kaiser", beta),
        )

    def _filter_signal(self, x: np.ndarray) -> np.ndarray:
        delay = int(0.5 * (self.filter_order_n - 1))
        y = lfilter(self.filter_taps, 1.0, x)
        return y[delay:]

    # ------------------------------------------------------------------
    # SSA
    # ------------------------------------------------------------------
    def _initialize_ssa(self):
        self.ssa = SingularSpectrumAnalysis(
            window_size=int(self.window_length),
            groups=[
                [0],  # trend
                [1],  # dominant oscillation
                np.arange(2, self.window_length),
            ],
        )
