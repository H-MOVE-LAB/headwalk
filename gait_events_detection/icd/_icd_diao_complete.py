from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, argrelextrema, firwin, kaiserord, filtfilt
from scipy.linalg import eigh
from pyts.decomposition import SingularSpectrumAnalysis
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdDiaoComplete(BaseIcDetector):
    """
    Initial Contact (IC) and Final Contact (FC) detection based on Diao et al. (2020).

    Pipeline (same logic as provided, just standardized to the IcdTcn interface):
      1) FIR low-pass filtering (Kaiser window design)
      2) Singular Spectrum Analysis (SSA) decomposition
         - SI: dominant component from SSA group [1]
         - ML: detrended from SSA groups [1] + [2]
      3) Initial IC detection on SI-dom + side assignment using ML slope  (anchors)
      4) Optional iterative mean filtering (IMF) refinement with anchoring
      5) FC detection using alternation rule on ML detrended extrema
      6) Store outputs:
          self.ic_list_ : DataFrame with column "ic" (sample indices)
          self.fc_list_ : DataFrame with column "fc" (sample indices)
          self.ic_side_ : Series of "L"/"R"
          self.fc_side_ : Series of "L"/"R"

    Notes:
      - Indices are returned in the same sample domain as the input DataFrame index.
      - This class does NOT change the original algorithm logic, only the structure / API.
    """

    def __init__(
        self,
        window_length: int | None = None,
        cutoff_hz: float = 3.2, #optimized: 3.2, best working (empirical): 5.0, #original: 3.0,
        ripple_db: float = 60.0,
        iterative_filtering: bool = True,
    ) -> None:
        self.window_length = window_length
        self.cutoff_hz = float(cutoff_hz)
        self.ripple_db = float(ripple_db)
        self.iterative_filtering = bool(iterative_filtering)

        self.sampling_rate_hz: float | None = None
        self.ssa: SingularSpectrumAnalysis | None = None
        self.filter_taps: np.ndarray | None = None

        # for debug / inspection
        self.ic_pre_refinement_: np.ndarray | None = None

        # results (set by detect)
        self.ic_list_: pd.DataFrame | None = None
        self.fc_list_: pd.DataFrame | None = None
        self.ic_side_: pd.Series | None = None
        self.fc_side_: pd.Series | None = None

    # ------------------------------------------------------------------
    # Public API (standardized)
    # ------------------------------------------------------------------
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_columns: list[str] = ["acc_is", "acc_ml", "acc_ap"],
        gyr_columns: list[str] | None = None,
        plot_debug: bool = True,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Run Diao IC/FC detection.

        Parameters
        ----------
        data:
            Input dataframe with at least acc_columns[0] (SI/IS) and acc_columns[1] (ML).
        sampling_rate_hz:
            Sampling rate of the input data.
        acc_columns:
            Uses acc_columns[0] as SI/IS axis and acc_columns[1] as ML axis.
            (Defaults match your previous implementation.)
        gyr_columns:
            Ignored (kept only for API compatibility with other detectors).
        plot_debug:
            If True, show intermediate and final plots.

        Returns
        -------
        Self (fluent API)
        """
        self.data = data
        self.sampling_rate_hz = float(sampling_rate_hz)

        if self.window_length is None:
            # same behavior: default = 1 second window
            self.window_length = int(self.sampling_rate_hz)

        self._initialize_filter()
        self._initialize_ssa()

        # --- 1) raw signals ---
        acc_si_raw = data[acc_columns[0]].to_numpy()
        acc_ml_raw = data[acc_columns[1]].to_numpy()

        # --- 2) preliminary filtering (same sign convention as your code) ---
        acc_si_filt = -self._filter_signal(acc_si_raw)
        acc_ml_filt = -self._filter_signal(acc_ml_raw)

        # --- 3) SSA decomposition (same component selection) ---
        si_ssa = self.ssa.fit_transform(acc_si_filt.reshape(1, -1))
        ml_ssa = self.ssa.fit_transform(acc_ml_filt.reshape(1, -1))

        acc_si_dom = si_ssa[0, 1, :]                # group [1]
        acc_ml_det = ml_ssa[0, 1, :] + ml_ssa[0, 2, :]  # groups [1] + [2]

        # --- 4) Initial IC detection (anchors) ---
        ic_idx, ic_side = self._detect_ic_with_side(acc_si_dom, acc_ml_det)
        self.ic_pre_refinement_ = ic_idx.copy()

        # --- 5) Optional IMF refinement (anchored) ---
        if self.iterative_filtering:
            ic_idx, ic_side = self._apply_full_imf(acc_si_filt, acc_ml_det, ic_idx, ic_side)

        # --- 6) FC detection ---
        fc_idx, fc_side = self._detect_fc_with_side(acc_ml_det, ic_idx, ic_side)

        # --- 7) Store results in dataframe index domain ---
        offset = int(data.index[0]) if len(data.index) > 0 else 0
        self._store_results(ic_idx + offset, ic_side, fc_idx + offset, fc_side)

        if plot_debug:
            self._plot_debug(
                si_raw=acc_si_raw,
                si_filt=acc_si_filt,
                ml_det=acc_ml_det,
                ic_pre=(self.ic_pre_refinement_ + offset) if self.ic_pre_refinement_ is not None else None,
            )

        return self

    # ------------------------------------------------------------------
    # Core algorithm pieces (unchanged logic, just cleaned)
    # ------------------------------------------------------------------
    def _initialize_filter(self) -> None:
        assert self.sampling_rate_hz is not None
        nyq = self.sampling_rate_hz / 2.0
        n, beta = kaiserord(self.ripple_db, 2.0 / nyq)
        # Keep same FIR design choice
        self.filter_taps = firwin(n, self.cutoff_hz / nyq, window=("kaiser", beta))

    def _filter_signal(self, x: np.ndarray) -> np.ndarray:
        assert self.filter_taps is not None
        return filtfilt(self.filter_taps, [1.0], x)

    def _initialize_ssa(self) -> None:
        assert self.window_length is not None
        # Same grouping
        self.ssa = SingularSpectrumAnalysis(window_size=int(self.window_length), groups=[[0], [1], [2]])

    def _detect_ic_with_side(self, si_signal: np.ndarray, ml_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """IC peaks from -SI; side from ML slope at peak."""
        assert self.sampling_rate_hz is not None
        peaks, _ = find_peaks(
            -si_signal,
            height=0.2,
            distance=int(0.2 * self.sampling_rate_hz),
        )
        sides = []
        for p in peaks:
            next_p = min(p + 1, len(ml_signal) - 1)
            slope = ml_signal[next_p] - ml_signal[p]
            sides.append("R" if slope > 0 else "L")
        return peaks, np.asarray(sides, dtype=str)

    def _detect_fc_with_side(
        self,
        ml_det: np.ndarray,
        ic_idx: np.ndarray,
        ic_side: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """FC detection based on alternating IC side and ML detrended extrema."""
        maxima = argrelextrema(ml_det, np.greater, order=2)[0]
        minima = argrelextrema(ml_det, np.less, order=2)[0]

        fc_idx: list[int] = []
        fc_side: list[str] = []

        for ic, side in zip(ic_idx, ic_side):
            cand = maxima[maxima > ic] if side == "R" else minima[minima > ic]
            if len(cand) > 0:
                fc_idx.append(int(cand[0]))
                fc_side.append("L" if side == "R" else "R")

        return np.asarray(fc_idx, dtype=int), np.asarray(fc_side, dtype=str)

    # ---------------- IMF refinement (same logic, formatted) ----------------
    def _apply_full_imf(
        self,
        acc_si: np.ndarray,
        acc_ml_det: np.ndarray,
        anchor_idx: np.ndarray,
        anchor_side: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterative mean filtering with anchoring (unchanged logic).
        """
        assert self.sampling_rate_hz is not None

        rhc_start = anchor_idx[anchor_side == "R"]
        lhc_start = anchor_idx[anchor_side == "L"]

        try:
            prev_metrics = self._calculate_imf_metrics(acc_ml_det, rhc_start, lhc_start)
        except Exception:
            return anchor_idx, anchor_side

        d = 3
        max_iter = 15
        stop_threshold = 0.001

        final_ic_idx = anchor_idx
        final_ic_side = anchor_side

        tolerance = 0.2 * self.sampling_rate_hz

        for _ in range(max_iter):
            # 1) smooth SI
            s_si = (
                pd.Series(acc_si)
                .rolling(window=d, center=True)
                .mean()
                .ffill()
                .bfill()
                .to_numpy()
            )

            # 2) detect candidates on smoothed SI
            current_idx, current_side = self._detect_ic_with_side(s_si, acc_ml_det)

            # 3) anchoring: match each anchor to nearest candidate within tolerance
            rhc: list[int] = []
            lhc: list[int] = []
            temp_idx: list[int] = []
            temp_side: list[str] = []

            for anchor, side in zip(anchor_idx, anchor_side):
                diffs = np.abs(current_idx - anchor)
                if len(diffs) > 0 and np.min(diffs) < tolerance:
                    match_pos = int(np.argmin(diffs))
                    matched_idx = int(current_idx[match_pos])
                    matched_side = str(current_side[match_pos])

                    temp_idx.append(matched_idx)
                    temp_side.append(matched_side)
                    if matched_side == "L":
                        lhc.append(matched_idx)
                    else:
                        rhc.append(matched_idx)

            if len(rhc) < 3 or len(lhc) < 3:
                break

            # 4) convergence check using gait-symmetry metrics
            try:
                metrics = self._calculate_imf_metrics(acc_ml_det, np.asarray(rhc), np.asarray(lhc))
                delta = np.abs(metrics - prev_metrics) / prev_metrics
                if np.all(delta < stop_threshold):
                    break

                final_ic_idx = np.asarray(temp_idx, dtype=int)
                final_ic_side = np.asarray(temp_side, dtype=str)
                prev_metrics = metrics
                d += 2
            except Exception:
                break

        return final_ic_idx, final_ic_side

    def _calculate_imf_metrics(self, acc_ml: np.ndarray, rhc: np.ndarray, lhc: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues in phase space (Gait Symmetry), unchanged logic.
        """

        def get_eigen_v(events: np.ndarray, is_rhc: bool = True) -> tuple[float, float]:
            m_list = []

            for i in range(len(events) - 2):
                try:
                    if is_rhc:
                        p = min(np.abs(lhc[i] - rhc[i]), np.abs(lhc[i + 1] - rhc[i + 1]))
                        tau, eps = (lhc[i] - rhc[i]) / p, (lhc[i + 1] - rhc[i + 1]) / p
                    else:
                        p = min(np.abs(rhc[i + 1] - lhc[i]), np.abs(rhc[i + 2] - lhc[i + 1]))
                        tau, eps = (rhc[i + 1] - lhc[i]) / p, (rhc[i + 2] - lhc[i + 1]) / p

                    k = int(p)
                    if k < 2:
                        continue

                    row1 = [acc_ml[int(round(events[i] + j * tau))] for j in range(1, k + 1)]
                    row2 = [-acc_ml[int(round(events[i + 1] + j * eps))] for j in range(1, k + 1)]
                    m_list.append(np.array([row1, row2]))
                except IndexError:
                    continue

            if not m_list:
                raise ValueError("Incongruent left/right events for IMF metrics")

            cov = np.cov(np.hstack(m_list))
            vals, vecs = eigh(cov)
            idx = np.argsort(vals)[::-1]
            u1 = vecs[:, idx[0]]
            v = np.abs(np.dot(u1.T, np.array([1, -1]) / np.sqrt(2)))
            return float(vals[idx[0]]), float(v)

        l1_m, v1 = get_eigen_v(rhc, True)
        l1_n, v2 = get_eigen_v(lhc, False)
        return np.asarray([l1_m, l1_n, v1, v2], dtype=float)

    # ------------------------------------------------------------------
    # Output formatting (same contract as IcdTcn)
    # ------------------------------------------------------------------
    def _store_results(self, ic: np.ndarray, ic_side: np.ndarray, fc: np.ndarray, fc_side: np.ndarray) -> None:
        # sort and store
        idx_i = np.argsort(ic) if len(ic) else np.array([], dtype=int)
        idx_f = np.argsort(fc) if len(fc) else np.array([], dtype=int)

        ic_sorted = ic[idx_i] if len(ic) else ic
        fc_sorted = fc[idx_f] if len(fc) else fc
        ic_side_sorted = ic_side[idx_i] if len(ic_side) else ic_side
        fc_side_sorted = fc_side[idx_f] if len(fc_side) else fc_side

        self.ic_list_ = pd.DataFrame({"ic": ic_sorted}, index=pd.RangeIndex(len(ic_sorted), name="step_id"))
        self.fc_list_ = pd.DataFrame({"fc": fc_sorted}, index=pd.RangeIndex(len(fc_sorted), name="step_id"))
        self.ic_side_ = pd.Series(ic_side_sorted, index=self.ic_list_.index, name="side")
        self.fc_side_ = pd.Series(fc_side_sorted, index=self.fc_list_.index, name="side")

    # ------------------------------------------------------------------
    # Debug plotting (standardized to IcdTcn style)
    # ------------------------------------------------------------------
    def _plot_debug(
        self,
        si_raw: np.ndarray,
        si_filt: np.ndarray,
        ml_det: np.ndarray,
        *,
        ic_pre: np.ndarray | None = None,
    ) -> None:
        """
        Debug plot similar in spirit to IcdTcn:
          - show raw vs filtered SI
          - show anchors (pre-refinement) and refined ICs
          - show ML detrended and FCs
        """
        assert self.ic_list_ is not None and self.fc_list_ is not None
        assert self.ic_side_ is not None and self.fc_side_ is not None

        t = np.arange(len(si_raw)) / float(self.sampling_rate_hz)

        fig = plt.figure(figsize=(14, 12))
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)

        model_type = self.__class__.__name__.replace("Icd", "")
        fig.suptitle(f"Debug Pipeline: {model_type}", fontsize=16)

        # 1) Raw SI
        ax1.plot(t, -si_raw, color="gray", alpha=0.25, label="Raw SI (negated)")
        ax1.set_title("Step 1: Raw SI (negated, as in Diao)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # 2) Filtered SI + Anchors + Refined ICs
        ax2.plot(t, si_filt, color="blue", alpha=0.7, label="Filtered SI (negated)")
        if ic_pre is not None and len(ic_pre) > 0:
            ic_pre = np.asarray(ic_pre, dtype=int)
            ic_pre = ic_pre[(ic_pre >= 0) & (ic_pre < len(si_filt))]
            ax2.scatter(
                ic_pre / float(self.sampling_rate_hz),
                si_filt[ic_pre],
                color="black",
                marker="x",
                label="Anchor ICs (pre-IMF)",
                zorder=3,
            )

        # refined ICs
        for i, s in zip(self.ic_list_["ic"].to_numpy(), self.ic_side_.to_numpy()):
            # convert dataframe index-domain sample idx to time
            tt = i / float(self.sampling_rate_hz)
            ax2.axvline(tt, color=("red" if s == "R" else "orange"), linestyle="--", alpha=0.8)

        ax2.set_title("Step 2: Filtered SI + Anchor ICs + Refined ICs")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        # 3) ML detrended + FCs
        ax3.plot(t, ml_det, color="green", alpha=0.8, label="ML detrended (SSA groups 1+2)")
        for i, s in zip(self.fc_list_["fc"].to_numpy(), self.fc_side_.to_numpy()):
            tt = i / float(self.sampling_rate_hz)
            ax3.axvline(tt, color=("darkred" if s == "R" else "gold"), linestyle=":", alpha=0.85)

        ax3.set_title("Step 3: FC detection on ML detrended")
        ax3.set_xlabel("Time (s)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
