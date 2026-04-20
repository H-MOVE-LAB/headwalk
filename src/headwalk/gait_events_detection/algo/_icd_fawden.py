import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from typing_extensions import Self
import matplotlib.pyplot as plt
from typing import Sequence


from .base import BaseIcDetector


class IcdFawden(BaseIcDetector):
    """
    TP-EAR (Temporal Parameters from the EAR) algorithm - cleaned & modernized.

    Notes:
      - This implementation preserves the original detection logic:
        bandpass IS (0.5-12Hz), SSA dominant extraction, adaptive windowing around
        dominant peaks, peak-count based IC/FC assignment, sharpening fallback.
      - Changed lateralization logic: IC side = 'R' if peakDiff < 0 else 'L'
        where peakDiff = dominant_ml[i+1] - dominant_ml[i].
      - FC side is set to the opposite of the immediately preceding IC.
      - Outputs match IcdTcn contract:
          self.ic_list_  -> DataFrame with column "ic" (sample indices, original domain)
          self.fc_list_  -> DataFrame with column "fc"
          self.ic_side_  -> Series indexed like ic_list_, values "L"/"R"
          self.fc_side_  -> Series indexed like fc_list_, values "L"/"R"
    """

    def __init__(
        self,
        si_low_hz: float = 0.5,
        si_high_hz: float = 12.0,
        ssa_window_len_s: float = 2.0,
        sharpening_weight: float = 0.45, #optimized: 0.45, #original: 0.6,
        amp_thres: float = 0.8,
        ml_lowpass_hz: float = 20.0,
    ) -> None:
        self.si_low_hz = float(si_low_hz)
        self.si_high_hz = float(si_high_hz)
        self.ssa_window_len_s = float(ssa_window_len_s)
        self.sharpening_weight = float(sharpening_weight)
        self.amp_thres = float(amp_thres)
        self.ml_lowpass_hz = float(ml_lowpass_hz)

        # result containers
        self.ic_list_ = pd.DataFrame(columns=["ic"])
        self.fc_list_ = pd.DataFrame(columns=["fc"])
        self.ic_side_ = pd.Series(dtype=str)
        self.fc_side_ = pd.Series(dtype=str)

        # internal
        self.sampling_rate_hz = None

    # -------------------------
    # Public API
    # -------------------------
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_is_col: str = "acc_is",
        acc_ml_col: str = "acc_ml",
        plot_debug: bool = True,
        **kwargs
    ) -> Self:
        """
        Run full detection pipeline on input dataframe.

        Input:
          data - pandas.DataFrame with columns acc_is_col and acc_ml_col
          sampling_rate_hz - original sampling frequency (Hz)

        Returns self with attributes:
          ic_list_, fc_list_, ic_side_, fc_side_
        """

        fs = float(sampling_rate_hz)
        self.sampling_rate_hz = fs

        # raw signals
        raw_is = data[acc_is_col].to_numpy()
        raw_ml = data[acc_ml_col].to_numpy()

        # 1) Filter IS (bandpass) and ML (lowpass for laterality)
        acc_is_filt = self._bandpass_filter(raw_is, self.si_low_hz, self.si_high_hz, fs)
        acc_ml_filt = self._lowpass_filter(raw_ml, self.ml_lowpass_hz, fs)

        # 2) SSA dominant components
        L = int(max(2, round(self.ssa_window_len_s * fs)))  # ensure >=2
        dom_is = self._get_ssa_dominant(acc_is_filt, L)
        dom_ml = self._get_ssa_dominant(acc_ml_filt, L)

        # 3) Dominant peaks -> candidate gait cycles
        min_dist = int(round(0.4 * fs))  # 0.4s minimum distance
        dom_peaks, _ = find_peaks(dom_is, distance=min_dist)

        ic_indices = []
        fc_indices = []

        # 4) Adaptive windowing + pick IC/FC inside filtered IS
        for peak_idx in dom_peaks:
            # window start: walk backward until amplitude drops below threshold% of peak
            threshold = self.amp_thres * dom_is[peak_idx]
            w_start = peak_idx
            while w_start > 0 and dom_is[w_start] > threshold:
                w_start -= 1

            # window end: advance until slope turns positive (local trough following peak)
            w_end = peak_idx
            while w_end < len(dom_is) - 1:
                if dom_is[w_end + 1] > dom_is[w_end]:
                    break
                w_end += 1

            # validate window size
            if w_end <= w_start + 2:
                continue

            # examine original filtered IS inside window
            window_sig = acc_is_filt[w_start : w_end + 1]
            win_peaks, _ = find_peaks(window_sig)

            final_ic = -1
            final_fc = -1

            # Case A: exactly 2 peaks -> 1st = IC, 2nd = FC
            if len(win_peaks) == 2:
                final_ic = w_start + win_peaks[0]
                final_fc = w_start + win_peaks[1]

            # Case B: >2 peaks -> IC closest to SSA dominant peak, FC latest positive amplitude peak
            elif len(win_peaks) > 2:
                abs_peaks = w_start + win_peaks
                closest_i = int(np.argmin(np.abs(abs_peaks - peak_idx)))
                final_ic = int(abs_peaks[closest_i])

                valid_fc = [int(p) for p in abs_peaks if acc_is_filt[p] > 0]
                final_fc = int(valid_fc[-1]) if valid_fc else int(abs_peaks[-1])

            # Case C: <=1 peak -> sharpening and fallback
            else:
                sharp_sig = self._sharpen_signal(window_sig, self.sharpening_weight)
                sharp_peaks, _ = find_peaks(sharp_sig)

                if len(sharp_peaks) >= 2:
                    abs_sh_peaks = w_start + sharp_peaks
                    closest_i = int(np.argmin(np.abs(abs_sh_peaks - peak_idx)))
                    final_ic = int(abs_sh_peaks[closest_i])

                    valid_fc = [int(p) for p in abs_sh_peaks if acc_is_filt[p] > 0]
                    final_fc = int(valid_fc[-1]) if valid_fc else int(abs_sh_peaks[-1])
                else:
                    # fallback: IC = dominant peak, FC = min gradient point after IC or window end
                    final_ic = int(peak_idx)
                    if final_ic < w_start + len(window_sig) - 1:
                        seg = window_sig[(final_ic - w_start):]
                        if len(seg) > 1:
                            g = np.gradient(seg)
                            min_g_idx = int(np.argmin(g))
                            final_fc = int(final_ic + min_g_idx)
                        else:
                            final_fc = int(w_end)
                    else:
                        final_fc = int(w_end)

            if final_ic != -1 and final_fc != -1:
                ic_indices.append(int(final_ic))
                fc_indices.append(int(final_fc))

        ic_indices = np.asarray(ic_indices, dtype=int)
        fc_indices = np.asarray(fc_indices, dtype=int)

        # 5) Laterality: classify IC by dominant ML slope rule:
        #    peakDiff = dom_ml[i+1] - dom_ml[i]; RIGHT if peakDiff < 0 else LEFT
        ic_sides = self._find_ic_side_from_dominant_ml(dom_ml, ic_indices)

        # 6) FC side = opposite of preceding IC
        fc_sides = self._assign_fc_side_opposite_ic(fc_indices, ic_indices, ic_sides)

        # 7) Store outputs with same structure as IcdTcn
        if len(ic_indices) > 0:
            order_ic = np.argsort(ic_indices)
            ic_sorted = ic_indices[order_ic]
            ic_side_sorted = ic_sides[order_ic]
            self.ic_list_ = pd.DataFrame({"ic": ic_sorted}, index=pd.RangeIndex(len(ic_sorted), name="step_id"))
            self.ic_side_ = pd.Series(ic_side_sorted, index=self.ic_list_.index, name="side")
        else:
            self.ic_list_ = pd.DataFrame(columns=["ic"])
            self.ic_side_ = pd.Series(dtype=str)

        if len(fc_indices) > 0:
            order_fc = np.argsort(fc_indices)
            fc_sorted = fc_indices[order_fc]
            fc_side_sorted = fc_sides[order_fc]
            self.fc_list_ = pd.DataFrame({"fc": fc_sorted}, index=pd.RangeIndex(len(fc_sorted), name="step_id"))
            self.fc_side_ = pd.Series(fc_side_sorted, index=self.fc_list_.index, name="side")
        else:
            self.fc_list_ = pd.DataFrame(columns=["fc"])
            self.fc_side_ = pd.Series(dtype=str)

        # 8) Optional debug plot (time in seconds)
        if plot_debug:
            t_sec = np.arange(len(acc_is_filt)) / fs
            # pass arrays and indices (ints)
            self._plot_debug(
                t_sec,
                acc_is_filt,
                dom_is,
                acc_ml_filt,
                dom_ml,
                ic_indices,
                ic_sides,
                fc_indices,
                fc_sides,
            )

        return self

    # ==========================================================
    # Helpers: lateralization and FC assignment
    # ==========================================================
    def _find_ic_side_from_dominant_ml(self, dom_ml: Sequence[float], ic_idx: np.ndarray) -> np.ndarray:
        """
        New lateralization rule:
          - compute peakDiff = dom_ml[i+1] - dom_ml[i]
          - side = 'R' if peakDiff < 0 else 'L'

        dom_ml can be a numpy array or sequence; ic_idx are sample indices (ints).
        Returns np.array of 'L'/'R' strings aligned with ic_idx.
        """
        dom_ml = np.asarray(dom_ml)
        if ic_idx is None or len(ic_idx) == 0:
            return np.asarray([], dtype=str)

        ic_idx = np.asarray(ic_idx, dtype=int)
        # safe indexing: for last IC use last available diff (0)
        last_valid = max(0, len(dom_ml) - 2)
        safe_idx = np.clip(ic_idx, 0, last_valid)

        peak_diff = dom_ml[safe_idx + 1] - dom_ml[safe_idx]
        sides = np.where(peak_diff < 0, "R", "L").astype(str)
        return sides

    def _assign_fc_side_opposite_ic(self, fc_idx: np.ndarray, ic_idx: np.ndarray, ic_side: np.ndarray) -> np.ndarray:
        """
        For each FC, find the immediately preceding IC (strictly before FC).
        Assign FC side as opposite of that IC. If none found -> 'U'.
        """
        if fc_idx is None or len(fc_idx) == 0:
            return np.asarray([], dtype=str)
        if ic_idx is None or len(ic_idx) == 0:
            return np.array(["U"] * len(fc_idx), dtype=str)

        ic_idx = np.asarray(ic_idx, dtype=int)
        ic_side = np.asarray(ic_side, dtype=str)
        fc_idx = np.asarray(fc_idx, dtype=int)

        order = np.argsort(ic_idx)
        ic_sorted = ic_idx[order]
        side_sorted = ic_side[order]

        positions = np.searchsorted(ic_sorted, fc_idx, side="left")
        fc_sides = []
        for pos in positions:
            prev_idx = pos - 1
            if prev_idx >= 0:
                ic_s = side_sorted[prev_idx]
                fc_sides.append("L" if ic_s == "R" else "R")
            else:
                fc_sides.append("U")
        return np.asarray(fc_sides, dtype=str)

    # ==========================================================
    # Signal processing utilities (unchanged logic)
    # ==========================================================
    def _bandpass_filter(self, data: np.ndarray, low: float, high: float, fs: float, order: int = 2) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, data)

    def _lowpass_filter(self, data: np.ndarray, cut: float, fs: float, order: int = 2) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(order, cut / nyq, btype="low")
        return filtfilt(b, a, data)

    def _sharpen_signal(self, y: np.ndarray, k: float) -> np.ndarray:
        """
        sharpening: f - k * f''  (approx with discrete second difference)
        """
        d2 = np.diff(y, n=2)
        d2_padded = np.pad(d2, (1, 1), mode="edge")
        return y - (k * d2_padded)

    def _get_ssa_dominant(self, signal: np.ndarray, L: int) -> np.ndarray:
        """
        Single-channel SSA (rank-1 reconstruction).
        Returns the dominant reconstructed component (same length as input).
        """
        N = len(signal)
        if L > N // 2:
            L = N // 2
        if L < 2:
            L = 2
        K = N - L + 1

        # Trajectory / Hankel matrix (L x K)
        X = np.column_stack([signal[i : i + L] for i in range(K)])

        # SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # rank-1 reconstruction
        X_elem = S[0] * np.outer(U[:, 0], Vt[0, :])

        # diagonal averaging
        return self._diagonal_averaging(X_elem)

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        L, K = matrix.shape
        N = L + K - 1
        series = np.zeros(N)
        for n in range(N):
            start = max(0, n - K + 1)
            end = min(L, n + 1)
            count = end - start
            s = 0.0
            for i in range(start, end):
                s += matrix[i, n - i]
            series[n] = s / count
        return series

    # ==========================================================
    # Debug plotting (updated as requested)
    # ==========================================================
    def _plot_debug(
        self,
        t: np.ndarray,
        si_filt: np.ndarray,
        si_dom: np.ndarray,
        ml_filt: np.ndarray,
        ml_dom: np.ndarray,
        ic_idx: np.ndarray,
        ic_side: np.ndarray,
        fc_idx: np.ndarray,
        fc_side: np.ndarray,
    ) -> None:
        """
        Two-panel plot:
          - Top: SI filtered (blue solid) and SI dominant (orange dashed).
                 IC: green star (L), red star (R)
                 FC: green triangle (L), red triangle (R)
          - Bottom: ML filtered + ML dominant, same marker scheme.
        t is time vector in seconds (same length as signals).
        """
        si_filt = np.asarray(si_filt)
        si_dom = np.asarray(si_dom)
        ml_filt = np.asarray(ml_filt)
        ml_dom = np.asarray(ml_dom)

        ic_idx = np.asarray(ic_idx, dtype=int) if ic_idx is not None else np.array([], dtype=int)
        fc_idx = np.asarray(fc_idx, dtype=int) if fc_idx is not None else np.array([], dtype=int)
        ic_side = np.asarray(ic_side, dtype=str) if ic_side is not None else np.array([], dtype=str)
        fc_side = np.asarray(fc_side, dtype=str) if fc_side is not None else np.array([], dtype=str)

        fig, (ax_si, ax_ml) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Fawden - Detection debug", fontsize=14)

        # --- SI axis plot ---
        ax_si.plot(t, si_filt, color="blue", linewidth=1.2, label="SI filtered")
        ax_si.plot(t, si_dom, color="orange", linestyle="--", linewidth=1.2, label="SI dominant")

        # IC markers (stars)
        if ic_idx.size > 0:
            mask_L = ic_side == "L"
            mask_R = ic_side == "R"
            if np.any(mask_L):
                ax_si.plot(t[ic_idx[mask_L]], si_filt[ic_idx[mask_L]], marker="*", linestyle="None", color="green", markersize=9, label="IC Left")
            if np.any(mask_R):
                ax_si.plot(t[ic_idx[mask_R]], si_filt[ic_idx[mask_R]], marker="*", linestyle="None", color="red", markersize=9, label="IC Right")

        # FC markers (triangles)
        if fc_idx.size > 0:
            mask_Lf = fc_side == "L"
            mask_Rf = fc_side == "R"
            if np.any(mask_Lf):
                ax_si.plot(t[fc_idx[mask_Lf]], si_filt[fc_idx[mask_Lf]], marker="^", linestyle="None", color="green", markersize=8, label="FC Left")
            if np.any(mask_Rf):
                ax_si.plot(t[fc_idx[mask_Rf]], si_filt[fc_idx[mask_Rf]], marker="^", linestyle="None", color="red", markersize=8, label="FC Right")

        ax_si.set_ylabel("Acc SI (m/s^2)")
        ax_si.grid(True, alpha=0.25)
        ax_si.legend(loc="upper right")

        # --- ML axis plot ---
        ax_ml.plot(t, ml_filt, color="blue", linewidth=1.2, label="ML filtered")
        ax_ml.plot(t, ml_dom, color="orange", linestyle="--", linewidth=1.2, label="ML dominant")

        # plot same markers on ML subplot for visual correspondence
        if ic_idx.size > 0:
            if np.any(mask_L):
                ax_ml.plot(t[ic_idx[mask_L]], ml_filt[ic_idx[mask_L]], marker="*", linestyle="None", color="green", markersize=9)
            if np.any(mask_R):
                ax_ml.plot(t[ic_idx[mask_R]], ml_filt[ic_idx[mask_R]], marker="*", linestyle="None", color="red", markersize=9)

        if fc_idx.size > 0:
            if np.any(mask_Lf):
                ax_ml.plot(t[fc_idx[mask_Lf]], ml_filt[fc_idx[mask_Lf]], marker="^", linestyle="None", color="green", markersize=8)
            if np.any(mask_Rf):
                ax_ml.plot(t[fc_idx[mask_Rf]], ml_filt[fc_idx[mask_Rf]], marker="^", linestyle="None", color="red", markersize=8)

        ax_ml.set_ylabel("Acc ML (m/s^2)")
        ax_ml.set_xlabel("Time (s)")
        ax_ml.grid(True, alpha=0.25)
        ax_ml.legend(loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()