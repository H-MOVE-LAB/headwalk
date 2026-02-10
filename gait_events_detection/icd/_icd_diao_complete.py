from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema, firwin, kaiserord, filtfilt
from scipy.linalg import eigh
from pyts.decomposition import SingularSpectrumAnalysis
from .base import BaseIcDetector


class IcdDiaoComplete(BaseIcDetector):
    """
    Initial Contact (IC) and Final Contact (FC) detection based on Diao et al. (2020).
    Includes the complete Iterative Mean Filtering (IMF) for signal refinement.

    IC: Minima of dominant SSA component of SI acceleration.
    FC: First local maximum in ML acceleration after IC with cross-lateral logic.
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
        self.filter_taps = None

        # Results
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
            iterative_filtering_flag: bool = False,
            plt_flag: bool = True,
            **_,
    ):
        self.sampling_rate_hz = sampling_rate_hz
        if self.window_length is None:
            self.window_length = int(self.sampling_rate_hz)

        self._initialize_filter()
        self._initialize_ssa()

        # 1. Preliminary Filtering
        acc_si_raw = data[acc_si_col].to_numpy()
        acc_ml_raw = data[acc_ml_col].to_numpy()

        acc_si_filt_orig = -self._filter_signal(acc_si_raw)  # Per debug plot
        acc_ml_filt = -self._filter_signal(acc_ml_raw)

        # Salviamo gli IC pre-refinement per il grafico
        # Usiamo una SSA temporanea o il segnale filtrato per vedere dove cadrebbero gli IC all'inizio
        acc_si_ssa_pre = self.ssa.fit_transform(acc_si_filt_orig.reshape(1, -1))
        self.ic_pre_refinement, _ = self._detect_ic_with_side(acc_si_ssa_pre[0, 1, :], acc_ml_filt)

        # 2. Complete Iterative Mean Filtering (Refinement)
        acc_si_filt = acc_si_filt_orig.copy()
        if iterative_filtering_flag:
            acc_si_filt = self._apply_full_imf(acc_si_filt, acc_ml_filt)

        # 3. SSA Decomposition (Final)
        acc_si_ssa = self.ssa.fit_transform(acc_si_filt.reshape(1, -1))
        acc_ml_ssa = self.ssa.fit_transform(acc_ml_filt.reshape(1, -1))

        acc_si_dom = acc_si_ssa[0, 1, :]
        acc_ml_det = acc_ml_ssa[0, 1, :] + acc_ml_ssa[0, 2, :]

        # 4. IC/FC Detection with Laterality
        ic_idx, ic_side = self._detect_ic_with_side(acc_si_dom, acc_ml_det)
        fc_idx, fc_side = self._detect_fc_with_side(acc_ml_det, ic_idx, ic_side)

        # 5. Result Storage
        offset = data.index[0]
        self._store_results(ic_idx + offset, ic_side, fc_idx + offset, fc_side)

        if plt_flag:
            # Passiamo anche gli IC pre-refinement (aggiungendo l'offset)
            self._plot_debug(acc_si_raw, acc_ml_raw, acc_si_dom, acc_ml_det,
                             ic_pre=self.ic_pre_refinement + offset)

        return self

    def _apply_full_imf(self, acc_si: np.ndarray, acc_ml: np.ndarray) -> np.ndarray:
        """Complete IMF algorithm implementation using phase space eigenvalue decomposition."""
        d = 1
        max_iter = 10
        prev_metrics = [np.inf] * 4
        best_si = acc_si.copy()

        for _ in range(max_iter):
            # Smooth signal
            s_si = pd.Series(acc_si).rolling(window=d, center=True).mean().ffill().bfill().to_numpy()

            # Detect temporary events for phase space construction
            ic_idx, ic_side = self._detect_ic_with_side(s_si, acc_ml)
            rhc, lhc = ic_idx[ic_side == "R"], ic_idx[ic_side == "L"]

            if len(rhc) < 3 or len(lhc) < 3: break

            try:
                metrics = self._calculate_imf_metrics(acc_ml, rhc, lhc)
                # If all metrics decrease, keep smoothing
                if all(c < p for c, p in zip(metrics, prev_metrics)):
                    best_si, prev_metrics, d = s_si, metrics, d + 2
                else:
                    break
            except:
                print("IMF Metrics not calculated")
                break
        return best_si

    def _calculate_imf_metrics(self, acc_ml, rhc, lhc):
        """Builds M/N matrices and returns [lambda1_m, lambda1_n, v1, v2]."""

        def get_eigen_v(events, is_rhc=True):
            m_list = []
            for i in range(len(events) - 2):
                p = min(np.abs(lhc[i] - rhc[i]), np.abs(lhc[i + 1] - rhc[i + 1])) if is_rhc else min(
                    np.abs(rhc[i + 1] - lhc[i]), np.abs(rhc[i + 2] - lhc[i + 1]))
                tau, eps, k = (lhc[i] - rhc[i]) / p, (lhc[i + 1] - rhc[i + 1]) / p, int(p)
                if not is_rhc: tau, eps = (rhc[i + 1] - lhc[i]) / p, (rhc[i + 2] - lhc[i + 1]) / p

                row1 = [acc_ml[int(events[i] + j * tau)] for j in range(1, k + 1)]
                row2 = [-acc_ml[int(events[i + 1] + j * eps)] for j in range(1, k + 1)]
                m_list.append(np.array([row1, row2]))

            cov = np.cov(np.hstack(m_list))
            vals, vecs = eigh(cov)
            idx = np.argsort(vals)[::-1]
            ref = np.array([1, -1 if is_rhc else 1]) / np.sqrt(2)
            return vals[idx[0]], np.abs(np.dot(vecs[:, idx[0]].T, ref))

        l1_m, v1 = get_eigen_v(rhc, True)
        l1_n, v2 = get_eigen_v(lhc, False)
        return [l1_m, l1_n, v1, v2]

    def _detect_ic_with_side(self, si_dom, ml_det):
        peaks, _ = find_peaks(-si_dom, height=0.2, distance=int(0.2 * self.sampling_rate_hz))
        sides = []
        for p in peaks:
            # Increasing ML -> Right | Decreasing ML -> Left
            slope = ml_det[min(p + 1, len(ml_det) - 1)] - ml_det[p]
            sides.append("R" if slope > 0 else "L")
        return peaks, np.array(sides)

    def _detect_fc_with_side(self, ml_det, ic_idx, ic_side):
        maxima = argrelextrema(ml_det, np.greater, order=2)[0]
        minima = argrelextrema(ml_det, np.less, order=2)[0]

        fc_idx, fc_side = [], []
        for ic, side in zip(ic_idx, ic_side):
            if side == "R":
                cand = maxima[maxima > ic]
            else:
                cand = minima[minima > ic]

            if len(cand) > 0:
                fc_idx.append(cand[0])
                fc_side.append("L" if side == "R" else "R")  # Cross-lateral logic
        return np.array(fc_idx), np.array(fc_side)

    def _store_results(self, ic, ic_side, fc, fc_side):
        idx_i, idx_f = np.argsort(ic), np.argsort(fc)
        self.ic_list_ = pd.DataFrame({"ic": ic[idx_i]}, index=pd.RangeIndex(len(ic), name="step_id"))
        self.fc_list_ = pd.DataFrame({"fc": fc[idx_f]}, index=pd.RangeIndex(len(fc), name="step_id"))
        self.ic_side_ = pd.Series(ic_side[idx_i], index=self.ic_list_.index, name="side")
        self.fc_side_ = pd.Series(fc_side[idx_f], index=self.fc_list_.index, name="side")

    def _initialize_filter(self):
        nyq = self.sampling_rate_hz / 2.0
        n, beta = kaiserord(self.ripple_db, 2.0 / nyq)
        self.filter_taps = firwin(n, self.cutoff_hz / nyq, window=("kaiser", beta))

    def _filter_signal(self, x):
        return filtfilt(self.filter_taps, [1.0], x)

    def _initialize_ssa(self):
        self.ssa = SingularSpectrumAnalysis(window_size=int(self.window_length),
                                            groups=[[0], [1], np.arange(2, self.window_length)])

    def _plot_debug(self, si_raw, ml_raw, si_dom, ml_det, ic_pre=None):
        """
        Visualizza il segnale SI e ML con gli eventi rilevati.
        Gli IC pre-refinement sono mostrati come piccoli punti neri per confronto.
        """

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Subplot 1: SI Axis (IC Detection)
        ax1.plot(si_raw, color='gray', alpha=0.3, label='Raw SI (Inverted)')
        ax1.plot(si_dom, color='blue', linewidth=1.5, label='Dominant SSA Component (Filtered)')

        # Plot IC prima del refinement (IMF)
        if ic_pre is not None:
            ax1.scatter(ic_pre, si_dom[np.clip((ic_pre - ic_pre[0]).astype(int), 0, len(si_dom) - 1)],
                        color='black', marker='x', s=40, label='IC Pre-Refinement', zorder=5)

        # Plot IC finali (Post-Refinement)
        for i, s in zip(self.ic_list_['ic'], self.ic_side_):
            color = 'red' if s == 'R' else 'orange'
            ax1.axvline(i, color=color, linestyle='--', alpha=0.8,
                        label=f'IC {s}' if f'IC {s}' not in ax1.get_legend_handles_labels()[1] else "")

        ax1.set_title("Superior-Inferior Axis & Initial Contacts")
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(alpha=0.3)

        # Subplot 2: ML Axis (FC Detection)
        ax2.plot(ml_raw, color='gray', alpha=0.3, label='Raw ML')
        ax2.plot(ml_det, color='green', linewidth=1.5, label='ML Detrended (SSA 1+2)')

        for i, s in zip(self.fc_list_['fc'], self.fc_side_):
            color = 'darkred' if s == 'R' else 'gold'
            ax2.axvline(i, color=color, linestyle=':', alpha=0.8,
                        label=f'FC {s}' if f'FC {s}' not in ax2.get_legend_handles_labels()[1] else "")

        ax2.set_title("Medio-Lateral Axis & Final Contacts")
        ax2.legend(loc='upper right', fontsize='small')
        ax2.grid(alpha=0.3)

        plt.xlabel("Index")
        plt.tight_layout()
        plt.show()