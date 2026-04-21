from typing import Any
import os
from pathlib import Path
from tcn import TCN
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, resample, find_peaks
from typing_extensions import Self, Unpack

from .base import BaseGeDetector


class GedTcn(BaseGeDetector):
    """
    Initial / Final Contact detection using a trained TCN model.
    """

    def __init__(
            self,
            model_name: str = "TCN_bestModel1",
            target_fs: float = 100.0,
            lowpass_hz: float = 20.0,
            butter_order: int = 4,
            window_size: int = 200,
            prob_threshold: float = 0.24,
            min_peak_distance: int = 25,
    ) -> None:

        self.model_name = model_name
        self.target_fs = target_fs

        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order

        self.window_size = window_size

        self.prob_threshold = prob_threshold
        self.min_peak_distance = min_peak_distance

        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            model_dir = Path(__file__).resolve().parents[1] / "models"
            arch_path = model_dir / f"{self.model_name}.json"
            weights_path = model_dir / f"{self.model_name}.weights.h5"
            if not os.path.exists(arch_path):
                raise FileNotFoundError(f"Model not found: {arch_path}")
            loaded_json = open(arch_path, 'r').read()
            reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
            reloaded_model.load_weights(weights_path)
            self._model = reloaded_model

    def _apply_lowpass(self, X: np.ndarray, fs: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(self.butter_order, self.lowpass_hz / nyq, btype="low")
        return filtfilt(b, a, X, axis=0)

    def _conditional_zscore(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        Xn = (X - mean) / std
        orig_max = np.max(np.abs(X), axis=0)
        norm_max = np.max(np.abs(Xn), axis=0)
        if np.all(norm_max < orig_max):
            return Xn
        return X

    def _resample_if_needed(self, X: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
        if np.isclose(fs, self.target_fs):
            return X, fs
        n_new = int(len(X) * self.target_fs / fs)
        X_rs = resample(X, n_new, axis=0)
        return X_rs, self.target_fs

    def _window_signal(self, X: np.ndarray) -> np.ndarray:
        windows = []
        for i in range(0, len(X) - self.window_size + 1, self.window_size):
            win = X[i:i + self.window_size]
            win = self._conditional_zscore(win)
            windows.append(win)
        return np.asarray(windows)

    def _reconstruct_output(self, Y: np.ndarray, total_len: int) -> np.ndarray:
        Yc = np.zeros((total_len, 4))
        idx = 0
        for i in range(Y.shape[0]):
            Yc[idx:idx + self.window_size] = Y[i]
            idx += self.window_size
        return Yc

    def _detect_peaks(self, prob: np.ndarray) -> np.ndarray:
        peaks, _ = find_peaks(prob, height=self.prob_threshold, distance=self.min_peak_distance)
        return peaks

    def _plot_debug(
            self,
            raw_acc_is: np.ndarray,
            filtered_acc_is: np.ndarray,
            resampled_acc_is: np.ndarray,
            Yc: np.ndarray,
            ic_indices: np.ndarray,
            ic_sides: np.ndarray
    ):
        """
        Debug plot with linked x-axes for the resampled/processed domain.
        """
        t = np.arange(0,len(raw_acc_is))/self.sampling_rate_hz
        t_rs = np.arange(0,len(resampled_acc_is))/self.target_fs
        # We create the figure. axes[1], [2], and [3] will share the same X-axis.
        fig = plt.figure(figsize=(14, 12))
        ax1 = plt.subplot(4, 1, 1)  # Raw/Filtered (Original FS)
        ax2 = plt.subplot(4, 1, 2, sharex=ax1)  # Resampled (Target FS)
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)  # Likelihood (Target FS)
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)  # Final Events (Target FS)

        model_type = self.__class__.__name__.replace("Icd", "")
        fig.suptitle(f'Debug Pipeline: {model_type} - {self.model_name}', fontsize=16)

        # --- 1. Filtering (Original Sample Domain) ---
        ax1.plot(t, raw_acc_is, color='gray', alpha=0.4, label='Raw')
        ax1.plot(t, filtered_acc_is, color='blue', label=f'Filtered ({self.lowpass_hz}Hz LP)')
        ax1.set_title('Step 1: Filtering (Original Sampling Rate)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # --- 2. Resampling (Target Sample Domain) ---
        ax2.plot(t_rs, resampled_acc_is, color='purple', label=f'Resampled to {self.target_fs}Hz')
        ax2.set_title('Step 2: Resampling')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # --- 3. Model Likelihood (Target Sample Domain) ---
        ax3.plot(t_rs, Yc[:, 0], label='L-IC Prob', color='green', alpha=0.7)
        ax3.plot(t_rs, Yc[:, 2], label='R-IC Prob', color='red', alpha=0.7)
        ax3.axhline(y=self.prob_threshold, color='black', linestyle=':', label='Threshold')
        ax3.set_title('Step 3: Model Likelihood (Linked X-Axis)')
        ax3.set_ylabel('Probability')
        ax3.set_ylim([-0.05, 1.05])
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # --- 4. Final Detected Events (Target Sample Domain) ---
        ax4.plot(t_rs, resampled_acc_is, color='black', alpha=0.6, label='Processed Signal')
        l_mask = (ic_sides == 'L')
        r_mask = (ic_sides == 'R')

        if np.any(l_mask):
            idx_l = ic_indices[l_mask]
            ax4.plot(t_rs[idx_l], resampled_acc_is[idx_l], 'go', label='Left IC', markersize=8)
        if np.any(r_mask):
            idx_r = ic_indices[r_mask]
            ax4.plot(t_rs[idx_r], resampled_acc_is[idx_r], 'ro', label='Right IC', markersize=8)

        ax4.set_title('Step 4: Final Events (Linked X-Axis)')
        ax4.set_xlabel('Samples (at Target Frequency)')
        ax4.set_ylabel('Acc (m/s^2)')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_columns: list[str] = ["acc_is", "acc_ml", "acc_ap"],
            gyr_columns: list[str] = ["gyr_is", "gyr_ml", "gyr_ap"],
            plot_debug: bool = True,
            **_: Unpack[dict[str, Any]],
    ) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self._load_model()

        # Capture original raw for plotting
        raw_acc_is = data[acc_columns[0]].to_numpy()

        acc = data[acc_columns].to_numpy()
        gyr = data[gyr_columns].to_numpy()
        X = np.concatenate([acc, gyr], axis=1)

        # Preprocessing chain
        X_filtered = self._apply_lowpass(X, sampling_rate_hz)
        filtered_acc_is = X_filtered[:, 0].copy()

        X_rs, fs_rs = self._resample_if_needed(X_filtered, sampling_rate_hz)
        resampled_acc_is = X_rs[:, 0].copy()

        X_win = self._window_signal(X_rs)

        if len(X_win) == 0:
            self.ic_list_ = pd.DataFrame(columns=["ic"])
            self.fc_list_ = pd.DataFrame(columns=["fc"])
            self.ic_side_ = pd.Series(dtype=str)
            self.fc_side_ = pd.Series(dtype=str)
            return self

        Y_pred = self._model.predict(X_win, verbose=0)
        total_len = len(X_win) * self.window_size
        Yc = self._reconstruct_output(Y_pred, total_len)

        # Match resampled signal length to windowed reconstruction
        if len(resampled_acc_is) > total_len:
            resampled_acc_is = resampled_acc_is[:total_len]

        L_ic = self._detect_peaks(Yc[:, 0])
        R_ic = self._detect_peaks(Yc[:, 2])
        L_fc = self._detect_peaks(Yc[:, 1])
        R_fc = self._detect_peaks(Yc[:, 3])

        ic, ic_side = [], []
        fc, fc_side = [], []

        for i in L_ic: ic.append(i); ic_side.append("L")
        for i in R_ic: ic.append(i); ic_side.append("R")
        for i in L_fc: fc.append(i); fc_side.append("L")
        for i in R_fc: fc.append(i); fc_side.append("R")

        ic, ic_side = np.asarray(ic), np.asarray(ic_side)
        fc, fc_side = np.asarray(fc), np.asarray(fc_side)

        if plot_debug:
            self._plot_debug(
                raw_acc_is,
                filtered_acc_is,
                resampled_acc_is,
                Yc,
                ic,
                ic_side
            )

        # --- Resampling detected indices back to original frequency ---
        resample_ratio = sampling_rate_hz / self.target_fs
        ic = (ic * resample_ratio).astype(int)
        fc = (fc * resample_ratio).astype(int)
        # --------------------------------------------------------------

        order_ic = np.argsort(ic)
        ic, ic_side = ic[order_ic], ic_side[order_ic]

        order_fc = np.argsort(fc)
        fc, fc_side = fc[order_fc], fc_side[order_fc]

        self.ic_list_ = pd.DataFrame({"ic": ic},
            index=pd.RangeIndex(len(ic), name="step_id"),)
        self.fc_list_ = pd.DataFrame({"fc": fc},
            index=pd.RangeIndex(len(fc), name="step_id"),)
        self.ic_side_ = pd.Series(ic_side, index=self.ic_list_.index, name="side")
        self.fc_side_ = pd.Series(fc_side, index=self.fc_list_.index, name="side")

        return self