# himu_pd/gait_events_detection/icd/_lr_cnnlstm.py
"""
LrCnnlstm: Left/Right laterality classification for IC/FC events using CNN-LSTM models.

This block is meant to run *after* event detection:
- Input: raw acc/gyr signals + detected IC indices (+ optionally FC indices)
- Output: ic_side_ and fc_side_ (same style as IcdTcn)

Models location (package-relative)
----------------------------------
himu_pd/gait_events_detection/models/
    bestModel_IC_hw_0p5_accML_gyrAP_gyrV_fold1.keras
    bestModel_FC_hw_0p5_accML_gyrAP_gyrV_fold1.keras   (if available)

Default configuration
---------------------
- half_window_s = 0.5  (=> 1.0 s total window)
- target_fs = 100 Hz
- channels: acc_ml, gyr_ap, gyr_is (3 channels)

Preprocessing
-------------
Mirrors IcdTcn:
1) low-pass Butterworth
2) resample to target_fs (if needed)
3) rotate to gravity (optional but recommended; enabled by default here)
4) window extraction around events
5) conditional z-score normalization per window

Important
---------
- Axis mapping: the training pipeline used the concatenated order:
    [AccX, AccY, AccZ, GyrX, GyrY, GyrZ]
  In this package, users pass acc_columns and gyr_columns that are commonly in
  [acc_is, acc_ml, acc_ap] and [gyr_is, gyr_ml, gyr_ap]. We simply concatenate in
  the given column order.

- The "channel selection" (acc_ml, gyr_ap, gyr_is) is applied AFTER concatenation,
  based on the provided column ordering.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, filtfilt, resample
from typing_extensions import Self, Unpack

from .base import BaseLrClassifier


class LrCnnlstm(BaseLrClassifier):
    """Laterality classification for IC/FC events using CNN-LSTM models."""

    def __init__(
        self,
        *,
        # Model selection
        ic_model_name: str = "bestModel_IC_hw_0p3_accML_gyrV_gyrAP_fold1",
        fc_model_name: str = "bestModel_FC_hw_0p3_accML_gyrV_gyrAP_fold1",
        # Signal processing
        target_fs: float = 100.0,
        lowpass_hz: float = 20.0,
        butter_order: int = 4,
        rotate_to_gravity: bool = False,
        # Windowing (half-window in seconds; total window = 2*half_window_s)
        half_window_s: float = 0.3,
        # Default channel names (package conventions)
        acc_columns: list[str] = ["acc_is", "acc_ml", "acc_ap"],
        gyr_columns: list[str] = ["gyr_is", "gyr_ml", "gyr_ap"],
        # Default channels for inference (subset of the above)
        # Default requested: acc_ml, gyr_ap, gyr_is
        use_channels: list[str] = ["acc_ml", "gyr_is", "gyr_ap"],
    ) -> None:
        self.ic_model_name = ic_model_name
        self.fc_model_name = fc_model_name

        self.target_fs = target_fs
        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order
        self.rotate_to_gravity = rotate_to_gravity

        self.half_window_s = float(half_window_s)

        self.acc_columns = acc_columns
        self.gyr_columns = gyr_columns
        self.use_channels = use_channels

        self._ic_model: Optional[tf.keras.Model] = None
        self._fc_model: Optional[tf.keras.Model] = None

    # -----------------------
    # --- MODEL LOADING -----
    # -----------------------
    def _model_dir(self) -> Path:
        # This file is in: himu_pd/gait_events_detection/icd/_lr_cnnlstm.py
        # models are in:   himu_pd/gait_events_detection/models/
        return Path(__file__).resolve().parents[1] / "models"

    def _load_models(self, need_fc: bool) -> None:
        model_dir = self._model_dir()

        if self._ic_model is None:
            ic_path = model_dir / f"{self.ic_model_name}.keras"
            if not ic_path.exists():
                raise FileNotFoundError(f"IC model not found: {ic_path}")
            self._ic_model = tf.keras.models.load_model(ic_path)

        if need_fc and self._fc_model is None:
            fc_path = model_dir / f"{self.fc_model_name}.keras"
            if not fc_path.exists():
                raise FileNotFoundError(f"FC model not found: {fc_path}")
            self._fc_model = tf.keras.models.load_model(fc_path)

    # -----------------------
    # --- PREPROCESSING -----
    # -----------------------
    def _apply_lowpass(self, X: np.ndarray, fs: float) -> np.ndarray:
        nyq = 0.5 * fs
        b, a = butter(self.butter_order, self.lowpass_hz / nyq, btype="low")
        return filtfilt(b, a, X, axis=0)

    def _resample_if_needed(self, X: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
        if np.isclose(fs, self.target_fs):
            return X, fs
        n_new = int(round(len(X) * self.target_fs / fs))
        X_rs = resample(X, n_new, axis=0)
        return X_rs, self.target_fs

    def _conditional_zscore(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        Xn = (X - mean) / std
        orig_max = np.max(np.abs(X), axis=0)
        norm_max = np.max(np.abs(Xn), axis=0)
        if np.all(norm_max < orig_max):
            return Xn
        return X

    def _rotate_to_gravity(self, acc: np.ndarray, gyr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Same method as your dataset creation.
        Rotates such that the gravity vector aligns with a target axis.

        This is a lightweight duplicate to avoid importing private utils.
        If you prefer, you can import rotate_to_gravity from your utils module.
        """
        # Estimate gravity from early segment
        if len(acc) < 100:
            gravity = np.mean(acc, axis=0)
        else:
            gravity = np.mean(acc[1:100, :], axis=0)

        g_norm = gravity / (np.linalg.norm(gravity) + 1e-12)
        vertical_target = np.array([1.0, 0.0, 0.0])  # keep consistent with your current utils

        rotation_axis = np.cross(g_norm, vertical_target)
        axis_norm = np.linalg.norm(rotation_axis)

        if axis_norm < 1e-6:
            return acc, gyr

        rotation_axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(g_norm, vertical_target), -1.0, 1.0))

        K = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        acc_rot = acc @ R.T
        gyr_rot = gyr @ R.T
        return acc_rot, gyr_rot

    def _build_channel_index_map(self) -> dict[str, int]:
        """
        Map channel names to indices in concatenated X:
          X = [acc_columns..., gyr_columns...]
        Example defaults:
          acc_columns = ["acc_is", "acc_ml", "acc_ap"] -> indices 0,1,2
          gyr_columns = ["gyr_is", "gyr_ml", "gyr_ap"] -> indices 3,4,5
        """
        mapping = {}
        for i, c in enumerate(self.acc_columns):
            mapping[c] = i
        for j, c in enumerate(self.gyr_columns):
            mapping[c] = len(self.acc_columns) + j
        return mapping

    def _select_channels(self, X: np.ndarray) -> np.ndarray:
        mapping = self._build_channel_index_map()
        try:
            idx = [mapping[c] for c in self.use_channels]
        except KeyError as e:
            raise KeyError(
                f"Requested channel {e} not found. "
                f"acc_columns={self.acc_columns}, gyr_columns={self.gyr_columns}, use_channels={self.use_channels}"
            )
        return X[:, idx]

    # -----------------------
    # --- WINDOWING ---------
    # -----------------------
    def _segment_around_events(self, X: np.ndarray, event_idx: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment windows around events.

        Parameters
        ----------
        X : (N, C) preprocessed signal at target_fs
        event_idx : indices in *original* sampling domain (fs provided separately)
        fs : original sampling rate

        Returns
        -------
        X_win : (M, T, C) windows at target_fs
        valid_event_idx_rs : (M,) event indices in resampled domain (center indices)
        """
        if event_idx is None or len(event_idx) == 0:
            return np.zeros((0, 0, X.shape[1]), dtype=np.float32), np.asarray([], dtype=int)

        # Convert event indices to resampled domain
        # resample_ratio: original_fs / target_fs
        # idx_rs = idx / resample_ratio
        resample_ratio = fs / self.target_fs
        event_idx_rs = np.asarray(np.round(event_idx / resample_ratio), dtype=int)

        half_win_samp = int(round(self.half_window_s * self.target_fs))
        win_len = 2 * half_win_samp

        windows = []
        valid_centers = []

        n = X.shape[0]
        for c in event_idx_rs:
            start = int(c) - half_win_samp
            end = start + win_len
            if start < 0 or end > n:
                continue
            w = X[start:end].copy()
            w = self._conditional_zscore(w)
            windows.append(w)
            valid_centers.append(c)

        if not windows:
            return np.zeros((0, win_len, X.shape[1]), dtype=np.float32), np.asarray([], dtype=int)

        return np.asarray(windows, dtype=np.float32), np.asarray(valid_centers, dtype=int)

    # -----------------------
    # --- PUBLIC API --------
    # -----------------------
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        ic: np.ndarray,
        fc: Optional[np.ndarray] = None,
        acc_columns: Optional[list[str]] = None,
        gyr_columns: Optional[list[str]] = None,
        use_channels: Optional[list[str]] = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Predict Left/Right laterality for already-detected ICs (and optionally FCs).

        Parameters
        ----------
        data : pd.DataFrame
            Raw IMU data of a single gait sequence.
        sampling_rate_hz : float
            Sampling rate (Hz) of the raw IMU data.
        ic : np.ndarray
            Detected IC indices in the ORIGINAL sampling domain.
        fc : Optional[np.ndarray]
            Detected FC indices in the ORIGINAL sampling domain.
        acc_columns, gyr_columns : Optional override
            Column names to read from `data`.
        use_channels : Optional override
            Subset of channels (by name) to pass to the CNN-LSTM.

        Returns
        -------
        self
            Sets:
              - self.ic_list_ (DataFrame with column "ic")
              - self.fc_list_ (DataFrame with column "fc")
              - self.ic_side_ (Series of 'L'/'R')
              - self.fc_side_ (Series of 'L'/'R', empty if fc not provided)
        """
        self.data = data
        self.sampling_rate_hz = float(sampling_rate_hz)

        if acc_columns is not None:
            self.acc_columns = acc_columns
        if gyr_columns is not None:
            self.gyr_columns = gyr_columns
        if use_channels is not None:
            self.use_channels = use_channels

        need_fc = False
        self._load_models(need_fc=need_fc)

        # ---------- Build raw X ----------
        acc = data[self.acc_columns].to_numpy(dtype=float)
        gyr = data[self.gyr_columns].to_numpy(dtype=float)
        X = np.concatenate([acc, gyr], axis=1)

        # ---------- Preprocessing chain (like IcdTcn) ----------
        X_filt = self._apply_lowpass(X, self.sampling_rate_hz)

        # Optional gravity alignment (only meaningful if acc/gyr are tri-axial)
        if self.rotate_to_gravity:
            acc_f = X_filt[:, : len(self.acc_columns)]
            gyr_f = X_filt[:, len(self.acc_columns) :]
            # We assume first 3 acc columns and first 3 gyr columns are the tri-axial signals.
            # If you pass fewer than 3 columns, we skip rotation.
            if acc_f.shape[1] >= 3 and gyr_f.shape[1] >= 3:
                acc_rot, gyr_rot = self._rotate_to_gravity(acc_f[:, :3], gyr_f[:, :3])
                acc_f = acc_f.copy()
                gyr_f = gyr_f.copy()
                acc_f[:, :3] = acc_rot
                gyr_f[:, :3] = gyr_rot
                X_filt = np.concatenate([acc_f, gyr_f], axis=1)

        X_rs, _ = self._resample_if_needed(X_filt, self.sampling_rate_hz)

        # Select channels by name (default: acc_ml, gyr_ap, gyr_is)
        X_sel = self._select_channels(X_rs).astype(np.float32)  # (N_rs, C_sel)

        # ---------- Segment around ICs ----------
        X_ic_win, ic_centers_rs = self._segment_around_events(X_sel, np.asarray(ic), self.sampling_rate_hz)

        if X_ic_win.shape[0] == 0:
            # No valid windows -> empty outputs
            self.ic_list_ = pd.DataFrame({"ic": np.asarray([], dtype=int)}, index=pd.RangeIndex(0, name="step_id"))
            self.fc_list_ = pd.DataFrame({"fc": np.asarray([], dtype=int)}, index=pd.RangeIndex(0, name="step_id"))
            self.ic_side_ = pd.Series(dtype=str, index=self.ic_list_.index, name="side")
            self.fc_side_ = pd.Series(dtype=str, index=self.fc_list_.index, name="side")
            return self

        # Inference (IC)
        ic_prob = self._ic_model.predict(X_ic_win, verbose=0)  # (M, 2)
        ic_pred = np.argmax(ic_prob, axis=1).astype(int)       # 0=Left, 1=Right
        ic_side = np.where(ic_pred == 0, "L", "R")

        # Convert valid centers back to original domain indices
        resample_ratio = self.sampling_rate_hz / self.target_fs
        ic_centers_orig = (ic_centers_rs * resample_ratio).astype(int)

        # Sort by time
        order_ic = np.argsort(ic_centers_orig)
        ic_centers_orig = ic_centers_orig[order_ic]
        ic_side = ic_side[order_ic]

        self.ic_list_ = pd.DataFrame(
            {"ic": ic_centers_orig},
            index=pd.RangeIndex(len(ic_centers_orig), name="step_id"),
        )
        self.ic_side_ = pd.Series(ic_side, index=self.ic_list_.index, name="side")

        # ---------- Segment around FCs (optional) ----------
        if need_fc:
            X_fc_win, fc_centers_rs = self._segment_around_events(X_sel, np.asarray(fc), self.sampling_rate_hz)

            if X_fc_win.shape[0] == 0:
                self.fc_list_ = pd.DataFrame({"fc": np.asarray([], dtype=int)}, index=pd.RangeIndex(0, name="step_id"))
                self.fc_side_ = pd.Series(dtype=str, index=self.fc_list_.index, name="side")
                return self

            fc_prob = self._fc_model.predict(X_fc_win, verbose=0)
            fc_pred = np.argmax(fc_prob, axis=1).astype(int)
            fc_side = np.where(fc_pred == 0, "L", "R")

            fc_centers_orig = (fc_centers_rs * resample_ratio).astype(int)

            order_fc = np.argsort(fc_centers_orig)
            fc_centers_orig = fc_centers_orig[order_fc]
            fc_side = fc_side[order_fc]

            self.fc_list_ = pd.DataFrame(
                {"fc": fc_centers_orig},
                index=pd.RangeIndex(len(fc_centers_orig), name="step_id"),
            )
            self.fc_side_ = pd.Series(fc_side, index=self.fc_list_.index, name="side")
        else:
            self.fc_list_ = pd.DataFrame({"fc": np.asarray([], dtype=int)}, index=pd.RangeIndex(0, name="step_id"))
            self.fc_side_ = pd.Series(dtype=str, index=self.fc_list_.index, name="side")

        return self