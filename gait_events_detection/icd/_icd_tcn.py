from typing import Any
import os
from tcn import TCN
from tensorflow.keras.models import model_from_json, load_model

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.signal import butter, filtfilt, resample, find_peaks
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdTcn(BaseIcDetector):
    """
    Initial / Final Contact detection using a trained TCN model.
    """

    def __init__(
        self,
        model_path: str = "gait_events_detection/models/lastTrainedModel",
        target_fs: float = 100.0,
        lowpass_hz: float = 20.0,
        butter_order: int = 4,
        window_size: int = 200,
        prob_threshold: float = 0.5,
        min_peak_distance: int = 25,
    ) -> None:

        self.model_path = model_path
        self.target_fs = target_fs

        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order

        self.window_size = window_size

        self.prob_threshold = prob_threshold
        self.min_peak_distance = min_peak_distance

        self._model = None

    # --------------------------------------------------

    def _load_model(self) -> None:
        """
        Load TCN model only once.
        """

        if self._model is None:

            if not os.path.exists(str(self.model_path) + ".json"):
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}"
                )
            # load model from file
            loaded_json = open(str(self.model_path) + '.json', 'r').read()
            reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
            # restore weights
            reloaded_model.load_weights(str(self.model_path) + '.weights.h5')
            self._model = reloaded_model

    # --------------------------------------------------

    def _apply_lowpass(self, X: np.ndarray, fs: float) -> np.ndarray:
        """
        Butterworth low-pass filter.
        """

        nyq = 0.5 * fs

        b, a = butter(
            self.butter_order,
            self.lowpass_hz / nyq,
            btype="low"
        )

        return filtfilt(b, a, X, axis=0)

    # --------------------------------------------------

    def _conditional_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Conditional Z-score normalization.
        """

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8

        Xn = (X - mean) / std

        orig_max = np.max(np.abs(X), axis=0)
        norm_max = np.max(np.abs(Xn), axis=0)

        if np.all(norm_max < orig_max):
            return Xn

        return X

    # --------------------------------------------------

    def _resample_if_needed(
        self,
        X: np.ndarray,
        fs: float
    ) -> tuple[np.ndarray, float]:
        """
        Resample to target frequency if needed.
        """

        if np.isclose(fs, self.target_fs):
            return X, fs

        n_new = int(len(X) * self.target_fs / fs)

        X_rs = resample(X, n_new, axis=0)

        return X_rs, self.target_fs

    # --------------------------------------------------

    def _window_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Split into non-overlapping windows.
        """

        windows = []

        for i in range(
            0,
            len(X) - self.window_size + 1,
            self.window_size
        ):

            win = X[i:i + self.window_size]

            win = self._conditional_zscore(win)

            windows.append(win)

        return np.asarray(windows)

    # --------------------------------------------------

    def _reconstruct_output(
        self,
        Y: np.ndarray,
        total_len: int
    ) -> np.ndarray:
        """
        Rebuild continuous prediction.
        """

        Yc = np.zeros((total_len, 4))

        idx = 0

        for i in range(Y.shape[0]):

            Yc[idx:idx + self.window_size] = Y[i]

            idx += self.window_size

        return Yc

    # --------------------------------------------------

    def _detect_peaks(self, prob: np.ndarray) -> np.ndarray:
        """
        Peak detection with threshold and distance.
        """

        peaks, _ = find_peaks(
            prob,
            height=self.prob_threshold,
            distance=self.min_peak_distance
        )

        return peaks

    # --------------------------------------------------

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_columns: list[str] = ["acc_is", "acc_ml", "acc_ap"],
        gyr_columns: list[str] = ["gyr_is", "gyr_ml", "gyr_ap"],
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect IC and FC using TCN.

        Input must contain 3 acc + 3 gyro columns.
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        fs = sampling_rate_hz

        # --------------------------------------------------
        # 1. Load model
        # --------------------------------------------------

        self._load_model()

        # --------------------------------------------------
        # 2. Read signals
        # --------------------------------------------------

        acc = data[acc_columns].to_numpy()
        gyr = data[gyr_columns].to_numpy()

        X = np.concatenate([acc, gyr], axis=1)

        # --------------------------------------------------
        # 3. Filtering
        # --------------------------------------------------

        X = self._apply_lowpass(X, fs)

        # --------------------------------------------------
        # 4. Resampling
        # --------------------------------------------------

        X, fs = self._resample_if_needed(X, fs)

        # --------------------------------------------------
        # 5. Windowing
        # --------------------------------------------------

        X_win = self._window_signal(X)

        if len(X_win) == 0:

            self.ic_list_ = pd.DataFrame(columns=["ic"])
            self.fc_list_ = pd.DataFrame(columns=["fc"])

            self.ic_side_ = pd.Series(dtype=str)
            self.fc_side_ = pd.Series(dtype=str)

            return self

        # --------------------------------------------------
        # 6. Inference
        # --------------------------------------------------

        Y_pred = self._model.predict(
            X_win,
            verbose=0
        )

        # --------------------------------------------------
        # 7. Reconstruct output
        # --------------------------------------------------

        total_len = len(X_win) * self.window_size

        Yc = self._reconstruct_output(
            Y_pred,
            total_len
        )

        # Channels
        L_IC = Yc[:, 0]
        L_FC = Yc[:, 1]
        R_IC = Yc[:, 2]
        R_FC = Yc[:, 3]

        # --------------------------------------------------
        # 8. Peak detection
        # --------------------------------------------------

        L_ic = self._detect_peaks(L_IC)
        R_ic = self._detect_peaks(R_IC)

        L_fc = self._detect_peaks(L_FC)
        R_fc = self._detect_peaks(R_FC)

        # --------------------------------------------------
        # 9. Merge + sort
        # --------------------------------------------------

        ic = []
        ic_side = []

        fc = []
        fc_side = []

        for i in L_ic:
            ic.append(i)
            ic_side.append("L")

        for i in R_ic:
            ic.append(i)
            ic_side.append("R")

        for i in L_fc:
            fc.append(i)
            fc_side.append("L")

        for i in R_fc:
            fc.append(i)
            fc_side.append("R")

        ic = np.asarray(ic)
        fc = np.asarray(fc)

        ic_side = np.asarray(ic_side)
        fc_side = np.asarray(fc_side)

        # Sort
        ic_order = np.argsort(ic)
        fc_order = np.argsort(fc)

        ic = ic[ic_order]
        ic_side = ic_side[ic_order]

        fc = fc[fc_order]
        fc_side = fc_side[fc_order]

        # --------------------------------------------------
        # 10. Store results
        # --------------------------------------------------

        self.ic_list_ = pd.DataFrame(
            {"ic": ic},
            index=pd.RangeIndex(len(ic), name="step_id"),
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": fc},
            index=pd.RangeIndex(len(fc), name="step_id"),
        )

        self.ic_side_ = pd.Series(
            ic_side,
            index=self.ic_list_.index,
            name="side"
        )

        self.fc_side_ = pd.Series(
            fc_side,
            index=self.fc_list_.index,
            name="side"
        )

        return self
