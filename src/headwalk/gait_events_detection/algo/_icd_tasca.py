from typing import Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly
from scipy.stats import zscore
from typing_extensions import Self, Unpack
import tensorflow as tf

from .base import BaseIcDetector


class IcdTasca(BaseIcDetector):
    """
    Initial Contact (IC) detection based on a Temporal Convolutional Network
    (Tasca et al., 2025).
    """

    def __init__(
        self,
        model_path: str,
        window_length: int = 200,  # 2 s @ 100 Hz
        n_channels: int = 6,
        peak_height: float = 0.05,
        peak_prominence: float = 0.04,
        min_peak_distance_s: float = 0.4,
        f_cutoff: float = 5.0,
        f_stop: float = 10.0,
    ) -> None:
        self.model_path = model_path
        self.window_length = window_length
        self.n_channels = n_channels
        self.peak_height = peak_height
        self.peak_prominence = peak_prominence
        self.min_peak_distance_s = min_peak_distance_s
        self.f_cutoff = f_cutoff
        self.f_stop = f_stop

        self.model = tf.keras.models.load_model(
            model_path, compile=False
        )

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_columns: list[str],
        gyr_columns: list[str],
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        Detect Initial Contacts (IC) using a pre-trained TCN model.
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # --------------------------------------------------
        # 1. SIGNAL SELECTION
        # --------------------------------------------------
        X = data[acc_columns + gyr_columns].to_numpy()

        if X.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {X.shape[1]}"
            )

        # --------------------------------------------------
        # 2. RESAMPLING 128 → 100 Hz
        # --------------------------------------------------
        if sampling_rate_hz != 100:
            X = resample_poly(X, up=100, down=int(sampling_rate_hz), axis=0)
            fs = 100.0
        else:
            fs = sampling_rate_hz

        # --------------------------------------------------
        # 3. LOW-PASS FILTERING (Tasca)
        # --------------------------------------------------
        X = _lowpass_filter_tasca(
            X,
            fs=fs,
            f_cutoff=self.f_cutoff,
            f_stop=self.f_stop,
        )

        # --------------------------------------------------
        # 4. Z-SCORE STANDARDIZATION
        # --------------------------------------------------
        X = zscore(X, axis=0, nan_policy="omit")

        # --------------------------------------------------
        # 5. WINDOWING (2 s)
        # --------------------------------------------------
        win = self.window_length
        n_windows = X.shape[0] // win

        X = X[: n_windows * win]
        X_win = X.reshape(
            n_windows, win, self.n_channels
        )

        # --------------------------------------------------
        # 6. MODEL INFERENCE
        # --------------------------------------------------
        prob = self.model.predict(X_win, verbose=0)[:, :, 0]

        # --------------------------------------------------
        # 7. PEAK DETECTION (IC)
        # --------------------------------------------------
        min_dist_samples = int(self.min_peak_distance_s * fs)

        ic_idx = []

        for i in range(n_windows):
            p = prob[i]

            peaks, _ = find_peaks(
                p,
                height=self.peak_height,
                prominence=self.peak_prominence,
                distance=min_dist_samples,
            )

            if p[0] >= self.peak_height:
                peaks = np.insert(peaks, 0, 0)
            if p[-1] >= self.peak_height:
                peaks = np.append(peaks, win - 1)

            ic_idx.extend(peaks + i * win)

        ic_idx = np.asarray(ic_idx, dtype=int)

        # --------------------------------------------------
        # 8. STORE RESULTS
        # --------------------------------------------------
        self.ic_list_ = pd.DataFrame(
            {"ic": ic_idx},
            index=pd.RangeIndex(len(ic_idx), name="step_id"),
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": np.full(len(ic_idx), np.nan)},
            index=self.ic_list_.index,
        )

        return self
