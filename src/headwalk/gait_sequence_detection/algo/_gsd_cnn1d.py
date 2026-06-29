"""
GsdCnn1D algorithm wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseGsdAlgorithm


DEFAULT_LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}


class GsdCnn1D(BaseGsdAlgorithm):
    """
    Reusable 1D-CNN model for Gait Sequence Detection.

    This child class specializes _detect_windows() by using raw accelerometer
    windows instead of handcrafted feature tables.
    """

    def __init__(self, artifact_dir: str | Path | None = None):
        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).resolve().parents[1]
                / "models"
                / "portable_artifacts"
                / self.__class__.__name__
            )

        self.artifact_dir = Path(artifact_dir)

        self.config = self._load_json(self.artifact_dir / "config.json")
        self.metadata = self._load_json(self.artifact_dir / "metadata.json")
        self.preprocessing = self._load_json(self.artifact_dir / "preprocessing.json")

        if self.config.get("artifact_format") != "keras":
            raise ValueError("GsdCnn1D expects a Keras artifact.")

        self.model_path = self.artifact_dir / self.config.get(
            "model_filename",
            "model.keras",
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.fs = float(self.preprocessing.get("sampling_rate_hz", 100.0))
        self.window_length_s = float(self.preprocessing.get("window_duration_s", 5.0))
        self.overlap_fraction = float(self.preprocessing.get("overlap_fraction", 0.5))
        self.window_config = self.config.get("window_config")

        self.raw_channel_names = self.preprocessing.get(
            "expected_raw_columns",
            ["acc_vt", "acc_ml", "acc_ap", "gyr_vt", "gyr_ml", "gyr_ap"],
        )

        self.acc_columns_default = self.raw_channel_names[:3]

        self.label_name_map = (
            self.metadata.get("label_map")
            or self.metadata.get("label_name_map")
            or DEFAULT_LABEL_NAME_MAP
        )

        self.label_name_map = {
            int(k) if str(k).lstrip("-").isdigit() else k: v
            for k, v in self.label_name_map.items()
        }

        self.model = self._load_keras_model()
        self.channel_mean, self.channel_std = self._load_channel_normalization()

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_dataframe_columns(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out.columns = [
            str(col).strip().replace(" ", "_").replace("-", "_").lower()
            for col in out.columns
        ]
        return out

    def _load_keras_model(self):
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to use GsdCnn1D. "
                "Install tensorflow in the active environment."
            ) from exc

        return keras.models.load_model(str(self.model_path))

    def _load_channel_normalization(self):
        cnn_meta = self.preprocessing.get("cnn_preprocessing_metadata", {})

        channel_mean = cnn_meta.get("channel_mean", 0.0)
        channel_std = cnn_meta.get("channel_std", 1.0)

        channel_mean = np.asarray(channel_mean, dtype=np.float32)
        channel_std = np.asarray(channel_std, dtype=np.float32)

        channel_std = np.where(channel_std < 1e-8, 1.0, channel_std)

        return channel_mean, channel_std

    def _data_to_acc_array(
        self,
        data,
        *,
        acc_columns: list[str] | None,
    ) -> np.ndarray:
        """Convert input data to raw accelerometer samples x channels."""

        if isinstance(data, pd.DataFrame):
            data = self._normalize_dataframe_columns(data)

            columns = [
                str(col).strip().replace(" ", "_").replace("-", "_").lower()
                for col in (acc_columns or self.acc_columns_default)
            ]

            missing = [col for col in columns if col not in data.columns]
            if missing:
                raise KeyError(
                    f"Missing accelerometer columns: {missing}. "
                    f"Available columns: {list(data.columns)}"
                )

            return data[columns].to_numpy(dtype=np.float32)

        signal = np.asarray(data, dtype=np.float32)

        if signal.ndim != 2:
            raise ValueError("Input signal must be samples x channels.")

        if signal.shape[1] > 3:
            signal = signal[:, :3]

        return signal

    def _build_raw_windows(
        self,
        acc_signal: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Create overlapping raw accelerometer windows."""

        acc_signal = np.asarray(acc_signal, dtype=np.float32)

        if acc_signal.ndim != 2:
            raise ValueError("Accelerometer signal must be samples x channels.")

        window_size = int(round(self.window_length_s * self.fs))
        step_size = int(round(window_size * (1.0 - self.overlap_fraction)))
        step_size = max(step_size, 1)

        if len(acc_signal) < window_size:
            raise ValueError(
                f"Signal is shorter than one CNN GSD window: {len(acc_signal)} < {window_size}."
            )

        windows = []
        rows = []

        for start in range(0, len(acc_signal) - window_size + 1, step_size):
            end = start + window_size

            windows.append(acc_signal[start:end])

            rows.append({
                "window_start_sample": start,
                "window_end_sample": end,
                "window_start_time": start / self.fs,
                "window_end_time": end / self.fs,
                "window_duration_s": window_size / self.fs,
            })

        return np.asarray(windows, dtype=np.float32), rows

    def _normalize_windows(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.channel_mean) / self.channel_std

    def _detect_from_raw_windows(
        self,
        windows: np.ndarray,
        metadata_rows: list[dict[str, float]],
    ) -> pd.DataFrame:
        """Run CNN inference and return window-level detections."""

        windows = self._normalize_windows(windows)

        probabilities = self.model.predict(windows, verbose=0)

        detected_labels = np.argmax(probabilities, axis=1).astype(int)

        window_detections = pd.DataFrame(metadata_rows)
        window_detections["gsd_label"] = detected_labels
        window_detections["gsd_label_name"] = [
            self.label_name_map.get(int(label), "unknown")
            for label in detected_labels
        ]

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            window_detections["gsd_probability_static"] = probabilities[:, 0]
            window_detections["gsd_probability_walking"] = probabilities[:, 1]

        window_detections["gsd_model_class"] = self.config.get(
            "class_name",
            self.__class__.__name__,
        )
        window_detections["gsd_source_model_name"] = self.config.get("source_model_name")
        window_detections["gsd_window_config"] = self.window_config
        window_detections["gsd_artifact_dir"] = str(self.artifact_dir)

        return window_detections

    def _detect_windows(
        self,
        data,
        *,
        sampling_rate_hz: float | None,
        feature_table: pd.DataFrame | None,
        acc_columns: list[str] | None,
        gyr_columns: list[str] | None,
        time_column: str | None,
        **kwargs,
    ) -> pd.DataFrame:
        """CNN specialization of BaseGsdAlgorithm.detect()."""

        if feature_table is not None:
            raise ValueError(
                "GsdCnn1D uses raw accelerometer signals, not handcrafted feature tables."
            )

        if data is None:
            raise ValueError("GsdCnn1D requires raw accelerometer data.")

        if sampling_rate_hz is not None:
            self.fs = float(sampling_rate_hz)

        acc_signal = self._data_to_acc_array(
            data,
            acc_columns=acc_columns,
        )

        windows, metadata_rows = self._build_raw_windows(acc_signal)

        return self._detect_from_raw_windows(windows, metadata_rows)

    def get_detection_metadata(self) -> dict[str, Any]:
        return {
            "algorithm_class": self.__class__.__name__,
            "config": self.config,
            "metadata": self.metadata,
            "preprocessing": self.preprocessing,
        }
