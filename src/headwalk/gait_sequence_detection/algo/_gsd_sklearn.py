"""
Shared sklearn-based implementation for portable GSD artifacts.

Public method:
    detect(...)

Portable artifact layout:
    portable_artifacts/<ModelName>/
        config.json
        metadata.json
        model.joblib
        preprocessing.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .base import BaseGsdAlgorithm


DEFAULT_LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}


class GsdSklearnModel(BaseGsdAlgorithm):
    """Base class for GSD models trained with sklearn-compatible estimators."""

    expected_model_name: str | None = None

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

        self._validate_model_identity()

        self.model_path = self.artifact_dir / self.config.get(
            "model_filename",
            "model.joblib",
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.config.get("artifact_format", "joblib") != "joblib":
            raise ValueError(
                f"{self.__class__.__name__} expects a joblib artifact."
            )

        # The sklearn estimator is intentionally loaded lazily.
        #
        # Initializing the algorithm should be lightweight. The joblib model is
        # loaded only when detection is actually executed.
        self.model = None
        self.model_load_strategy_ = None

        self.selected_features = self.config.get("selected_features", [])
        if not self.selected_features:
            raise ValueError(
                f"No selected_features found in {self.artifact_dir / 'config.json'}"
            )

        self.fs = float(self.preprocessing.get("sampling_rate_hz", 100.0))

        self.raw_channel_names = self.preprocessing.get(
            "expected_raw_columns",
            ["acc_vt", "acc_ml", "acc_ap", "gyr_vt", "gyr_ml", "gyr_ap"],
        )

        self.feature_channel_names = self.preprocessing.get(
            "feature_channel_names",
            ["acc_VT", "acc_ML", "acc_AP", "gyr_VT", "gyr_ML", "gyr_AP"],
        )

        # Backward-compatible attribute name.
        self.channel_names = self.raw_channel_names

        self.window_length_s = float(self.preprocessing.get("window_duration_s", 1.0))
        self.overlap_fraction = float(self.preprocessing.get("overlap_fraction", 0.5))
        self.window_config = self.config.get("window_config")

        self.label_name_map = (
            self.metadata.get("label_map")
            or self.metadata.get("label_name_map")
            or DEFAULT_LABEL_NAME_MAP
        )

        self.label_name_map = {
            int(k) if str(k).lstrip("-").isdigit() else k: v
            for k, v in self.label_name_map.items()
        }

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_model_identity(self) -> None:
        if self.expected_model_name is None:
            return

        source_model_name = str(self.config.get("source_model_name", "")).lower()

        if source_model_name != self.expected_model_name:
            raise ValueError(
                "The loaded artifact does not match this algorithm class. "
                f"Expected {self.expected_model_name!r}, found {source_model_name!r}."
            )

    def _ensure_model_loaded(self):
        """
        Load the sklearn/joblib estimator only when it is actually used.
        """

        if self.model is None:
            self.model = joblib.load(self.model_path)
            self.model_load_strategy_ = "joblib_model"

        return self.model

    @staticmethod
    def _normalize_dataframe_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Force dataframe column names to lowercase snake-case."""

        out = data.copy()
        out.columns = [
            str(col).strip().replace(" ", "_").replace("-", "_").lower()
            for col in out.columns
        ]

        return out

    def _data_to_array(
        self,
        data,
        *,
        acc_columns: list[str] | None,
        gyr_columns: list[str] | None,
    ) -> np.ndarray:
        """Convert input IMU signal to samples x channels array."""

        if isinstance(data, pd.DataFrame):
            data = self._normalize_dataframe_columns(data)

            if acc_columns is not None or gyr_columns is not None:
                columns = [
                    str(col).strip().replace(" ", "_").replace("-", "_").lower()
                    for col in ((acc_columns or []) + (gyr_columns or []))
                ]
            else:
                columns = self.raw_channel_names

            missing = [col for col in columns if col not in data.columns]
            if missing:
                raise KeyError(
                    f"Missing input signal columns: {missing}. "
                    f"Available columns: {list(data.columns)}"
                )

            return data[columns].to_numpy(dtype=float)

        signal = np.asarray(data, dtype=float)

        if signal.ndim != 2:
            raise ValueError("Input signal must be samples x channels.")

        return signal

    def _build_sliding_windows(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Create overlapping windows from a preprocessed IMU signal."""

        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 2:
            raise ValueError("Input signal must be samples x channels.")

        if signal.shape[1] != len(self.raw_channel_names):
            raise ValueError(
                "The signal channel count does not match the artifact schema. "
                f"Signal columns={signal.shape[1]}, expected={len(self.raw_channel_names)}."
            )

        window_size = int(round(self.window_length_s * self.fs))
        step_size = int(round(window_size * (1.0 - self.overlap_fraction)))
        step_size = max(step_size, 1)

        if len(signal) < window_size:
            raise ValueError(
                f"Signal is shorter than one GSD window: {len(signal)} < {window_size}."
            )

        windows = []
        rows = []

        for start in range(0, len(signal) - window_size + 1, step_size):
            end = start + window_size

            windows.append(signal[start:end])

            rows.append({
                "window_start_sample": start,
                "window_end_sample": end,
                "window_start_time": start / self.fs,
                "window_end_time": end / self.fs,
                "window_duration_s": window_size / self.fs,
            })

        return np.asarray(windows), rows

    def _detect_from_signal(self, signal: np.ndarray) -> pd.DataFrame:
        """
        Build handcrafted features from signal and run estimator.
        """

        from ..features import compute_window_features

        windows, metadata_rows = self._build_sliding_windows(signal)

        rows = []

        for window, metadata_row in zip(windows, metadata_rows):
            feature_row = compute_window_features(
                window=window,
                fs=self.fs,
                channel_names=self.feature_channel_names,
            )
            feature_row.update(metadata_row)
            rows.append(feature_row)

        feature_table = pd.DataFrame(rows)

        return self._detect_from_feature_table(feature_table)

    def _ensure_window_metadata(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """Ensure that a feature table has window timing columns."""

        table = feature_table.copy()

        window_size = int(round(self.window_length_s * self.fs))
        step_size = int(round(window_size * (1.0 - self.overlap_fraction)))
        step_size = max(step_size, 1)

        if "window_start_sample" not in table.columns:
            table["window_start_sample"] = np.arange(len(table)) * step_size

        if "window_end_sample" not in table.columns:
            table["window_end_sample"] = table["window_start_sample"] + window_size

        if "window_start_time" not in table.columns:
            table["window_start_time"] = table["window_start_sample"] / self.fs

        if "window_end_time" not in table.columns:
            table["window_end_time"] = table["window_end_sample"] / self.fs

        if "window_duration_s" not in table.columns:
            table["window_duration_s"] = self.window_length_s

        return table

    def _select_model_input(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """Select and order exactly the features expected by the saved model."""

        missing_features = [
            feature for feature in self.selected_features
            if feature not in feature_table.columns
        ]

        if missing_features:
            preview = ", ".join(missing_features[:20])
            raise KeyError(
                "The feature table does not contain all selected features. "
                f"Missing {len(missing_features)} feature(s): {preview}"
            )

        return feature_table[self.selected_features].copy()

    def _detect_from_feature_table(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """
        Run estimator inference and return window-level detections.
        """

        feature_table = self._ensure_window_metadata(feature_table)

        X = self._select_model_input(feature_table)

        model = self._ensure_model_loaded()

        detected_labels = np.asarray(model.predict(X)).astype(int)

        window_detections = feature_table.copy()
        window_detections["gsd_label"] = detected_labels
        window_detections["gsd_label_name"] = [
            self.label_name_map.get(int(label), "unknown")
            for label in detected_labels
        ]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
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
        """
        Sklearn specialization of BaseGsdAlgorithm.detect().
        """

        if sampling_rate_hz is not None:
            self.fs = float(sampling_rate_hz)

        if feature_table is not None:
            return self._detect_from_feature_table(feature_table)

        if data is None:
            raise ValueError("Either data or feature_table must be provided.")

        signal = self._data_to_array(
            data,
            acc_columns=acc_columns,
            gyr_columns=gyr_columns,
        )

        return self._detect_from_signal(signal)

    def get_detection_metadata(self) -> dict[str, Any]:
        return {
            "algorithm_class": self.__class__.__name__,
            "config": self.config,
            "metadata": self.metadata,
            "preprocessing": self.preprocessing,
            "model_load_strategy": self.model_load_strategy_,
        }
