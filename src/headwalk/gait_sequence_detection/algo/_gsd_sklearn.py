"""
Shared sklearn-based implementation for portable GSD artifacts.

Portable artifact layout:

    portable_artifacts/<ModelName>/
        config.json
        metadata.json
        model.joblib
        preprocessing.json

The saved sklearn model receives handcrafted window-level features as input.
This wrapper can either:
1. predict from an already computed feature table;
2. build sliding windows from a preprocessed IMU signal and compute features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .base import BaseGsdAlgorithm, GsdPredictionResult


DEFAULT_LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}


class GsdSklearnModel(BaseGsdAlgorithm):
    """
    Base class for GSD models trained with sklearn-compatible estimators.

    If artifact_dir is not provided, the class automatically searches in:

        src/headwalk/gait_sequence_detection/models/portable_artifacts/<ClassName>

    Example
    -------
    GsdSvm() loads:

        models/portable_artifacts/GsdSvm/
    """

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

        self.config_path = self.artifact_dir / "config.json"
        self.metadata_path = self.artifact_dir / "metadata.json"
        self.preprocessing_path = self.artifact_dir / "preprocessing.json"

        self._validate_json_files()

        self.config = self._load_json(self.config_path)
        self.metadata = self._load_json(self.metadata_path)
        self.preprocessing = self._load_json(self.preprocessing_path)

        model_filename = self.config.get("model_filename", "model.joblib")
        self.model_path = self.artifact_dir / model_filename

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        artifact_format = self.config.get("artifact_format", "joblib")
        if artifact_format != "joblib":
            raise ValueError(
                f"{self.__class__.__name__} expects a joblib sklearn artifact, "
                f"but artifact_format={artifact_format!r}."
            )

        self._validate_model_identity()

        self.model = joblib.load(self.model_path)

        self.selected_features = self.config.get("selected_features", [])
        if not self.selected_features:
            raise ValueError(
                f"No selected_features found in config.json: {self.config_path}"
            )

        self.fs = float(
            self.preprocessing.get(
                "sampling_rate_hz",
                self.preprocessing.get("sampling_frequency_hz", 100.0),
            )
        )

        self.channel_names = self.preprocessing.get(
            "expected_raw_columns",
            ["acc_VT", "acc_ML", "acc_AP", "gyr_VT", "gyr_ML", "gyr_AP"],
        )

        self.label_name_map = (
            self.metadata.get("label_map")
            or self.metadata.get("label_name_map")
            or DEFAULT_LABEL_NAME_MAP
        )

        self.label_name_map = {
            int(k) if str(k).lstrip("-").isdigit() else k: v
            for k, v in self.label_name_map.items()
        }

        self.window_length_s = float(
            self.preprocessing.get(
                "window_duration_s",
                self.preprocessing.get("window_length_s", 1.0),
            )
        )

        self.overlap_fraction = float(
            self.preprocessing.get("overlap_fraction", 0.5)
        )

        self.window_config = self.config.get(
            "window_config",
            self.preprocessing.get("window_config"),
        )

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_json_files(self) -> None:
        missing_files = [
            path for path in [
                self.config_path,
                self.metadata_path,
                self.preprocessing_path,
            ]
            if not path.exists()
        ]

        if missing_files:
            missing_text = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                "Incomplete GSD portable artifact folder. Missing files:\n"
                f"{missing_text}"
            )

    def _validate_model_identity(self) -> None:
        if self.expected_model_name is None:
            return

        source_model_name = str(
            self.config.get("source_model_name", "")
        ).lower()

        if source_model_name != self.expected_model_name:
            raise ValueError(
                "The loaded artifact does not match this algorithm class. "
                f"Expected source_model_name={self.expected_model_name!r}, "
                f"found {source_model_name!r}."
            )

    def _build_sliding_windows(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """
        Create overlapping windows from a preprocessed IMU signal.

        The signal must already be aligned and filtered consistently with the
        training pipeline.
        """

        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 2:
            raise ValueError("signal must be a 2D array with shape samples x channels.")

        if signal.shape[1] != len(self.channel_names):
            raise ValueError(
                "The signal channel count does not match the artifact schema. "
                f"Signal columns={signal.shape[1]}, expected={len(self.channel_names)}."
            )

        window_size = int(round(self.window_length_s * self.fs))
        step_size = int(round(window_size * (1.0 - self.overlap_fraction)))
        step_size = max(step_size, 1)

        if len(signal) < window_size:
            raise ValueError(
                "Signal is shorter than one GSD window. "
                f"Signal samples={len(signal)}, window_size={window_size}."
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

    def _compute_feature_table_from_signal(self, signal: np.ndarray) -> pd.DataFrame:
        """
        Compute window-level features from a preprocessed IMU signal.

        The feature implementation is imported here, not at module import time,
        so feature-table prediction remains usable even when optional feature
        dependencies are not needed immediately.
        """

        from ..features import compute_window_features

        windows, metadata_rows = self._build_sliding_windows(signal)
        rows = []

        for window, metadata_row in zip(windows, metadata_rows):
            feature_row = compute_window_features(
                window=window,
                fs=self.fs,
                channel_names=self.channel_names,
            )
            feature_row.update(metadata_row)
            rows.append(feature_row)

        return pd.DataFrame(rows)

    def _select_model_input(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """
        Select and order exactly the features expected by the saved model.
        """

        missing_features = [
            feature for feature in self.selected_features
            if feature not in feature_table.columns
        ]

        if missing_features:
            preview = ", ".join(missing_features[:20])
            raise KeyError(
                "The feature table does not contain all selected features. "
                f"Missing {len(missing_features)} feature(s). "
                f"First missing values: {preview}"
            )

        return feature_table[self.selected_features].copy()

    def _predict_probabilities_if_available(
        self,
        X: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Return class probabilities when the model exposes predict_proba().
        """

        if not hasattr(self.model, "predict_proba"):
            return {}

        probabilities = self.model.predict_proba(X)

        output = {}

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            output["gsd_probability_static"] = probabilities[:, 0]
            output["gsd_probability_walking"] = probabilities[:, 1]

        return output

    def predict_from_feature_table(
        self,
        feature_table: pd.DataFrame,
    ) -> GsdPredictionResult:
        """
        Predict static/walking labels from an already computed feature table.
        """

        X = self._select_model_input(feature_table)

        y_pred = np.asarray(self.model.predict(X)).astype(int)

        predictions = feature_table.copy()
        predictions["gsd_prediction"] = y_pred
        predictions["gsd_prediction_name"] = [
            self.label_name_map.get(int(label), "unknown")
            for label in y_pred
        ]

        probability_columns = self._predict_probabilities_if_available(X)
        for column_name, values in probability_columns.items():
            predictions[column_name] = values

        predictions["gsd_model_class"] = self.config.get(
            "class_name",
            self.__class__.__name__,
        )
        predictions["gsd_source_model_name"] = self.config.get("source_model_name")
        predictions["gsd_window_config"] = self.window_config
        predictions["gsd_artifact_dir"] = str(self.artifact_dir)

        return GsdPredictionResult(
            predictions=predictions,
            artifact_dir=self.artifact_dir,
            metadata={
                "config": self.config,
                "metadata": self.metadata,
                "preprocessing": self.preprocessing,
            },
        )

    def predict_from_signal(self, signal, time=None) -> GsdPredictionResult:
        """
        Predict GSD labels from a preprocessed IMU signal.

        Important
        ---------
        The signal must already be in the same aligned convention used during
        training. This class does not redo dataset-specific gravity alignment,
        pitch-roll correction, filtering or gap filling.
        """

        feature_table = self._compute_feature_table_from_signal(signal)

        if time is not None:
            feature_table["input_time_available"] = True

        return self.predict_from_feature_table(feature_table)
