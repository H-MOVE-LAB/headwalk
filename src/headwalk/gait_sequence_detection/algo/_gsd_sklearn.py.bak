"""
Shared sklearn-based implementation for portable GSD artifacts.

The trained model itself is saved as model.joblib. The surrounding artifact
folder also stores the selected feature list and metadata needed to reproduce
the exact input schema expected by the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .base import BaseGsdAlgorithm, GsdPredictionResult
from ..features import compute_window_features


DEFAULT_LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}


class GsdSklearnModel(BaseGsdAlgorithm):
    """
    Base class for GSD models trained with sklearn pipelines.

    Parameters
    ----------
    artifact_dir:
        Folder containing model.joblib, selected_features.json, metadata.json
        and preprocessing.json.
    """

    expected_model_name: str | None = None

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.model_path = self.artifact_dir / "model.joblib"
        self.selected_features_path = self.artifact_dir / "selected_features.json"
        self.metadata_path = self.artifact_dir / "metadata.json"
        self.preprocessing_path = self.artifact_dir / "preprocessing.json"

        self._validate_artifact_files()

        self.model = joblib.load(self.model_path)
        self.selected_features = self._load_json(self.selected_features_path)
        self.metadata = self._load_json(self.metadata_path)
        self.preprocessing = self._load_json(self.preprocessing_path)

        if self.expected_model_name is not None:
            artifact_model_name = str(self.metadata.get("model", "")).lower()
            if artifact_model_name != self.expected_model_name:
                raise ValueError(
                    "The loaded artifact does not match this algorithm class. "
                    f"Expected {self.expected_model_name}, found {artifact_model_name}."
                )

        self.fs = float(self.preprocessing.get("sampling_frequency_hz", 100))
        self.channel_names = self.preprocessing.get(
            "channel_names",
            ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"],
        )
        self.label_name_map = self.metadata.get("label_name_map", DEFAULT_LABEL_NAME_MAP)

        # JSON object keys are loaded as strings. Normalize them to int whenever possible.
        self.label_name_map = {
            int(k) if str(k).lstrip("-").isdigit() else k: v
            for k, v in self.label_name_map.items()
        }

        self.window_length_s = float(
            self.metadata.get(
                "window_length_s",
                self.preprocessing.get("window_length_s", 1.0),
            )
        )
        self.overlap_fraction = float(
            self.metadata.get(
                "overlap_fraction",
                self.preprocessing.get("overlap_fraction", 0.5),
            )
        )

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_artifact_files(self) -> None:
        missing_files = [
            path for path in [
                self.model_path,
                self.selected_features_path,
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

    def _build_sliding_windows(self, signal: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Create 50%-overlap windows from a preprocessed IMU signal."""

        signal = np.asarray(signal, dtype=float)

        if signal.ndim != 2:
            raise ValueError("signal must be a 2D array with shape samples x channels.")

        if signal.shape[1] != len(self.channel_names):
            raise ValueError(
                "The signal channel count does not match the artifact channel schema. "
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
        """Compute the trained feature set from a preprocessed IMU signal."""

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
        """Select and order the exact features expected by the saved model."""

        missing_features = [
            feature for feature in self.selected_features
            if feature not in feature_table.columns
        ]

        if missing_features:
            preview = ", ".join(missing_features[:20])
            raise KeyError(
                "The feature table does not contain all selected features. "
                f"Missing {len(missing_features)} feature(s). First missing values: {preview}"
            )

        return feature_table[self.selected_features].copy()

    def predict_from_feature_table(self, feature_table: pd.DataFrame) -> GsdPredictionResult:
        """Predict GSD labels from an already computed feature table."""

        X = self._select_model_input(feature_table)
        y_pred = self.model.predict(X).astype(int)

        predictions = feature_table.copy()
        predictions["gsd_prediction"] = y_pred
        predictions["gsd_prediction_name"] = [
            self.label_name_map.get(int(label), "unknown")
            for label in y_pred
        ]
        predictions["gsd_model"] = self.metadata.get("model", self.expected_model_name)
        predictions["gsd_window_config"] = self.metadata.get("window_config")
        predictions["gsd_artifact_dir"] = str(self.artifact_dir)

        return GsdPredictionResult(
            predictions=predictions,
            artifact_dir=self.artifact_dir,
            metadata=self.metadata,
        )

    def predict_from_signal(self, signal, time=None) -> GsdPredictionResult:
        """
        Predict GSD labels from a preprocessed IMU signal.

        The signal must already be in the same AP/ML/VT aligned convention used
        during training. This class deliberately does not redo dataset-specific
        gravity alignment, pitch-roll correction or filtering.
        """

        feature_table = self._compute_feature_table_from_signal(signal)

        if time is not None:
            # The current implementation keeps model windows based on sampling
            # frequency. The time vector is accepted only for API compatibility.
            feature_table["input_time_available"] = True

        return self.predict_from_feature_table(feature_table)
