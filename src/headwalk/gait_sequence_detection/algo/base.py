"""
Base interfaces for Gait Sequence Detection algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class GsdPredictionResult:
    """Container returned by a GSD algorithm."""

    predictions: pd.DataFrame
    artifact_dir: Path
    metadata: dict[str, Any]


class BaseGsdAlgorithm(ABC):
    """Minimal reusable interface for GSD algorithms."""

    @abstractmethod
    def predict_from_signal(self, signal, time=None) -> GsdPredictionResult:
        """Predict static/walking windows from a preprocessed IMU signal."""

    @abstractmethod
    def predict_from_feature_table(self, feature_table: pd.DataFrame) -> GsdPredictionResult:
        """Predict static/walking windows from an already computed feature table."""
