"""
Base interfaces for Gait Sequence Detection algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tpcp import Algorithm
except ImportError:
    class Algorithm:
        """Fallback when tpcp is not installed."""
        pass


@dataclass
class GsdDetectionResult:
    """Container returned by a GSD algorithm."""

    window_detections: pd.DataFrame
    gs_list: pd.DataFrame
    artifact_dir: Path
    metadata: dict[str, Any]


class BaseGsdAlgorithm(Algorithm, ABC):
    """
    Base class for Gait Sequence Detection algorithms.

    Public API:
        algorithm.detect(...)

    Standard output attributes:
        self.window_detections_
        self.gs_list_

    self.gs_list_ always contains:
        start_s
        end_s
        start_samples
        end_samples
    """

    def detect(
        self,
        data=None,
        *,
        sampling_rate_hz: float | None = None,
        feature_table: pd.DataFrame | None = None,
        acc_columns: list[str] | None = None,
        gyr_columns: list[str] | None = None,
        time_column: str | None = None,
        min_sequence_duration_s: float = 0.0,
        merge_gap_s: float = 0.0,
        **kwargs,
    ):
        """
        Run complete GSD detection.

        This method is shared by all GSD algorithms. Child classes specialize
        only _detect_windows().
        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        self.window_detections_ = self._detect_windows(
            data=data,
            sampling_rate_hz=sampling_rate_hz,
            feature_table=feature_table,
            acc_columns=acc_columns,
            gyr_columns=gyr_columns,
            time_column=time_column,
            **kwargs,
        )

        self.gs_list_ = self._build_gs_list(
            window_detections=self.window_detections_,
            min_sequence_duration_s=min_sequence_duration_s,
            merge_gap_s=merge_gap_s,
        )

        self.result_ = GsdDetectionResult(
            window_detections=self.window_detections_,
            gs_list=self.gs_list_,
            artifact_dir=self.artifact_dir,
            metadata=self.get_detection_metadata(),
        )

        return self

    @abstractmethod
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
        """Return window-level GSD detections."""

    def get_detection_metadata(self) -> dict[str, Any]:
        return {
            "algorithm_class": self.__class__.__name__,
        }

    @staticmethod
    def _build_gs_list(
        window_detections: pd.DataFrame,
        *,
        min_sequence_duration_s: float = 0.0,
        merge_gap_s: float = 0.0,
    ) -> pd.DataFrame:
        """
        Convert detected walking windows into gait-sequence intervals.
        """

        output_columns = ["start_s", "end_s", "start_samples", "end_samples"]

        if window_detections.empty:
            return pd.DataFrame(columns=output_columns)

        required_cols = [
            "gsd_label",
            "window_start_sample",
            "window_end_sample",
            "window_start_time",
            "window_end_time",
        ]

        missing = [col for col in required_cols if col not in window_detections.columns]

        if missing:
            raise KeyError(
                "Cannot build gs_list_. Missing required columns: "
                f"{missing}"
            )

        walking_df = window_detections[
            window_detections["gsd_label"].astype(int) == 1
        ].copy()

        if walking_df.empty:
            return pd.DataFrame(columns=output_columns)

        walking_df = walking_df.sort_values("window_start_sample")

        sequences = []

        current_start_sample = int(walking_df.iloc[0]["window_start_sample"])
        current_end_sample = int(walking_df.iloc[0]["window_end_sample"])
        current_start_time = float(walking_df.iloc[0]["window_start_time"])
        current_end_time = float(walking_df.iloc[0]["window_end_time"])

        for _, row in walking_df.iloc[1:].iterrows():
            start_sample = int(row["window_start_sample"])
            end_sample = int(row["window_end_sample"])
            start_time = float(row["window_start_time"])
            end_time = float(row["window_end_time"])

            gap_s = start_time - current_end_time

            if start_sample <= current_end_sample or gap_s <= merge_gap_s:
                current_end_sample = max(current_end_sample, end_sample)
                current_end_time = max(current_end_time, end_time)
            else:
                duration_s = current_end_time - current_start_time

                if duration_s >= min_sequence_duration_s:
                    sequences.append({
                        "start_s": current_start_time,
                        "end_s": current_end_time,
                        "start_samples": current_start_sample,
                        "end_samples": current_end_sample,
                    })

                current_start_sample = start_sample
                current_end_sample = end_sample
                current_start_time = start_time
                current_end_time = end_time

        duration_s = current_end_time - current_start_time

        if duration_s >= min_sequence_duration_s:
            sequences.append({
                "start_s": current_start_time,
                "end_s": current_end_time,
                "start_samples": current_start_sample,
                "end_samples": current_end_sample,
            })

        return pd.DataFrame(sequences, columns=output_columns)
