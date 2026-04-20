# himu_pd/gait_events_detection/algo/base.py
"""
Base classes for gait event detection and laterality classification.

Author: Paolo Tasca (extended)
Date: 26-Jan-2026
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class BaseIcDetector(Algorithm):
    """Base class for IC detectors.

    Child classes should implement the `detect` method, which sets:
        - self.data
        - self.sampling_rate_hz
        - self.ic_list_ : pd.DataFrame with column "ic"
    """

    # Input attributes
    data: pd.DataFrame
    sampling_rate_hz: float

    # Output
    ic_list_: pd.DataFrame

    def detect(
        self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]
    ) -> Self:
        """Detect initial contacts in the input data."""
        raise NotImplementedError("Subclasses must implement the detect method")


class BaseLrClassifier(Algorithm):
    """Base class for gait-event laterality classifiers.

    This block is meant to be applied *after* an event detector:
    it receives the raw IMU signals and the indices of already-detected ICs
    (and optionally FCs), and predicts Left/Right for each event.

    Child classes should implement `predict`, which sets:
        - self.data
        - self.sampling_rate_hz
        - self.ic_side_ : pd.Series of 'L'/'R' indexed by step_id
        - self.fc_side_ : pd.Series of 'L'/'R' indexed by step_id (optional, can be empty)

    Notes
    -----
    - This base mirrors the output style of BaseIcDetector/IcdTcn:
      you expose `ic_list_`, `fc_list_` (if applicable) and `ic_side_`, `fc_side_`.
    """

    # Inputs
    data: pd.DataFrame
    sampling_rate_hz: float

    # Inputs (event indices)
    ic_list_: pd.DataFrame
    fc_list_: pd.DataFrame

    # Outputs (laterality per event)
    ic_side_: pd.Series
    fc_side_: pd.Series

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        ic: np.ndarray,
        fc: Optional[np.ndarray] = None,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Predict laterality for provided IC (and optional FC) indices."""
        raise NotImplementedError("Subclasses must implement the predict method")