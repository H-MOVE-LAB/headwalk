"""
Base class for Initial Contacts Detection (ICD) algorithms.

Author: Paolo Tasca
Date: 26-Jan-2026
"""

from typing import Any
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
        """
        Detect initial contacts in the input data.

        Parameters
        ----------
        data : pd.DataFrame
            Raw IMU data of a single gait sequence (columns can be x/y/z or named signals)
        sampling_rate_hz : float
            Sampling rate of the IMU data in Hz

        Returns
        -------
        self : instance
            Sets `self.ic_list_` with detected ICs as a DataFrame
        """
        raise NotImplementedError("Subclasses must implement the detect method")
