"""
GsdRf algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdRf(GsdSklearnModel):
    """Reusable RF model for Gait Sequence Detection."""

    expected_model_name = "rf"
