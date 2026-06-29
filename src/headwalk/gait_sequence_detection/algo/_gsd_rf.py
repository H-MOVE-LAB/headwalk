"""
GsdRandomForest algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdRandomForest(GsdSklearnModel):
    """Reusable Random Forest model for Gait Sequence Detection."""

    expected_model_name = "rf"


# Backward-compatible alias.
GsdRf = GsdRandomForest
