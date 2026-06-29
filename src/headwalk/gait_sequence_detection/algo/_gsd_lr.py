"""
GsdLogisticRegression algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdLogisticRegression(GsdSklearnModel):
    """Reusable Logistic Regression model for Gait Sequence Detection."""

    expected_model_name = "lr"


# Backward-compatible alias.
GsdLr = GsdLogisticRegression
