"""
GsdLr algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdLr(GsdSklearnModel):
    """Reusable LR model for Gait Sequence Detection."""

    expected_model_name = "lr"
