"""
GsdGnb algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdGnb(GsdSklearnModel):
    """Reusable GNB model for Gait Sequence Detection."""

    expected_model_name = "gnb"
