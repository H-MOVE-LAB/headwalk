"""
GsdKnn algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdKnn(GsdSklearnModel):
    """Reusable KNN model for Gait Sequence Detection."""

    expected_model_name = "knn"
