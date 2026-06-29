"""
GsdSvm algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdSvm(GsdSklearnModel):
    """Reusable SVM model for Gait Sequence Detection."""

    expected_model_name = "svm"
