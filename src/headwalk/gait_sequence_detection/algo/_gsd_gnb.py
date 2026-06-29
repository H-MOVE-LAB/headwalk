"""
GsdGaussianNaiveBayes algorithm wrapper.
"""

from __future__ import annotations

from ._gsd_sklearn import GsdSklearnModel


class GsdGaussianNaiveBayes(GsdSklearnModel):
    """Reusable Gaussian Naive Bayes model for Gait Sequence Detection."""

    expected_model_name = "gnb"


# Backward-compatible alias.
GsdGnb = GsdGaussianNaiveBayes
