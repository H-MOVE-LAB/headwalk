"""
GsdRuleBased algorithm wrapper.

The saved rule-based artifact is a joblib model produced during development.
This wrapper exposes it with the same interface as the other GSD algorithms.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer

from ._gsd_sklearn import GsdSklearnModel


class RuleBasedGSDClassifier(BaseEstimator, ClassifierMixin):
    """
    Compatibility class for loading the saved rule-based joblib artifact.
    """

    def __init__(
        self,
        threshold_quantile=0.50,
        score_direction="walking_higher",
    ):
        self.threshold_quantile = threshold_quantile
        self.score_direction = score_direction

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_array = X.to_numpy(dtype=float)
        else:
            X_array = np.asarray(X, dtype=float)
            self.feature_names_in_ = [
                f"feature_{i}" for i in range(X_array.shape[1])
            ]

        self.imputer_ = SimpleImputer(strategy="median")
        X_imputed = self.imputer_.fit_transform(X_array)

        self.center_ = np.nanmedian(X_imputed, axis=0)
        q25 = np.nanpercentile(X_imputed, 25, axis=0)
        q75 = np.nanpercentile(X_imputed, 75, axis=0)
        iqr = q75 - q25
        self.scale_ = np.where(np.abs(iqr) < 1e-12, 1.0, iqr)

        train_scores = self.decision_function(X)
        self.threshold_ = float(
            np.quantile(train_scores, self.threshold_quantile)
        )

        self.classes_ = np.array([0, 1], dtype=int)

        return self

    def decision_function(self, X):
        if isinstance(X, pd.DataFrame):
            X_array = X.to_numpy(dtype=float)
        else:
            X_array = np.asarray(X, dtype=float)

        X_imputed = self.imputer_.transform(X_array)
        X_scaled = (X_imputed - self.center_) / self.scale_

        return np.nanmean(X_scaled, axis=1)

    def predict(self, X):
        scores = self.decision_function(X)

        if self.score_direction == "walking_higher":
            return (scores >= self.threshold_).astype(int)

        return (scores <= self.threshold_).astype(int)


class GsdRuleBased(GsdSklearnModel):
    """Reusable rule-based model for Gait Sequence Detection."""

    expected_model_name = "rule_based"

    def __init__(self, artifact_dir=None):
        # If the joblib artifact was saved while the training script was run as
        # __main__, pickle needs to find RuleBasedGSDClassifier there.
        sys.modules["__main__"].RuleBasedGSDClassifier = RuleBasedGSDClassifier
        super().__init__(artifact_dir=artifact_dir)
