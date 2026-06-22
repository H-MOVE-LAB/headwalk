"""
This module provides functions to compute quantitative gait metrics.

Its goal is to transform method outputs into interpretable gait-related
parameters that can be used for evaluation and comparison.

Typical metrics include:
- step and stride duration
- temporal gait parameters
- walking-bout duration
- event-based performance measures
- method comparison metrics

These functions are used after the core processing blocks to summarize performance.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_elements(y_true, y_pred):
    """
    Compute the confusion matrix elements for binary classification.

    Returns the following integer numbers:
    tn (true negative: 0 predicted as 0),
    fp (false positive: 0 predicted as 1),
    fn (false negative: 1 predicted as 0),
    tp (true positive: 1 predicted as 1).
    """
    # .ravel() serves the purpose of transforming a 2-by-2 matrix into a vector.
    # 'labels': List of labels to index the matrix. This may be used to reorder
    # or select a subset of labels. If None is given, those that appear at least
    # once in y_true or y_pred are used in sorted order.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    # Convert gold standard and predictions into arrays.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute accuracy as mean value of gold standard
    # and prediction correspondences; in other words, mean computes
    # the percentage of correct predictions (y_true == y_pred outputs an
    # array made of True and False logic labels).
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    """
    Compute precision for binary classification.
    """
    tn, fp, fn, tp = compute_confusion_elements(y_true, y_pred)
    denom = tp + fp
    # Precision defined as tp/(tp+fp) if denominator is greater than 0;
    # otherwise, precision manually set to 0.
    return tp / denom if denom > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Compute recall (sensitivity) for binary classification.
    """
    tn, fp, fn, tp = compute_confusion_elements(y_true, y_pred)
    denom = tp + fn
    # Recall defined as tp/(tp+fn) if denominator is greater than 0;
    # otherwise, recall manually set to 0.
    return tp / denom if denom > 0 else 0.0


def specificity_score(y_true, y_pred):
    """
    Compute specificity for binary classification.
    """
    tn, fp, fn, tp = compute_confusion_elements(y_true, y_pred)
    denom = tn + fp
    # Recall defined as tn/(tn+fp) if denominator is greater than 0;
    # otherwise, recall manually set to 0.
    return tn / denom if denom > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    Compute F1-score for binary classification.
    """
    # Compute precision score from y_true and y_pred.
    precision = precision_score(y_true, y_pred)
    # Compute recall score from y_true and y_pred.
    recall = recall_score(y_true, y_pred)

    denom = precision + recall

    # F1-score defined as 2*Precision*Recall/(Precision+Recall)
    # if denominator is greater than 0;
    # otherwise, F1-score manually set to 0.
    return 2 * precision * recall / denom if denom > 0 else 0.0


def classification_report_dict(y_true, y_pred):
    """
    Return a dictionary with the main binary classification metrics.
    """
    tn, fp, fn, tp = compute_confusion_elements(y_true, y_pred)

    # Extract the main binary classification metrics and
    # put them in a metric-specific dictionary.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def mean_absolute_error(y_true, y_pred):
    """
    Compute the mean absolute error (MAE).
    """
    # Convert y_true and y_pred into arrays.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Mean of absolute value computed on the difference between
    # y_true and y_pred.
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """
    Compute the root mean squared error (RMSE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # mean() computes the percentage of the squared difference between
    # correct labels and predictions. Then, square root applied to this value.
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_error(y_true, y_pred):
    """
    Compute the signed mean error (bias).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_pred - y_true)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    Parameters
    y_true : array-like
        Reference values
    y_pred : array-like
        Predicted values

    Returns
    Mean absolute percentage error (mape) in %, as float number.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Avoid division by zero. y_true != 0 outputs a logic mask whose
    # elements are set to True whether the corresponding y_true element
    # is different from 0 and set to False if it is not.
    non_zero_mask = y_true != 0

    # np.any(...) returns True whether non_zero_mask contains
    # at least one element True (different from 0).
    # If not ... returns mape = 0 if non_zero_mask contains all False
    # (all values in y_true are 0).
    if not np.any(non_zero_mask):
        return 0.0

    # When y_true contains non-zero elements, errors is computed as
    # (y_true-y_pred)/y_true.
    errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])

    # Compute mean of errors, then multiplied by 100.
    return np.mean(errors) * 100
