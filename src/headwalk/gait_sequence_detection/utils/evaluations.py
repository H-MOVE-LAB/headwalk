"""
Evaluation utilities for gait activity classification.

This module computes performance metrics at different aggregation levels:
- overall test-set level
- per-dataset level
- per-subject level
- per-subject and per-dataset level

Main metrics:
- window-level accuracy
- window-level balanced accuracy
- per-class recall

The functions are dataset-independent and can be reused for WearGaitPD,
INDIVI, TOWalk, MoveWise, or future external test sets.
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


DEFAULT_LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
}

def infer_label_name_map(y_true, y_pred):
    """
    Infer the label-name map from the labels actually present in y_true and y_pred.

    This prevents binary GSD evaluation from being polluted by old HAR labels
    such as ascending and descending.
    """

    present_labels = sorted(
        set(np.asarray(y_true).astype(int)) |
        set(np.asarray(y_pred).astype(int))
    )

    base_map = {
        0: "static",
        1: "walking",
        2: "ascending",
        3: "descending",
    }

    return {
        label: base_map.get(label, f"class_{label}")
        for label in present_labels
    }

def build_results_dataframe(df_test, y_true, y_pred, label_name_map):
    """
    Build a result dataframe by adding true and predicted labels
    to the original test dataframe.

    Keeping the original metadata is important because metrics must be
    aggregated by dataset, subject, task, and other experimental factors.
    """

    results_df = df_test.copy()

    results_df["y_true"] = np.asarray(y_true).astype(int)
    results_df["y_pred"] = np.asarray(y_pred).astype(int)

    results_df["y_true_name"] = results_df["y_true"].map(label_name_map)
    results_df["y_pred_name"] = results_df["y_pred"].map(label_name_map)

    return results_df


def compute_metric_row(results_df, label_name_map, group_name="overall", group_value="overall"):
    """
    Compute one row of metrics for a given subset of the results dataframe.

    The function returns:
    - accuracy
    - balanced accuracy
    - per-class recall
    - number of windows
    """

    y_true = results_df["y_true"].astype(int)
    y_pred = results_df["y_pred"].astype(int)

    labels = sorted(label_name_map.keys())

    row = {
        "group_name": group_name,
        "group_value": group_value,
        "n_windows": len(results_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": np.mean(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average=None,
                zero_division=0
            )
        ),
    }

    labels = sorted(label_name_map.keys())

    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    for label_id, recall_value in zip(labels, recalls):
        label_name = label_name_map[label_id]
        row[f"recall_{label_name}"] = recall_value

    return row


def compute_overall_metrics(results_df, label_name_map):
    """
    Compute global metrics on the full test set.
    """

    row = compute_metric_row(
        results_df=results_df,
        label_name_map=label_name_map,
        group_name="overall",
        group_value="overall"
    )

    return pd.DataFrame([row])


def compute_grouped_metrics(results_df, group_cols, label_name_map):
    """
    Compute metrics after grouping the test results by one or more columns.

    Examples:
    - group_cols=["dataset"]
    - group_cols=["subject_id"]
    - group_cols=["dataset", "subject_id"]
    """

    rows = []

    grouped = results_df.groupby(group_cols)

    for group_values, group_df in grouped:

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_value_str = " | ".join([
            f"{col}={val}"
            for col, val in zip(group_cols, group_values)
        ])

        row = compute_metric_row(
            results_df=group_df,
            label_name_map=label_name_map,
            group_name="+".join(group_cols),
            group_value=group_value_str
        )

        for col, val in zip(group_cols, group_values):
            row[col] = val

        rows.append(row)

    return pd.DataFrame(rows)


def save_confusion_matrix(results_df, output_path, label_name_map):
    """
    Save the confusion matrix as a CSV file.
    """

    labels = sorted(label_name_map.keys())
    label_names = [label_name_map[label] for label in labels]

    cm = confusion_matrix(
        results_df["y_true"],
        results_df["y_pred"],
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names]
    )

    cm_df.to_csv(output_path)

    return cm_df


def save_classification_report(results_df, output_path, label_name_map):
    """
    Save the sklearn classification report as a text file.
    """

    labels = sorted(label_name_map.keys())
    target_names = [label_name_map[label] for label in labels]

    report = classification_report(
        results_df["y_true"],
        results_df["y_pred"],
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    with open(output_path, "w") as f:
        f.write(report)

    return report


def generate_basic_insights(overall_df, per_dataset_df, per_subject_df):
    """
    Generate simple automatic textual insights from the metric tables.

    These insights are meant as a first diagnostic summary, not as final
    scientific interpretation.
    """

    insights = []

    overall = overall_df.iloc[0]

    insights.append(
        f"Overall balanced accuracy: {overall['balanced_accuracy']:.3f}"
    )

    insights.append(
        f"Overall accuracy: {overall['accuracy']:.3f}"
    )

    recall_cols = [
        col for col in overall_df.columns
        if col.startswith("recall_")
    ]

    for col in recall_cols:
        insights.append(
            f"Overall {col}: {overall[col]:.3f}"
        )

    if len(per_dataset_df) > 0:
        best_dataset = per_dataset_df.sort_values(
            "balanced_accuracy",
            ascending=False
        ).iloc[0]

        worst_dataset = per_dataset_df.sort_values(
            "balanced_accuracy",
            ascending=True
        ).iloc[0]

        insights.append(
            f"Best dataset-level balanced accuracy: "
            f"{best_dataset['group_value']} = {best_dataset['balanced_accuracy']:.3f}"
        )

        insights.append(
            f"Worst dataset-level balanced accuracy: "
            f"{worst_dataset['group_value']} = {worst_dataset['balanced_accuracy']:.3f}"
        )

    if len(per_subject_df) > 0:
        low_subjects = per_subject_df[
            per_subject_df["balanced_accuracy"] < 0.50
        ]

        insights.append(
            f"Subjects with balanced accuracy below 0.50: {len(low_subjects)}"
        )

    return insights


def run_full_evaluation(df_test, y_true, y_pred, output_dir, model_name):
    """
    Run the complete evaluation workflow and save all outputs.

    Saved outputs:
    - results dataframe with predictions
    - overall metrics
    - per-dataset metrics
    - per-subject metrics
    - per-dataset/per-subject metrics
    - confusion matrix
    - classification report
    - basic textual insights
    """

    os.makedirs(output_dir, exist_ok=True)
    label_name_map = infer_label_name_map(y_true, y_pred)

    results_df = build_results_dataframe(
        df_test=df_test,
        y_true=y_true,
        y_pred=y_pred,
        label_name_map=label_name_map
    )

    overall_df = compute_overall_metrics(
        results_df=results_df,
        label_name_map=label_name_map
    )

    per_dataset_df = compute_grouped_metrics(
        results_df=results_df,
        group_cols=["dataset"],
        label_name_map=label_name_map
    )

    per_subject_df = compute_grouped_metrics(
        results_df=results_df,
        group_cols=["subject_id"],
        label_name_map=label_name_map
    )

    per_dataset_subject_df = compute_grouped_metrics(
        results_df=results_df,
        group_cols=["dataset", "subject_id"],
        label_name_map=label_name_map
    )

    save_confusion_matrix(
        results_df=results_df,
        output_path=os.path.join(output_dir, f"{model_name}_confusion_matrix.csv"),
        label_name_map=label_name_map
    )

    report = save_classification_report(
        results_df=results_df,
        output_path=os.path.join(output_dir, f"{model_name}_classification_report.txt"),
        label_name_map=label_name_map
    )

    insights = generate_basic_insights(
        overall_df=overall_df,
        per_dataset_df=per_dataset_df,
        per_subject_df=per_subject_df
    )

    insights_path = os.path.join(output_dir, f"{model_name}_insights.txt")

    with open(insights_path, "w") as f:
        for insight in insights:
            f.write(insight + "\n")

    print(f"\n===== {model_name} EVALUATION SUMMARY =====")
    print(overall_df.to_string(index=False))

    print("\nClassification report:")
    print(report)

    print("\nBasic insights:")
    for insight in insights:
        print("-", insight)

    return {
        "results": results_df,
        "overall": overall_df,
        "per_dataset": per_dataset_df,
        "per_subject": per_subject_df,
        "per_dataset_subject": per_dataset_subject_df,
        "insights": insights,
    }