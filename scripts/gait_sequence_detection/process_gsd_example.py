"""
Minimal example: load one portable GSD model and predict from a feature table.

This example is intentionally simple. It checks that the exported artifact can
be loaded and that the selected feature schema matches an existing CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from headwalk.gait_sequence_detection import GsdSvm


def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()

    for parent in [current] + list(current.parents):
        if (parent / "src").exists():
            return parent

    raise FileNotFoundError("Project root not found.")


def main() -> None:
    project_root = find_project_root(Path(__file__))

    artifact_dir = (
        project_root /
        "src" /
        "headwalk" /
        "gait_sequence_detection" /
        "models" /
        "portable_artifacts" /
        "svm_1s"
    )

    feature_csv = (
        project_root /
        "results" /
        "WearGaitPD_dataset" /
        "features" /
        "weargaitpd_features_1s_50p_overlap_clustered_internal_test.csv"
    )

    model = GsdSvm(artifact_dir=artifact_dir)
    feature_table = pd.read_csv(feature_csv)
    result = model.predict_from_feature_table(feature_table)

    output_csv = project_root / "results" / "gsd_svm_1s_example_predictions.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output_csv, index=False)

    print("Predictions saved to:")
    print(output_csv)


if __name__ == "__main__":
    main()
