from pathlib import Path
import argparse
import json
import shutil

import pandas as pd


MODEL_MAP = {
    "svm": "GsdSvm",
    "rf": "GsdRandomForest",
    "knn": "GsdKnn",
    "lr": "GsdLogisticRegression",
    "gnb": "GsdGaussianNaiveBayes",
    "rule_based": "GsdRuleBased",
    "cnn1d": "GsdCnn1D",
}


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPIED] {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper-root", required=True)
    parser.add_argument("--headwalk-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wrapper_root = Path(args.wrapper_root).resolve()
    headwalk_root = Path(args.headwalk_root).resolve()

    if not wrapper_root.exists():
        raise FileNotFoundError(f"Wrapper root not found: {wrapper_root}")

    out_root = (
        headwalk_root
        / "src"
        / "headwalk"
        / "gait_sequence_detection"
        / "models"
        / "portable_artifacts"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    config_files = sorted(wrapper_root.glob("*/*/*_BEST_configuration.csv"))

    if len(config_files) == 0:
        raise FileNotFoundError(
            f"No *_BEST_configuration.csv files found under: {wrapper_root}"
        )

    rows = []

    for config_file in config_files:
        df = pd.read_csv(config_file)
        if df.empty:
            continue

        row = df.iloc[0].to_dict()
        row["config_path"] = str(config_file)
        rows.append(row)

    all_best_df = pd.DataFrame(rows)

    if "balanced_accuracy" not in all_best_df.columns:
        raise ValueError("Column 'balanced_accuracy' not found in BEST configuration files.")

    selected_rows = []

    for model_name, class_name in MODEL_MAP.items():
        model_df = all_best_df[all_best_df["model"] == model_name].copy()

        if model_df.empty:
            print(f"[WARNING] No BEST configuration found for model: {model_name}")
            continue

        best_idx = model_df["balanced_accuracy"].astype(float).idxmax()
        best_row = model_df.loc[best_idx].to_dict()
        selected_rows.append(best_row)

        window_config = best_row["window_config"]
        n_features = int(best_row["n_features"])

        src_model_dir = wrapper_root / window_config / model_name
        dst_model_dir = out_root / class_name

        if dst_model_dir.exists() and args.overwrite:
            shutil.rmtree(dst_model_dir)

        dst_model_dir.mkdir(parents=True, exist_ok=True)

        print("\n============================================================")
        print(f"[MODEL] {model_name} -> {class_name}")
        print(f"[WINDOW] {window_config}")
        print(f"[N FEATURES] {n_features}")
        print(f"[BALANCED ACCURACY] {best_row['balanced_accuracy']}")
        print("============================================================")

        if model_name == "cnn1d":
            stem = f"best_model_{n_features}"

            copy_file(src_model_dir / f"{stem}.keras", dst_model_dir / f"{class_name}.keras")
            copy_file(src_model_dir / f"{stem}.json", dst_model_dir / f"{class_name}.json")
            copy_file(src_model_dir / f"{stem}.weights.h5", dst_model_dir / f"{class_name}.weights.h5")
            copy_file(
                src_model_dir / f"{stem}_preprocessing_metadata.joblib",
                dst_model_dir / f"{class_name}_preprocessing_metadata.joblib",
            )

            artifact_format = "keras_json_weights_h5"

        else:
            src_model_file = src_model_dir / f"{model_name}_top{n_features}_features.joblib"
            copy_file(src_model_file, dst_model_dir / f"{class_name}.joblib")
            artifact_format = "joblib"

        copy_file(
            src_model_dir / f"{model_name}_BEST_configuration.csv",
            dst_model_dir / "BEST_configuration.csv",
        )

        copy_file(
            src_model_dir / f"{model_name}_BEST_selected_features.csv",
            dst_model_dir / "selected_features.csv",
        )

        metadata = {
            "class_name": class_name,
            "source_model_name": model_name,
            "window_config": window_config,
            "n_features": n_features,
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "accuracy": float(best_row["accuracy"]),
            "recall_static": float(best_row["recall_static"]),
            "recall_walking": float(best_row["recall_walking"]),
            "artifact_format": artifact_format,
            "label_map": {
                "0": "static",
                "1": "walking",
            },
            "training_dataset": "WearGaitPD",
            "selection_rule": "Best internal-validation balanced accuracy across window configurations",
            "source_wrapper_root": str(wrapper_root),
            "source_best_configuration": str(src_model_dir / f"{model_name}_BEST_configuration.csv"),
        }

        with open(dst_model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_root / "GSD_portable_artifacts_summary.csv", index=False)

    print("\n[SUCCESS] GSD portable artifacts exported to:")
    print(out_root)
    print("\n[SUCCESS] Summary saved to:")
    print(out_root / "GSD_portable_artifacts_summary.csv")


if __name__ == "__main__":
    main()
