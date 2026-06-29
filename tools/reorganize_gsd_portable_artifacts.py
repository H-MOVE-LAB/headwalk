from pathlib import Path
import json
import shutil

import pandas as pd


MODEL_CLASSES = [
    "GsdSvm",
    "GsdRandomForest",
    "GsdKnn",
    "GsdLogisticRegression",
    "GsdGaussianNaiveBayes",
    "GsdRuleBased",
    "GsdCnn1D",
]


def to_jsonable(obj):
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    return obj


def parse_window_config(window_config: str, sampling_rate_hz: float = 100.0):
    duration_s = None
    overlap_fraction = None

    if "s_" in window_config:
        duration_s = float(window_config.split("s_")[0])

    if "50p_overlap" in window_config:
        overlap_fraction = 0.50

    window_samples = None
    step_samples = None

    if duration_s is not None:
        window_samples = int(round(duration_s * sampling_rate_hz))

    if window_samples is not None and overlap_fraction is not None:
        step_samples = int(round(window_samples * (1.0 - overlap_fraction)))

    return {
        "window_config": window_config,
        "sampling_rate_hz": sampling_rate_hz,
        "window_duration_s": duration_s,
        "window_samples": window_samples,
        "overlap_fraction": overlap_fraction,
        "step_samples": step_samples,
    }


def load_best_row(model_dir: Path):
    best_config_path = model_dir / "BEST_configuration.csv"

    if not best_config_path.exists():
        raise FileNotFoundError(f"Missing BEST_configuration.csv in {model_dir}")

    df = pd.read_csv(best_config_path)

    if df.empty:
        raise ValueError(f"Empty BEST_configuration.csv in {model_dir}")

    return df.iloc[0].to_dict()


def parse_json_field(value, default):
    if value is None:
        return default

    if pd.isna(value):
        return default

    try:
        return json.loads(value)
    except Exception:
        return default


def main():
    repo_root = Path.cwd()
    models_root = repo_root / "src" / "headwalk" / "gait_sequence_detection" / "models"
    portable_root = models_root / "portable_artifacts"

    if not portable_root.exists():
        raise FileNotFoundError(f"portable_artifacts not found: {portable_root}")

    summary_path = portable_root / "GSD_portable_artifacts_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    for class_name in MODEL_CLASSES:
        model_dir = portable_root / class_name

        if not model_dir.exists():
            print(f"[WARNING] Missing artifact folder: {model_dir}")
            continue

        print("\n============================================================")
        print(f"[INFO] Reorganizing {class_name}")
        print("============================================================")

        best_row = load_best_row(model_dir)

        window_config = str(best_row["window_config"])
        source_model_name = str(best_row["model"])
        n_features = int(best_row["n_features"])
        selected_features = parse_json_field(best_row.get("features"), default=[])
        best_params = parse_json_field(best_row.get("best_params"), default={})

        preprocessing = parse_window_config(window_config)
        preprocessing.update({
            "input_signal_type": (
                "raw_accelerometer_windows"
                if class_name == "GsdCnn1D"
                else "handcrafted_feature_table"
            ),
            "expected_raw_columns": [
                "acc_VT",
                "acc_ML",
                "acc_AP",
                "gyr_VT",
                "gyr_ML",
                "gyr_AP",
            ],
            "label_map": {
                "0": "static",
                "1": "walking",
            },
            "notes": (
                "Classical sklearn models expect the selected handcrafted features "
                "listed in config.json. The CNN expects raw accelerometer windows."
            ),
        })

        if class_name == "GsdCnn1D":
            old_model = model_dir / "GsdCnn1D.keras"
            new_model = model_dir / "model.keras"

            if old_model.exists():
                shutil.copy2(old_model, new_model)

            # Also keep Paolo-style best-model files in models root.
            for suffix in [".keras", ".json", ".weights.h5"]:
                src = model_dir / f"GsdCnn1D{suffix}"
                dst = models_root / f"GsdCnn1D{suffix}"
                if src.exists():
                    shutil.copy2(src, dst)

            cnn_meta_path = model_dir / "GsdCnn1D_preprocessing_metadata.joblib"

            if cnn_meta_path.exists():
                try:
                    import joblib
                    cnn_meta = joblib.load(cnn_meta_path)
                    preprocessing["cnn_preprocessing_metadata"] = to_jsonable(cnn_meta)
                except Exception as exc:
                    preprocessing["cnn_preprocessing_metadata_error"] = str(exc)

            # Remove non-portable internal duplicates from portable folder.
            for old_name in [
                "GsdCnn1D.keras",
                "GsdCnn1D.json",
                "GsdCnn1D.weights.h5",
                "GsdCnn1D_preprocessing_metadata.joblib",
            ]:
                old_path = model_dir / old_name
                if old_path.exists():
                    old_path.unlink()

            model_filename = "model.keras"
            artifact_format = "keras"

        else:
            old_model = model_dir / f"{class_name}.joblib"
            new_model = model_dir / "model.joblib"

            if old_model.exists():
                shutil.copy2(old_model, new_model)
                old_model.unlink()

            model_filename = "model.joblib"
            artifact_format = "joblib"

        config = {
            "class_name": class_name,
            "source_model_name": source_model_name,
            "model_filename": model_filename,
            "artifact_format": artifact_format,
            "window_config": window_config,
            "n_features": n_features,
            "selected_features": selected_features,
            "best_params": best_params,
        }

        metadata = {
            "class_name": class_name,
            "task": "gait_sequence_detection",
            "training_dataset": "WearGaitPD",
            "selection_rule": "Best internal-validation balanced accuracy across window configurations",
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "accuracy": float(best_row["accuracy"]),
            "recall_static": float(best_row["recall_static"]),
            "recall_walking": float(best_row["recall_walking"]),
            "label_map": {
                "0": "static",
                "1": "walking",
            },
            "model_version": "v2",
        }

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        with open(model_dir / "preprocessing.json", "w", encoding="utf-8") as f:
            json.dump(preprocessing, f, indent=4)

        # Remove wrapper-development CSVs from portable artifact folders.
        for csv_name in ["BEST_configuration.csv", "selected_features.csv"]:
            csv_path = model_dir / csv_name
            if csv_path.exists():
                csv_path.unlink()

        print(f"[OK] {class_name} reorganized.")

    print("\n[SUCCESS] Portable artifacts reorganized in Paolo-style layout.")


if __name__ == "__main__":
    main()
