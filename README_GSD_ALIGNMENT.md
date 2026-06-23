# GSD Paolo-style alignment package

This folder contains the first alignment step for the Gait Sequence Detection pipeline.
It converts the current development workflow into a reusable structure analogous to Paolo's GED organization.

## What this package adds

```text
src/headwalk/gait_sequence_detection/
    __init__.py
    features.py
    algo/
        base.py
        _gsd_sklearn.py
        _gsd_svm.py
        _gsd_rf.py
        _gsd_knn.py
        _gsd_lr.py
        _gsd_gnb.py
    models/
        portable_artifacts/

tools/gait_sequence_detection/
    export_portable_gsd_artifacts.py

examples/gait_sequence_detection/
    03_load_and_evaluate_gsd_model.py

validation/gait_sequence_detection/
    load_portable_artifact_smoke_test.py
```

## Why this step is needed

The bottom-up wrapper is a development script. It trains and compares many candidate configurations.
Paolo-style code instead needs portable algorithm classes that load one selected trained model and apply it consistently.

The trained `.joblib` file alone is not enough. Each portable artifact also stores:

- `selected_features.json`: exact feature names and order expected by the model;
- `metadata.json`: model family, window length, validation metrics, label mapping, training notes;
- `preprocessing.json`: expected sampling frequency, channel order and upstream preprocessing assumptions;
- optional copied CSV/TXT files from the wrapper for traceability.

## How to run the export locally

From the repository root:

```bash
python tools/gait_sequence_detection/export_portable_gsd_artifacts.py --overwrite
```

The script expects this source folder to exist:

```text
results/WearGaitPD_dataset/features/bottomup_wrapper_feature_selection/
```

For each model/window pair, it reads:

```text
<window_config>/bottomup_wrapper_BEST_by_model.csv
<window_config>/<model>/<model>_topN_features.joblib
```

and creates:

```text
src/headwalk/gait_sequence_detection/models/portable_artifacts/<model>_<window>/
    model.joblib
    selected_features.json
    metadata.json
    preprocessing.json
```

## Important limitation

This package does not contain your trained models because the uploaded zip only contained source code.
The export must be run on the local repository where the `results/` folder and the `.joblib` files exist.

## Suggested immediate workflow

1. Copy these folders into Paolo's cloned repository.
2. Run the artifact export script.
3. Check `portable_artifacts_export_summary.csv`.
4. Run the smoke test.
5. Only after this step, build the external validation script that applies these artifacts to INDIVI, TOWalk, MOVEWISE and Tobii.
