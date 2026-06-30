# Gait Sequence Detection examples

This folder contains example scripts for running Gait Sequence Detection (GSD)
with the `headwalk.gait_sequence_detection` package.

Gait Sequence Detection is the binary task of detecting walking / locomotion
periods inside a single trial. The main standardized output is a gait-sequence
list, stored in each algorithm instance as:

    algorithm.gs_list_

The `gs_list_` table has four columns:

    start_s
    end_s
    start_samples
    end_samples

All times and sample indices are relative to the beginning of the input trial.

The public method exposed by the current GSD algorithms is:

    algorithm.detect(...)

After calling `detect(...)`, each algorithm instance stores:

    algorithm.window_detections_
    algorithm.gs_list_
    algorithm.result_

`window_detections_` contains one row per analysed window.
`gs_list_` contains the final walking / locomotion intervals obtained by
post-processing the window-level detections.

-------------------------------------------------------------------------------
Repository sections involved in GSD
-------------------------------------------------------------------------------

The current GSD implementation is distributed across these repository sections:

    src/headwalk/gait_sequence_detection/
    examples/gait_sequence_detection/
    example_data/gait_sequence_detection/
    tools/
    scripts/gait_sequence_detection/

Each section has a different role.

-------------------------------------------------------------------------------
1. Core package
-------------------------------------------------------------------------------

The core package is located in:

    src/headwalk/gait_sequence_detection/

The current structure is:

    src/headwalk/gait_sequence_detection/
        algo/
        preprocessing/
        models/
        features.py

This folder contains the reusable package code.

-------------------------------------------------------------------------------
1.1 Algorithm classes
-------------------------------------------------------------------------------

Algorithm classes are located in:

    src/headwalk/gait_sequence_detection/algo/

The current files are:

    base.py
    _gsd_sklearn.py
    _gsd_svm.py
    _gsd_rf.py
    _gsd_knn.py
    _gsd_lr.py
    _gsd_gnb.py
    _gsd_rule_based.py
    _gsd_cnn1d.py
    __init__.py

The main design choice is that all GSD algorithms expose the same public API:

    algorithm.detect(...)

The shared detection flow is:

    input trial or feature table
        -> window-level detection
        -> post-processing of walking windows
        -> standardized gs_list_

The common output attributes are:

    algorithm.window_detections_
    algorithm.gs_list_
    algorithm.result_

The base class handles the shared logic. Child classes specialize only the
algorithm-specific window detection step.

-------------------------------------------------------------------------------
base.py
-------------------------------------------------------------------------------

`base.py` defines the shared GSD algorithm interface.

Main class:

    BaseGsdAlgorithm

Main public method:

    detect(...)

The `detect(...)` method:

    1. receives a raw/preprocessed signal or a feature table;
    2. calls the child-specific `_detect_windows(...)` method;
    3. stores the window-level output in `window_detections_`;
    4. converts walking windows into gait-sequence intervals;
    5. stores the final intervals in `gs_list_`;
    6. stores a combined result object in `result_`;
    7. returns `self`.

The final `gs_list_` format is always:

    start_s
    end_s
    start_samples
    end_samples

This keeps the external interface consistent across all GSD algorithms.

-------------------------------------------------------------------------------
_gsd_sklearn.py
-------------------------------------------------------------------------------

`_gsd_sklearn.py` defines the shared implementation for classical
sklearn-compatible GSD models.

These models can be used in two ways:

    1. from a preprocessed IMU signal;
    2. from an already computed feature table.

When a signal is provided, the class:

    1. builds sliding windows;
    2. computes handcrafted window features;
    3. selects the exact features required by the stored model;
    4. runs the sklearn estimator;
    5. returns window-level GSD detections.

When a feature table is provided, the class directly selects the required
features and runs the estimator.

The following model wrappers inherit from this class:

    GsdSvm
    GsdRandomForest
    GsdKnn
    GsdLogisticRegression
    GsdGaussianNaiveBayes

-------------------------------------------------------------------------------
_gsd_rule_based.py
-------------------------------------------------------------------------------

`_gsd_rule_based.py` exposes a rule-based GSD model through the same `detect(...)`
interface used by the other algorithms.

The wrapper also includes a compatibility class used to load the stored rule-
based joblib artifact.

The public usage remains:

    algorithm = GsdRuleBased()
    algorithm.detect(...)

-------------------------------------------------------------------------------
_gsd_cnn1d.py
-------------------------------------------------------------------------------

`_gsd_cnn1d.py` defines the CNN-based GSD algorithm.

Unlike the sklearn-compatible models, the CNN does not use handcrafted feature
tables. It uses preprocessed accelerometer windows through a dedicated pathway.

The external interface remains identical:

    algorithm = GsdCnn1D()
    algorithm.detect(...)

This makes the usage consistent even though the internal model family is
different.

-------------------------------------------------------------------------------
2. Preprocessing
-------------------------------------------------------------------------------

Preprocessing utilities are located in:

    src/headwalk/gait_sequence_detection/preprocessing/

The current files are:

    gsd_imu_preprocessing.py
    _quality.py
    __init__.py

-------------------------------------------------------------------------------
gsd_imu_preprocessing.py
-------------------------------------------------------------------------------

This module contains the shared head-IMU preprocessing currently used by the GSD
examples.

The intended preprocessing chain is:

    raw accelerometer / gyroscope
        -> optional fixed axis transformation
        -> optional accelerometer unit handling
        -> NaN-aware gravity alignment
        -> IMU quality mask from original missing-data regions
        -> numerical gap filling for filtering only
        -> continuous pitch-roll correction
        -> gravity removal
        -> zero-phase low-pass filtering
        -> standardized six-channel IMU signal

The standardized signal convention used by the current GSD models is:

    acc_vt
    acc_ml
    acc_ap
    gyr_vt
    gyr_ml
    gyr_ap

Important distinction:

    Numerical gap filling is used to make filtering and orientation correction
    possible on a complete signal. It does not automatically make every
    interpolated sample scientifically reliable.

Sample reliability is controlled separately by quality masks when labelled
constructed datasets are used for training, validation or benchmarking.

-------------------------------------------------------------------------------
_quality.py
-------------------------------------------------------------------------------

`_quality.py` contains sample-wise and window-wise quality-mask utilities.

It separates two concepts:

    IMU quality:
        reliability based on original IMU missing-data regions.

    Walkway / instrumented-mat quality:
        optional reliability information available only for datasets or tasks
        that contain meaningful foot-contact / instrumented-walkway data.

The final quality mask is:

    final_quality = imu_quality

for datasets or tasks without instrumented walkway quality, and:

    final_quality = imu_quality AND walkway_quality

for datasets or tasks where walkway quality is meaningful and explicitly
enabled.

For labelled dataset construction or validation, a window should be accepted
only if:

    all samples in the window have final_quality == True

and:

    no sample in the window has the none / invalid label

For raw unlabelled inference, labels are not available. In that case, only the
quality-mask condition can be applied by the caller.

Quality masks are not required for the basic raw single-trial inference examples.

-------------------------------------------------------------------------------
3. Feature extraction
-------------------------------------------------------------------------------

Handcrafted feature extraction is implemented in:

    src/headwalk/gait_sequence_detection/features.py

This module computes the window-level features used by the classical
sklearn-compatible models.

Feature names must remain consistent with the names stored in each portable
artifact configuration, because the saved models expect a specific input schema.

-------------------------------------------------------------------------------
4. Model artifacts
-------------------------------------------------------------------------------

GSD model artifacts are stored in:

    src/headwalk/gait_sequence_detection/models/

The portable artifact layout is:

    src/headwalk/gait_sequence_detection/models/portable_artifacts/
        GsdSvm/
        GsdRandomForest/
        GsdKnn/
        GsdLogisticRegression/
        GsdGaussianNaiveBayes/
        GsdRuleBased/
        GsdCnn1D/

Each portable artifact folder contains:

    config.json
    metadata.json
    preprocessing.json
    model.joblib or model.keras

The JSON files describe:

    model identity
    window configuration
    selected features or raw input requirements
    expected sampling frequency
    label map
    training / validation metadata

The model file contains the trained estimator.

For the CNN model, additional Keras files may also be present in the models
folder for compatibility with the existing model-loading style.

-------------------------------------------------------------------------------
5. Development tools
-------------------------------------------------------------------------------

The `tools/` folder contains development utilities used to export and reorganize
trained GSD artifacts:

    tools/export_gsd_best_artifacts.py
    tools/reorganize_gsd_portable_artifacts.py

These tools are not required for normal single-trial inference.

They are intended for development workflows in which trained models are selected,
exported or reorganized into the portable artifact layout.

-------------------------------------------------------------------------------
6. Scripts
-------------------------------------------------------------------------------

The folder:

    scripts/gait_sequence_detection/

contains package-level development scripts.

These scripts are separate from the public examples in:

    examples/gait_sequence_detection/

The public examples should be preferred when demonstrating how to run GSD on a
single trial.

-------------------------------------------------------------------------------
7. Example data
-------------------------------------------------------------------------------

Compact example trials are stored in:

    example_data/gait_sequence_detection/

The current files are:

    weargaitpd_pd_freewalk_nls192.csv
    weargaitpd_control_freewalk_whc021.csv

These files contain only the columns required by the current examples:

    Time
    GeneralEvent
    Forehead_Acc_X
    Forehead_Acc_Y
    Forehead_Acc_Z
    Forehead_Gyr_X
    Forehead_Gyr_Y
    Forehead_Gyr_Z

`GeneralEvent` is used only when the optional visual reference is requested.
The GSD algorithms themselves do not require labels.

-------------------------------------------------------------------------------
8. Example scripts
-------------------------------------------------------------------------------

The current GSD examples are:

    01_apply_gsd_single_trial.py
    04_load_and_evaluate_single_trial.py

-------------------------------------------------------------------------------
01_apply_gsd_single_trial.py
-------------------------------------------------------------------------------

This is the minimal single-trial inference example.

It shows how to:

    1. load a raw single-trial CSV;
    2. apply the current head-IMU preprocessing;
    3. run one or more GSD algorithms through `detect(...)`;
    4. print the resulting `gs_list_`;
    5. optionally save `window_detections_` and `gs_list_` tables.

This script does not build a reference and does not generate a plot.

It is intended to demonstrate the raw inference workflow:

    raw trial
        -> preprocessing
        -> algorithm.detect(...)
        -> algorithm.gs_list_

By default, it uses:

    example_data/gait_sequence_detection/weargaitpd_pd_freewalk_nls192.csv

A different trial can be passed with:

    --trial-csv path/to/trial.csv

-------------------------------------------------------------------------------
04_load_and_evaluate_single_trial.py
-------------------------------------------------------------------------------

This is the visual evaluation example.

It shows how to:

    1. load one or two raw trials;
    2. apply the current head-IMU preprocessing;
    3. run all available GSD algorithms;
    4. optionally build a binary reference from trial labels;
    5. plot IMU signals and detected gait sequences;
    6. save `window_detections_`, `gs_list_` and optional reference tables.

The visual output contains stacked rows:

    preprocessed IMU signals
    optional REF row
    one row per GSD algorithm

Detected gait sequences are shown as semi-transparent shaded bands.

The reference row is optional. The algorithms do not require reference labels.

-------------------------------------------------------------------------------
9. Run the minimal apply example
-------------------------------------------------------------------------------

From the repository root:

    PYTHONPATH=src python examples/gait_sequence_detection/01_apply_gsd_single_trial.py \
      --sampling-rate-hz 100 \
      --algorithm svm \
      --save-outputs \
      --output-dir results/gait_sequence_detection/apply_single_trial

To run all available algorithms:

    PYTHONPATH=src python examples/gait_sequence_detection/01_apply_gsd_single_trial.py \
      --sampling-rate-hz 100 \
      --algorithm all \
      --save-outputs \
      --output-dir results/gait_sequence_detection/apply_single_trial_all

-------------------------------------------------------------------------------
10. Run the visual evaluation example
-------------------------------------------------------------------------------

From the repository root:

    PYTHONPATH=src python examples/gait_sequence_detection/04_load_and_evaluate_single_trial.py \
      --sampling-rate-hz 100 \
      --start-s 0 \
      --duration-s 60 \
      --max-bands 10 \
      --show-reference \
      --output-dir results/gait_sequence_detection/example_data_check

By default, the script uses both compact WearGaitPD FreeWalk example trials:

    weargaitpd_pd_freewalk_nls192.csv
    weargaitpd_control_freewalk_whc021.csv

-------------------------------------------------------------------------------
11. Optional binary reference
-------------------------------------------------------------------------------

The visual evaluation script can display an optional reference row with:

    --show-reference

For binary GSD, the default walking-like labels are:

    walk
    walking
    stair
    stairs
    ascend
    ascending
    descend
    descending
    upstairs
    downstairs
    turn
    turning

These labels are treated as walking-like because GSD is a binary locomotion
detection task.

Labels such as:

    standing
    sitting
    chair
    opendoor

are not included in the default walking-like set.

If a trial does not contain the reference label column, the visual example can
still be run without `--show-reference`.

-------------------------------------------------------------------------------
12. Raw inference versus labelled validation
-------------------------------------------------------------------------------

The GSD algorithms are intended to work on raw, not necessarily labelled,
single-trial data.

Labels and reference events are not required for inference. They are only needed
for validation, benchmarking or visual comparison.

Core raw inference workflow:

    raw trial
        -> preprocessing
        -> algorithm.detect(...)
        -> algorithm.gs_list_

Constructed datasets may contain additional fields such as:

    labels
    activity annotations
    quality masks
    reference annotations

These fields are useful for training and validation, but they are not required
by the raw inference interface.

-------------------------------------------------------------------------------
13. Current scope and future extensions
-------------------------------------------------------------------------------

The current example scripts are intentionally focused on compact WearGaitPD
FreeWalk CSV files because those files are included in `example_data`.

The core algorithm interface is more general than the example data.

Future dataset-specific adapters can standardize other raw datasets into the
same six-channel input schema:

    acc_vt
    acc_ml
    acc_ap
    gyr_vt
    gyr_ml
    gyr_ap

Once a trial has this standardized schema, the same `detect(...)` interface can
be used across algorithms.

The current quality-mask utilities already support the distinction between:

    IMU-only quality

and:

    IMU quality combined with optional walkway / instrumented-mat quality

but this is mainly relevant for labelled training, validation and benchmarking
workflows. Basic raw inference does not require labels.

-------------------------------------------------------------------------------
14. Optional quality-aware raw inference
-------------------------------------------------------------------------------

The minimal apply script can optionally remove low-quality windows after model
inference:

    --exclude-low-quality-windows

This option uses the sample-wise `imu_quality` mask produced by preprocessing.

The logic is:

    1. preprocessing is applied to the full trial;
    2. the selected GSD algorithm runs on the preprocessed trial;
    3. each detected window is checked against `imu_quality`;
    4. windows touching at least one sample with `imu_quality == False` are removed;
    5. `gs_list_` is rebuilt from the remaining high-quality windows.

This option does not require labels and does not use a reference system.

It is useful for raw inference when the original IMU signal contains long
missing-data regions. Short gaps may be filled numerically during preprocessing,
but long original dropouts remain marked as low quality by the quality mask.

Example:

    PYTHONPATH=src python examples/gait_sequence_detection/01_apply_gsd_single_trial.py \
      --sampling-rate-hz 100 \
      --algorithm svm \
      --exclude-low-quality-windows \
      --save-outputs \
      --output-dir results/gait_sequence_detection/apply_single_trial_quality_aware

This is different from labelled dataset construction or validation.

For labelled datasets, a stricter rule can be applied:

    keep a window only if all samples have final_quality == True
    and no sample has the none / invalid label

For raw inference, labels are usually unavailable, so only signal-quality based
filtering can be applied.
