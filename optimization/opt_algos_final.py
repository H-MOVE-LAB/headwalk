import os
import glob
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

from src.headwalk.gait_events_detection import GedJiang
# Import preprocessing utilities from your existing file
from src.headwalk.utils import (
    load_head_data,
    rotate_to_gravity,
    compute_quality_mask,
)

# Import algorithms
from src.headwalk.gait_events_detection import (
    GedJarchi,
    GedSeifer,
    GedTransformer
)

# =============================================================================
# 1. CONFIGURATION & SEARCH SPACE
# =============================================================================

algorithms_config = {
    # "Tomc": {
    #     "class": IcdTomc,
    #     "params": {
    #         # AFO gain (Dynamics)
    #         "k_phi": {"type": "float", "low": 1.0, "high": 5.5, "step": 0.5},
    #         "k0": {"type": "float", "low": 0.1, "high": 0.31, "step": 0.01},
    #         # Structural parameters
    #         "omega0": {"type": "float", "low": 0.5, "high": 1.6, "step": 0.1},
    #     }
    # },
    # "Fang": {
    #     "class": IcdFang,
    #     "params": {
    #         "lowpass_hz": {"type": "float", "low": 2.6, "high": 4.0, "step": 0.2},
    #         "wavelet_type": {"type": "categorical", "choices": ["gaus1", "gaus2"]}
    #     }
    # },
    # "Diao": {
    #     "class": IcdDiaoComplete,
    #     "params": {
    #         "cutoff_hz": {"type": "float", "low": 2.0, "high": 6.2, "step": 0.2},
    #     }
    # },
    # "Fawden": {
    #     "class": IcdFawden,
    #     "params": {
    #         "sharpening_weight": {"type": "float", "low": 0.4, "high": 0.81, "step": 0.01},
    #     }
    # },
    "Jiang": {
        "class": GedJiang,
        "params": {
            "fc_hz": {"type": "float", "low": 2.6, "high": 6.1, "step": 0.2},
            "c_constant": {"type": "float", "low": 0, "high": 2.2, "step": 0.2},
        }
    },
    "Seifer": {
        "class": GedSeifer,
        "params": {
            "cutoff_hz": {"type": "float", "low": 2.0, "high": 6.1, "step": 0.2},
        }
    },
    "Transformer": {
        "class": GedTransformer,
        "params": {
            "offset": {"type": "float", "low": 0, "high": 0.051, "step": 0.001},
        }
    },
    "Jarchi": {
        "class": GedJarchi,
        "params": {
            "radius_s": {"type": "float", "low": 0.05, "high": 0.21, "step": 0.01},
        }
    },
}

# Paths
BASE_OUTPUT_PATH = "C:/algo_optimization/"
DATA_ROOT_CONTROL = r"X:\Visiting_Staff\2025 - Paolo Tasca\Code\weargaitpd_processing\data\Version 1\CONTROL PARTICIPANTS\CSV files"
DATA_ROOT_PD = r"X:\Visiting_Staff\2025 - Paolo Tasca\Code\weargaitpd_processing\data\Version 1\PD PARTICIPANTS\CSV files"

# Optuna Database Configuration
DB_PATH = os.path.join(BASE_OUTPUT_PATH, "weargait_optimization.db")
STORAGE_NAME = f"sqlite:///{DB_PATH}"

# Global Parameters
SAMPLING_RATE = 100.0
TOLERANCE_SEC = 0.25  # Matching tolerance
WINDOW_TOL_SEC = 0.26  # Boundary tolerance for Ground Truth inclusion
TRIALS_PER_STUDY = 20

os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)


# =============================================================================
# 2. DATA PRE-LOADING CACHE
# =============================================================================

def preload_data_to_memory(file_list: list[str]) -> list[dict]:
    """
    Loads, rotates to gravity, and computes quality masks for all files once.
    This prevents redundant I/O and deterministic calculations during Optuna trials.
    """
    print(f"Starting data pre-loading and preprocessing for {len(file_list)} files...")
    cached_data = []

    for fpath in tqdm(file_list):
        fname = os.path.basename(fpath)

        # Apply exclusion criteria
        if "balance" in fpath or "tandem" in fpath or "hc143" in fname:
            # print(f"  -> Skipped (Exclusion Criteria): {fname}")
            continue

        # Extract metadata
        subject_id = fname.split('_')[0]
        cohort = "CONTROL" if "CONTROL" in fpath else "PD"
        test_type = fname.split('_')[1] if len(fname.split('_')) > 1 else "unknown"

        # Load raw data
        X, Y, L_cont, R_cont = load_head_data(fpath)
        if X is None:
            continue

        # Check mask early to avoid unnecessary processing
        quality_mask = compute_quality_mask(X, L_cont, R_cont)
        if not np.any(quality_mask):
            continue

        # Rotate to gravity and combine into DataFrame
        acc_aligned, gyr_aligned = rotate_to_gravity(X[:, :3], X[:, 3:])
        imu_data = np.concatenate((acc_aligned, gyr_aligned), axis=1)
        data_df = pd.DataFrame(
            imu_data,
            columns=["acc_is", "acc_ml", "acc_ap", "gyr_is", "gyr_ml", "gyr_ap"]
        )

        # Store in dictionary
        cached_data.append({
            'fname': fname,
            'subject_id': subject_id,
            'cohort': cohort,
            'test_type': test_type,
            'data_df': data_df,
            'Y': Y,
            'quality_mask': quality_mask
        })

    print(f"Successfully cached {len(cached_data)} valid files into memory.\n")
    return cached_data


# =============================================================================
# 3. FAST MATCHING LOGIC (NumPy arrays instead of DataFrames)
# =============================================================================

def match_events(pred_ic, pred_sides, ref_dict, tol_samples, mask):
    """
    Fast logic to match predicted ICs with reference ICs using NumPy arrays.
    Eliminates pandas overhead for maximum optimization speed.
    """
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return []

    # Matching boundaries
    w_start = valid_idx[0] - int(WINDOW_TOL_SEC * SAMPLING_RATE)
    w_end = valid_idx[-1] + int(WINDOW_TOL_SEC * SAMPLING_RATE)

    # 1. Extract and filter Reference events
    ref_idx_list = []
    ref_side_list = []
    for s in ['L', 'R']:
        for idx in ref_dict.get(f'{s}_IC', []):
            if w_start <= idx <= w_end:
                ref_idx_list.append(idx)
                ref_side_list.append(s)

    ref_indices = np.array(ref_idx_list, dtype=int)
    ref_sides = np.array(ref_side_list, dtype=str)

    # 2. Extract, filter, and sort Predicted events
    pred_idx_list = []
    pred_side_list = []
    if pred_ic is not None:
        for i, idx in enumerate(pred_ic):
            if w_start <= idx <= w_end:
                pred_idx_list.append(idx)
                # Safely extract side
                if hasattr(pred_sides, 'iloc'):
                    side = pred_sides.iloc[i]
                elif hasattr(pred_sides, '__getitem__') and len(pred_sides) > i:
                    side = pred_sides[i]
                else:
                    side = 'U'
                pred_side_list.append(side)

    # Sort predictions temporally
    if len(pred_idx_list) > 0:
        sort_order = np.argsort(pred_idx_list)
        pred_indices = np.array(pred_idx_list, dtype=int)[sort_order]
        pred_sides_arr = np.array(pred_side_list, dtype=str)[sort_order]
    else:
        pred_indices = np.array([], dtype=int)
        pred_sides_arr = np.array([], dtype=str)

    # 3. Matching logic using array operations
    matches = []
    used_ref_pos = set()

    if len(pred_indices) > 0 and len(ref_indices) > 0:
        for p_idx, p_side in zip(pred_indices, pred_sides_arr):
            # Calculate absolute distance to all reference indices
            dists = np.abs(ref_indices - p_idx)

            # Find indices of references within tolerance
            valid_candidates = np.where(dists <= tol_samples)[0]

            best_ref_pos = -1
            min_d = float('inf')

            # Find the closest unused reference
            for r_pos in valid_candidates:
                if r_pos not in used_ref_pos and dists[r_pos] < min_d:
                    min_d = dists[r_pos]
                    best_ref_pos = r_pos

            if best_ref_pos != -1:
                used_ref_pos.add(best_ref_pos)
                r_idx = ref_indices[best_ref_pos]
                r_side = ref_sides[best_ref_pos]
                matches.append({'PRED': p_idx, 'REF': r_idx, 'DIFF': r_idx - p_idx,
                                'SIDE': f"{p_side}{r_side}", 'TYPE': 'TP'})
            else:
                matches.append({'PRED': p_idx, 'REF': -999, 'DIFF': -999, 'SIDE': f"{p_side}N", 'TYPE': 'FP'})

    elif len(pred_indices) > 0:
        # No references, all predictions are False Positives
        for p_idx, p_side in zip(pred_indices, pred_sides_arr):
            matches.append({'PRED': p_idx, 'REF': -999, 'DIFF': -999, 'SIDE': f"{p_side}N", 'TYPE': 'FP'})

    # Any references that were never matched are False Negatives
    if len(ref_indices) > 0:
        for r_pos, (r_idx, r_side) in enumerate(zip(ref_indices, ref_sides)):
            if r_pos not in used_ref_pos:
                matches.append({'PRED': -999, 'REF': r_idx, 'DIFF': -999, 'SIDE': f"N{r_side}", 'TYPE': 'FN'})

    return matches


# =============================================================================
# 4. GENERIC OBJECTIVE FUNCTION
# =============================================================================

def objective(trial, algo_class, param_space, cached_data):
    """
    Optimizes any algorithm by dynamically sampling parameters and evaluating
    against the preloaded data cache.
    """
    # 1. Dynamic Parameter Sampling
    sampled_params = {}
    for p_name, p_conf in param_space.items():
        if p_conf['type'] == 'float':
            sampled_params[p_name] = trial.suggest_float(p_name, p_conf['low'], p_conf['high'], step=p_conf.get('step'))
        elif p_conf['type'] == 'int':
            sampled_params[p_name] = trial.suggest_int(p_name, p_conf['low'], p_conf['high'], step=p_conf.get('step'))
        elif p_conf['type'] == 'categorical':
            sampled_params[p_name] = trial.suggest_categorical(p_name, p_conf['choices'])

    # 2. Initialize Algorithm
    model = algo_class(**sampled_params)

    f1_scores = []
    raw_rows = []

    print(f"\n--- [Trial {trial.number}] Testing {algo_class.__name__} with params: {sampled_params} ---")

    # 3. Iterate over preloaded cache instead of reading files
    for data in cached_data:
        # Run Algorithm
        try:
            model.detect(data['data_df'], sampling_rate_hz=SAMPLING_RATE, plot_debug = False)
        except Exception as e:
            f1_scores.append(0.0)
            continue

        # Matching & Stats
        tol_samples = int(TOLERANCE_SEC * SAMPLING_RATE)

        # Safely extract predictions
        pred_ic = model.ic_list_['ic'].values if (hasattr(model, 'ic_list_') and model.ic_list_ is not None) else []
        pred_sides = model.ic_side_ if (hasattr(model, 'ic_side_') and model.ic_side_ is not None) else []

        matches = match_events(pred_ic, pred_sides, data['Y'], tol_samples, data['quality_mask'])

        tp = sum(1 for m in matches if m['TYPE'] == 'TP')
        fp = sum(1 for m in matches if m['TYPE'] == 'FP')
        fn = sum(1 for m in matches if m['TYPE'] == 'FN')

        # Calculate F1 for this file
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

        # Store Raw Prediction data - ADDED TRIAL NUMBER
        for m in matches:
            raw_rows.append({
                'TRIAL': trial.number,
                'ID subject': data['subject_id'],
                'Cohort': data['cohort'],
                'Test name': data['test_type'],
                'WINDOW ID': 1,
                'PREDICTED IC': m['PRED'],
                'REFERENCE IC': m['REF'],
                'DIFFERENCE': m['DIFF'],
                'SIDE': m['SIDE']
            })

    # Save results to CSV for this trial at the end of the trial
    if raw_rows:
        algo_dir = os.path.join(BASE_OUTPUT_PATH, "results", algo_class.__name__)
        os.makedirs(algo_dir, exist_ok=True)
        csv_file = os.path.join(algo_dir, "raw_prediction_IC.csv")
        # Append to the CSV. The 'TRIAL' column ensures rows are identifiable.
        pd.DataFrame(raw_rows).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
    print(f"Trial {trial.number} Finished | Mean F1: {mean_f1:.2%}")

    return mean_f1


# =============================================================================
# 5. EXECUTION LOOP
# =============================================================================

if __name__ == "__main__":
    # Gather all files
    all_files = glob.glob(os.path.join(DATA_ROOT_CONTROL, "*.csv")) + \
                glob.glob(os.path.join(DATA_ROOT_PD, "*.csv"))

    print(f"Total raw files found on disk: {len(all_files)}")

    # Preload and process data ONCE
    cached_dataset = preload_data_to_memory(all_files)

    # Proceed only if data was successfully loaded
    if not cached_dataset:
        print("Error: No valid data loaded. Exiting.")
        exit()

    # Optimize each algorithm defined in config
    for algo_name, config in algorithms_config.items():
        print(f"\n" + "=" * 50)
        print(f"OPTIMIZING ALGORITHM: {algo_name}")
        print("=" * 50)

        # Initialize/Load Optuna Study
        study = optuna.create_study(
            study_name=f"optimization_{algo_name}",
            storage=STORAGE_NAME,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Run study using the cached dataset
        study.optimize(
            lambda trial: objective(trial, config["class"], config["params"], cached_dataset),
            n_trials=TRIALS_PER_STUDY,
            show_progress_bar=True,
            # Uncomment the next line to enable multi-core processing
            n_jobs=-1
        )

        print(f"\nBest configuration for {algo_name}:")
        print(study.best_params)
        print(f"Best Mean F1 Score: {study.best_value:.2%}")