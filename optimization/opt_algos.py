import os
import glob
import numpy as np
import pandas as pd
import optuna

# Import preprocessing utilities from your existing file
from src.headwalk.utils import (
    load_head_data,
    rotate_to_gravity,
    compute_quality_mask,
)

# Import algorithms
from src.headwalk.gait_events_detection import (
    GedFang, GedTomc
)

# =============================================================================
# 1. CONFIGURATION & SEARCH SPACE
# =============================================================================

# Define algorithms and their hyperparameter search space here
# This makes the objective function independent of the specific algorithm
algorithms_config = {
    "Fang": {
        "class": GedFang,
        "params": {
            "lowpass_hz": {"type": "float", "low": 2.6, "high": 4.0, "step": 0.2},
            "wavelet_type": {"type": "categorical", "choices": ["gaus1", "gaus2"]}
        }
    },
    "Tomc": {
        "class": GedTomc,
        "params": {
            # AFO gain (Dynamics)
            "k_omega": {"type": "float", "low": 1.0, "high": 10.0, "step": 0.5},
            "k_phi": {"type": "float", "low": 1.0, "high": 10.0, "step": 0.5},
            "k_alpha": {"type": "float", "low": 0.05, "high": 0.5, "step": 0.05},
            "k0": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1},

            # Structural parameters
            "omega0": {"type": "float", "low": 0.5, "high": 1.5, "step": 0.1},
        }
    },
    # "Jarchi": {
    #     "class": IcdJarchi,
    #     "params": {
    #         "window_size": {"type": "int", "low": 100, "high": 400, "step": 50},
    #         "min_step_duration_s": {"type": "float", "low": 0.3, "high": 0.5, "step": 0.05}
    #     }
    # },
}

# Paths
BASE_OUTPUT_PATH = "C:/algo_optimization/"
DATA_ROOT_CONTROL = r"X:\Visiting_Staff\2025 - Paolo Tasca\Code\weargaitpd_processing\data\Version 1\CONTROL PARTICIPANTS\CSV files"
DATA_ROOT_PD = r"X:\Visiting_Staff\2025 - Paolo Tasca\Code\weargaitpd_processing\data\Version 1\PD PARTICIPANTS\CSV files"

# Optuna Database
DB_PATH = os.path.join(BASE_OUTPUT_PATH, "weargait_optimization.db")
STORAGE_NAME = f"sqlite:///{DB_PATH}"

# Global Params
SAMPLING_RATE = 100.0
TOLERANCE_SEC = 0.25  # Matching tolerance
WINDOW_TOL_SEC = 0.26  # Boundary tolerance for GT inclusion
TRIALS_PER_STUDY = 20

os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)


# =============================================================================
# 2. MATCHING LOGIC
# =============================================================================

def match_events(pred_ic, pred_sides, ref_dict, tol_samples, mask):
    """
    Generic logic to match predicted ICs with reference ICs.
    """
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0: return []

    # Matching boundaries
    w_start = valid_idx[0] - int(WINDOW_TOL_SEC * SAMPLING_RATE)
    w_end = valid_idx[-1] + int(WINDOW_TOL_SEC * SAMPLING_RATE)

    # Reference events within window
    refs = []
    for s in ['L', 'R']:
        for idx in ref_dict.get(f'{s}_IC', []):
            if w_start <= idx <= w_end:
                refs.append({'idx': idx, 'side': s})
    df_ref = pd.DataFrame(refs)

    # Predicted events within window
    preds = []
    if pred_ic is not None:
        for i, idx in enumerate(pred_ic):
            if w_start <= idx <= w_end:
                side = pred_sides.iloc[i] if hasattr(pred_sides, 'iloc') else 'U'
                preds.append({'idx': idx, 'side': side})
    df_pred = pd.DataFrame(preds)

    matches = []
    used_ref = set()

    if not df_pred.empty and not df_ref.empty:
        for _, p in df_pred.sort_values('idx').iterrows():
            dists = np.abs(df_ref['idx'] - p['idx'])
            valid_candidates = dists[dists <= tol_samples]

            best_ref, min_d = -1, float('inf')
            for r_idx, d in valid_candidates.items():
                if r_idx not in used_ref and d < min_d:
                    min_d = d;
                    best_ref = r_idx

            if best_ref != -1:
                used_ref.add(best_ref)
                r_row = df_ref.loc[best_ref]
                matches.append({'PRED': p['idx'], 'REF': r_row['idx'], 'DIFF': r_row['idx'] - p['idx'],
                                'SIDE': f"{p['side']}{r_row['side']}", 'TYPE': 'TP'})
            else:
                matches.append({'PRED': p['idx'], 'REF': -999, 'DIFF': -999, 'SIDE': f"{p['side']}N", 'TYPE': 'FP'})

    elif not df_pred.empty:
        for _, p in df_pred.iterrows():
            matches.append({'PRED': p['idx'], 'REF': -999, 'DIFF': -999, 'SIDE': f"{p['side']}N", 'TYPE': 'FP'})

    if not df_ref.empty:
        for r_idx, r in df_ref.iterrows():
            if r_idx not in used_ref:
                matches.append({'PRED': -999, 'REF': r['idx'], 'DIFF': -999, 'SIDE': f"N{r['side']}", 'TYPE': 'FN'}) # NA --> N
    return matches


# =============================================================================
# 3. GENERIC OBJECTIVE FUNCTION
# =============================================================================

def objective(trial, algo_class, param_space, file_list):
    """
    Optimizes any algorithm by dynamically sampling parameters from param_space.
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

    print(f"\n--- [Trial {trial.number}] Testing {algo_class.__name__} ---")

    for fpath in file_list:
        fname = os.path.basename(fpath)
        if "balance" in fpath or "tandem" in fpath or "hc143" in fname:
            print(f"Skipped {fname}")
            continue
        subject_id = fname.split('_')[0]
        cohort = "CONTROL" if "CONTROL" in fpath else "PD"
        test_type = fname.split('_')[1] if len(fname.split('_')) > 1 else "unknown"

        # Preprocessing Steps (using imported functions)
        X, Y, L_cont, R_cont = load_head_data(fpath)
        if X is None: continue

        acc_aligned, gyr_aligned = rotate_to_gravity(X[:, :3], X[:, 3:])
        # Combine acc and gyr data
        imu_data = np.concatenate((acc_aligned, gyr_aligned), axis=1)
        data_df = pd.DataFrame(imu_data,
                               columns=["acc_is", "acc_ml", "acc_ap", "gyr_is", "gyr_ml", "gyr_ap"],)

        quality_mask = compute_quality_mask(X, L_cont, R_cont)
        if not np.any(quality_mask): continue

        # Run Algorithm
        try:
            model.detect(data_df, sampling_rate_hz=SAMPLING_RATE)
            # plot_events_single_algo(SAMPLING_RATE, data_df, events, alg)
        except Exception as e:
            f1_scores.append(0.0)
            continue

        # Matching & Stats
        tol_samples = int(TOLERANCE_SEC * SAMPLING_RATE)
        matches = match_events(model.ic_list_['ic'].values, model.ic_side_, Y, tol_samples, quality_mask)

        tp = sum(1 for m in matches if m['TYPE'] == 'TP')
        fp = sum(1 for m in matches if m['TYPE'] == 'FP')
        fn = sum(1 for m in matches if m['TYPE'] == 'FN')

        # Calculate F1 for this trial
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

        print(f"File: {fname} | TP: {tp}, FP: {fp}, FN: {fn} | F1: {f1:.2%}")

        # Store Raw Prediction data
        for m in matches:
            raw_rows.append({
                'ID subject': subject_id, 'Cohort': cohort, 'Test name': test_type,
                'WINDOW ID': 1, 'PREDICTED IC': m['PRED'], 'REFERENCE IC': m['REF'],
                'DIFFERENCE': m['DIFF'], 'SIDE': m['SIDE']
            })

    # Save results to CSV for this algorithm
    if raw_rows:
        algo_dir = os.path.join(BASE_OUTPUT_PATH, "results", algo_class.__name__)
        os.makedirs(algo_dir, exist_ok=True)
        csv_file = os.path.join(algo_dir, "raw_prediction_IC.csv")
        pd.DataFrame(raw_rows).to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    return np.mean(f1_scores) if f1_scores else 0.0


# =============================================================================
# 4. EXECUTION LOOP
# =============================================================================

# Gather all files
all_files = glob.glob(os.path.join(DATA_ROOT_CONTROL, "*.csv")) + \
            glob.glob(os.path.join(DATA_ROOT_PD, "*.csv"))

print(f"Total files found: {len(all_files)}")

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

    # Run study
    study.optimize(
        lambda trial: objective(trial, config["class"], config["params"], all_files),
        n_trials=TRIALS_PER_STUDY,
        show_progress_bar=True
    )

    print(f"\nBest configuration for {algo_name}:")
    print(study.best_params)
    print(f"Best Mean F1 Score: {study.best_value:.2%}")