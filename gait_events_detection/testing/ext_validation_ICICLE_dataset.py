from tqdm import tqdm

from gait_events_detection.icd import (
    IcdFang,
    IcdHwang,
    IcdHwangImproved,
    IcdJarchi,
    IcdDiao,
    IcdDiaoComplete,
    IcdSeifer,
    IcdTasca,
    IcdTomc,
    IcdTcn,
    IcdFawden,
    IcdJiang,
    IcdCnn,
    IcdTransformer,
)
from testing_utils import *

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXCEL_PATH = ROOT / "metadata" / "Paolo_step_data.xlsx"
OUT_BASE = ROOT / "gait_events_detection" / "testing" / "results"

# Config
FS = 128.0
TOL = 0.25
DATA_FOLDERS = ["ICICLE-Gait_F2_SC", "ICICLE-Gait_F2_SI", "ICICLE-Gait_F3_SI", "ICICLE-Gait_F3_SC", "ICICLE-Gait_F4_SI", "ICICLE-Gait_F4_SC"]
STATIC_FOLDER = ["StaticStandToCheck"]
ALGS = {
    "Fang": IcdFang(),
    "Hwang": IcdHwang(),
    "Jarchi": IcdJarchi(),
    "DiaoIMF": IcdDiaoComplete(),
    "Tomc": IcdTomc(),
    "TCN": IcdTcn(),
    "CNN": IcdCnn(),
    "Seifer": IcdSeifer(),
    "Fawden": IcdFawden(),
    "Jiang": IcdJiang(),
    'Transformer': IcdTransformer(),
}
plot_trial_flag = False
# Load Database Once
print("Loading GaitRite Database...")
db = pd.read_excel(EXCEL_PATH)

files = get_all_h5_files_v2(DATA_DIR, exclude_strings=STATIC_FOLDER)
static_files = get_all_h5_files_v2(DATA_DIR, exclude_strings=DATA_FOLDERS)

for name, alg in ALGS.items():
    print(f"\nProcessing Algorithm: {name}")
    res_dir = OUT_BASE / name
    res_dir.mkdir(parents=True, exist_ok=True)

    all_matches = []
    trial_metrics = []

    for f in tqdm(files):
        try:
            ss_file = find_matching_h5_file(f, static_files)

            df_imu, ref_df, time_rs = preprocess_trial_data(f, db, FS, h5_file_ss=ss_file)
            alg.detect(df_imu, sampling_rate_hz=FS, plot_debug=False)

            # Extract Predictions
            p_ic_t = alg.ic_list_['ic'].values / FS
            p_ic_s = alg.ic_side_.values if hasattr(alg, 'ic_side_') else ["U"] * len(p_ic_t)
            p_fc_t = alg.fc_list_['fc'].values / FS
            p_fc_s = alg.fc_side_.values if hasattr(alg, 'fc_side_') else ["U"] * len(p_fc_t)

            # Matching
            m_ic = match_and_validate(p_ic_t, p_ic_s, ref_df, 'IC', TOL)
            m_fc = match_and_validate(p_fc_t, p_fc_s, ref_df, 'FC', TOL)

            for m in [m_ic, m_fc]:
                m['filename'] = f.name
                all_matches.append(m)

                # Basic Trial Stats
                tp, fp, fn = (m['status'] == 'TP').sum(), (m['status'] == 'FP').sum(), (m['status'] == 'FN').sum()
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                trial_metrics.append({'filename': f.name, 'type': m['type'].iloc[0], 'F1': f1,
                                      'MAE': m.loc[m['status'] == 'TP', 'diff'].abs().mean(),
                                      'Missed%': (fn / (tp + fn) * 100), 'Extra%': (fp / (tp + fn) * 100)})

            if plot_trial_flag:
                plot_trial(time_rs, df_imu['acc_is'].values,
                       pd.DataFrame({'time': p_ic_t, 'side': p_ic_s}),
                       pd.DataFrame({'time': p_fc_t, 'side': p_fc_s}),
                       ref_df, res_dir / f"{f.stem}.png")
            trial_metrics_message = f"Trial: {trial_metrics[-2]['filename']} F1: {trial_metrics[-2]['F1']}  MAE: {trial_metrics[-2]['MAE']}s"
            tqdm.write(trial_metrics_message)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    # Save Summaries
    final_matches = pd.concat(all_matches)
    final_metrics = pd.DataFrame(trial_metrics)

    final_matches.to_csv(res_dir / "matches_ss_optimized.csv", index=False)
    final_metrics.to_csv(res_dir / "metrics_per_trial_ss_optimized.csv", index=False)

    # Generate Final Figures
    print(f"Generating summary plots for {name}...")
    plot_global_errors(final_matches, res_dir / "Summary_Errors.png")
    plot_global_boxplots(final_metrics, res_dir / "Summary_Metrics.png")

print("\nAll algorithms processed successfully.")
print("Validation Complete.")