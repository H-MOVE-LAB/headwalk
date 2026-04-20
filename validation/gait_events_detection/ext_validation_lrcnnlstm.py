"""
test_lrcnnlstm_iciclepd.py

Evaluate the CNN-LSTM laterality classifier (LrCnnlstm) on ICICLE-PD using
REFERENCE events (IC and FC) as inputs.

Because we feed reference events to the classifier:
- All events are "matched" by construction
- There are no FP/FN for timing
- We only evaluate LEFT/RIGHT classification accuracy

Outputs (same folder structure style as existing ICD testing)
------------------------------------------------------------
himu_pd/gait_events_detection/testing/results/CNNLSTM_0p1/
    matches.csv            # one row per event with ref side + predicted side
    metrics_per_trial.csv  # per-trial accuracy (overall, left, right) for IC and FC

Notes
-----
- This script assumes your testing_utils.py provides:
    - preprocess_trial_data(h5_file, db, FS) -> (df_imu, ref_df, time_rs)
    - get_all_h5_files(data_dir, exclude)
  and that ref_df contains reference IC/FC times and sides.

- Expected reference format (common in your pipeline):
    ref_df columns include at least:
      - 'type' in {'IC','FC'}
      - 'time' in seconds
      - 'side' in {'L','R'}  (or {'Left','Right'}; handled below)

- LrCnnlstm expects IC/FC indices in SAMPLES at the input sampling rate.

- By default LrCnnlstm uses acc_ml, gyr_ap, gyr_is and half_window_s=0.5.

Author: (generated)
"""

from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.headwalk.gait_events_detection import LrCnnlstm  # ensure it is exported in algo/__init__.py
from testing_utils import preprocess_trial_data, get_all_h5_files


# -----------------------
# --- CONFIG ------------
# -----------------------
FS = 128.0
EXCLUDE = []

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXCEL_PATH = ROOT / "metadata" / "Paolo_step_data.xlsx"

OUT_BASE = ROOT / "gait_events_detection" / "testing" / "results"
RES_DIR = OUT_BASE / "CNNLSTM_0p3"
RES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_side(side) -> str:
    """
    Normalize side labels to 'L'/'R'.
    Accepts: 'L','R','Left','Right','left','right', 0/1, etc.
    """
    if side is None or (isinstance(side, float) and np.isnan(side)):
        return "U"
    s = str(side).strip().lower()
    if s in {"l", "left", "0"}:
        return "L"
    if s in {"r", "right", "1"}:
        return "R"
    return "U"


def _trial_accuracy(df_events: pd.DataFrame) -> dict:
    """
    Compute overall/left/right accuracy from a matches subset (single filename + type).
    df_events must include columns: ref_side, pred_side
    """
    ref = df_events["side_ref"].values
    pred = df_events["side_pred"].values

    valid = (ref != "U") & (pred != "U")
    if not np.any(valid):
        return {"overall_acc": np.nan, "left_acc": np.nan, "right_acc": np.nan, "n": int(len(ref))}

    ref_v = ref[valid]
    pred_v = pred[valid]

    overall = float(np.mean(ref_v == pred_v))

    left_mask = ref_v == "L"
    right_mask = ref_v == "R"

    left_acc = float(np.mean(pred_v[left_mask] == "L")) if np.any(left_mask) else np.nan
    right_acc = float(np.mean(pred_v[right_mask] == "R")) if np.any(right_mask) else np.nan

    return {
        "overall_acc": overall,
        "left_acc": left_acc,
        "right_acc": right_acc,
        "n": int(len(ref_v)),
        "n_left": int(np.sum(left_mask)),
        "n_right": int(np.sum(right_mask)),
    }


def main():
    print("Loading ICICLE-PD reference database...")
    db = pd.read_excel(EXCEL_PATH)

    files = get_all_h5_files(DATA_DIR, EXCLUDE)
    print(f"[INFO] Found {len(files)} trials.")

    # Instantiate the classifier once (models are lazy-loaded inside)
    lr = LrCnnlstm()

    all_rows = []
    trial_metrics = []

    for f in tqdm(files):
        try:
            df_imu, ref_df, time_rs = preprocess_trial_data(f, db, FS)

            # --- extract reference events ---
            ref_df = ref_df.copy()
            if "side" in ref_df.columns:
                ref_df["side"] = ref_df["side"].apply(_normalize_side)

            # IC
            ic_t = ref_df["IC"].to_numpy(dtype=float)  # seconds
            ic_side = ref_df["side"].to_numpy(dtype=object)

            # FC
            fc_t = ref_df["FC"].to_numpy(dtype=float)
            fc_side = ref_df["side"].to_numpy(dtype=object)

            # Convert ref times to sample indices at FS
            ic_idx = np.asarray(np.round(ic_t * FS), dtype=int)
            fc_idx = np.asarray(np.round(fc_t * FS), dtype=int) if len(fc_t) else None

            # Run laterality classifier using reference indices
            lr.detect(df_imu, sampling_rate_hz=FS, ic=ic_idx, fc=fc_idx)

            # Extract predicted sides aligned to the classifier outputs.
            # NOTE: LrCnnlstm internally drops events that would produce out-of-bounds windows.
            # We therefore join by nearest time within a tiny tolerance in sample domain.
            # For ICICLE-PD, reference events near edges are rare; still, we handle robustly.

            # Pred IC
            p_ic_idx = lr.ic_list_["ic"].to_numpy(dtype=int) if len(lr.ic_list_) else np.asarray([], dtype=int)
            p_ic_side = lr.ic_side_.to_numpy(dtype=object) if len(lr.ic_list_) else np.asarray([], dtype=object)

            # Pred FC
            p_fc_idx = lr.fc_list_["fc"].to_numpy(dtype=int) if hasattr(lr, "fc_list_") and len(lr.fc_list_) else np.asarray([], dtype=int)
            p_fc_side = lr.fc_side_.to_numpy(dtype=object) if hasattr(lr, "fc_side_") and len(p_fc_idx) else np.asarray([], dtype=object)

            # Helper: map predictions to reference rows by exact index match (preferred), else nearest within tol
            def align_pred_to_ref(ref_idx: np.ndarray, ref_side_arr: np.ndarray,
                                  pred_idx: np.ndarray, pred_side_arr: np.ndarray,
                                  event_type: str, tol_samples: int = 2):
                rows = []
                if len(ref_idx) == 0:
                    return rows

                # Build a quick lookup for exact matches
                pred_map = {int(i): str(s) for i, s in zip(pred_idx, pred_side_arr)}

                for ridx, rside in zip(ref_idx, ref_side_arr):
                    ridx = int(ridx)
                    # exact
                    if ridx in pred_map:
                        pside = pred_map[ridx]
                        used_idx = ridx
                    else:
                        # nearest within tolerance
                        if len(pred_idx) == 0:
                            pside = "U"
                            used_idx = None
                        else:
                            j = int(np.argmin(np.abs(pred_idx - ridx)))
                            if abs(int(pred_idx[j]) - ridx) <= tol_samples:
                                used_idx = int(pred_idx[j])
                                pside = str(pred_side_arr[j])
                            else:
                                used_idx = None
                                pside = "U"

                    rows.append(
                        {
                            "filename": f.name,
                            "type": event_type,
                            "time_ref": float(ridx) / FS,
                            "ref_index": ridx,
                            "side_ref": _normalize_side(rside),
                            "pred_index": used_idx if used_idx is not None else np.nan,
                            "time_pred": float(used_idx) / FS if used_idx is not None else np.nan,
                            "side_pred": _normalize_side(pside),
                            # for compatibility with other scripts naming
                            "status": "TP",     # always "matched" (timing not evaluated here)
                            "diff": 0.0,        # timing error forced to 0
                            "correct_side": bool(_normalize_side(pside) == _normalize_side(rside)) if _normalize_side(pside) != "U" else False,
                        }
                    )
                return rows

            ic_rows = align_pred_to_ref(ic_idx, ic_side, p_ic_idx, p_ic_side, "IC")
            fc_rows = align_pred_to_ref(fc_idx if fc_idx is not None else np.asarray([], dtype=int),
                                        fc_side, p_fc_idx, p_fc_side, "FC")

            all_rows.extend(ic_rows)
            all_rows.extend(fc_rows)

            # --- per-trial metrics (IC / FC separately) ---
            if ic_rows:
                df_ic = pd.DataFrame(ic_rows)
                m = _trial_accuracy(df_ic)
                trial_metrics.append(
                    {
                        "filename": f.name,
                        "type": "IC",
                        **m,
                    }
                )
            if fc_rows:
                df_fc = pd.DataFrame(fc_rows)
                m = _trial_accuracy(df_fc)
                trial_metrics.append(
                    {
                        "filename": f.name,
                        "type": "FC",
                        **m,
                    }
                )

            # Progress message (IC overall)
            if ic_rows:
                msg = f"Trial: {f.name} | IC Acc: {trial_metrics[-(2 if fc_rows else 1)]['overall_acc']:.2%}"
                tqdm.write(msg)

        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    # -----------------------
    # --- SAVE OUTPUTS ------
    # -----------------------
    if not all_rows:
        print("[ERROR] No events processed. Check inputs and reference format.")
        return

    matches_df = pd.DataFrame(all_rows)
    metrics_df = pd.DataFrame(trial_metrics)

    matches_df.to_csv(RES_DIR / "matches.csv", index=False)
    metrics_df.to_csv(RES_DIR / "metrics_per_trial.csv", index=False)

    print(f"\n[SAVED] {RES_DIR / 'matches.csv'}")
    print(f"[SAVED] {RES_DIR / 'metrics_per_trial.csv'}")

    # Optional: global summary prints
    for et in ["IC", "FC"]:
        sub = matches_df[matches_df["type"] == et].copy()
        if len(sub) == 0:
            continue
        acc = float(np.mean(sub["side_ref"].values == sub["side_pred"].values))
        left_mask = sub["side_ref"].values == "L"
        right_mask = sub["side_ref"].values == "R"
        left_acc = float(np.mean(sub.loc[left_mask, "side_pred"].values == "L")) if np.any(left_mask) else np.nan
        right_acc = float(np.mean(sub.loc[right_mask, "side_pred"].values == "R")) if np.any(right_mask) else np.nan
        print(f"\n[GLOBAL] {et} Overall Acc: {acc:.2%} | Left Acc: {left_acc:.2%} | Right Acc: {right_acc:.2%}")


if __name__ == "__main__":
    main()