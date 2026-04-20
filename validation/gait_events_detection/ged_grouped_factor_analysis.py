"""
ged_factor_pd_colored.py

Produce per-algorithm boxplots (Control vs PD) with overlaid trial dots.
For PD trials, dots are colored by subgroup (Motor Phenotype, HY score, FOG).

Generates 3 factors x 2 event types = 6 figures, saved under:
  headwalk/gait_events_detection/testing/results/_sub_analysis_plot/

Author: adapted to user codebase
Date: 2026-02-27
"""

from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Any, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# -------------------------
# Paths / config (adjust if needed)
# -------------------------
ROOT = Path(__file__).resolve().parents[2]  # headwalk
PER_TRIAL_CSV = ROOT / "gait_events_detection" / "testing" / "results" / "_analysis_plots"/ "per_trial_metrics.csv"
METADATA_XLSX = ROOT / "metadata" / "ICICLE Gait - 20250414_Paolo.xlsx"
OUT_DIR = ROOT / "gait_events_detection" / "testing" / "results" / "_sub_analysis_plot" / "_grouped_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# regex to extract subject code and timepoint from filename:
_SUBJECT_RE = re.compile(r"(ING[CP]\d{3})(?:[^\w]|_)?(F[234])", re.IGNORECASE)

# mapping of timepoints -> metadata columns (adjust if your Excel differs)
TIMEPOINT_MAP = {
    "F2": {"age_col": "Age_t36", "hy_col": "HY_t36", "phen_col": "Motor_Phenotype_t36", "nfog_col": "NFOG_Binary_t32"},
    "F3": {"age_col": "Age_t54", "hy_col": "HY_t54", "phen_col": "Motor_Phenotype_t54", "nfog_col": "NFOG_Binary_t54"},
    "F4": {"age_col": "Age_t72", "hy_col": "HY_t72", "phen_col": "Motor_Phenotype_t72", "nfog_col": "NFOG_Binary_t72"},
}

# color palettes
COLOR_CTRL = "#1f77b4"  # blue
COLOR_PD_BASE = "#ff7f0e"  # orange (default PD)
# specific palettes for phenotypes (extend if needed)
PHEN_PALETTE = {"Tremor": "#2ca02c", "PIGD": "#d62728", "ID": "#9467bd", "Unknown": "#888888"}
# palette for FOG
FOG_PALETTE = {"FOG": "#8c564b", "NoFOG": "#e377c2", "Unknown": "#888888"}

# -------------------------
# Utilities
# -------------------------
def extract_subject_and_tp(filename: str) -> Tuple[str | None, str | None]:
    m = _SUBJECT_RE.search(filename)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).upper()

def nfog_normalize(x) -> str:
    if pd.isna(x):
        return "Unknown"
    xs = str(x).strip()
    if xs in {"1", "Yes", "Y", "TRUE", "True", "true"}:
        return "FOG"
    if xs in {"0", "No", "N", "FALSE", "False", "false"}:
        return "NoFOG"
    # sometimes Excel has numeric 0/1 floats
    try:
        if float(xs) == 1.0:
            return "FOG"
        if float(xs) == 0.0:
            return "NoFOG"
    except Exception:
        pass
    return xs or "Unknown"

def safe_groups_ordered(algo_list: List[str]) -> List[str]:
    """Return deterministic ordering of algorithms (alphabetical)."""
    return sorted(algo_list)

def jitter_x(x: float, scale: float = 0.08) -> float:
    return x + np.random.normal(scale=scale)

# -------------------------
# Load & merge metrics + metadata
# -------------------------
def load_and_merge() -> pd.DataFrame:
    if not PER_TRIAL_CSV.exists():
        raise FileNotFoundError(f"Per-trial CSV not found: {PER_TRIAL_CSV}")
    metrics = pd.read_csv(PER_TRIAL_CSV)
    metrics.columns = [c.strip() for c in metrics.columns]

    # minimal required columns
    req = {"filename", "cohort", "event_type", "algo", "f1", "mae"}
    missing = req - set(metrics.columns)
    if missing:
        raise ValueError(f"Input metrics missing columns: {missing}")

    # load metadata excel
    meta = pd.read_excel(METADATA_XLSX)
    if "First_name" not in meta.columns:
        raise ValueError("Metadata Excel missing 'First_name' column")
    # ------------------------------------------------------------------
    # Normalize Motor Phenotype columns (numeric codes -> string labels)
    # ------------------------------------------------------------------

    phen_mapping = {
        1: "PIGD",
        1.0: "PIGD",
        2: "ID",
        2.0: "ID",
        3: "Tremor",
        3.0: "Tremor",
    }

    for tp in TIMEPOINT_MAP.values():
        col = tp["phen_col"]
        if col in meta.columns:
            meta[col] = (
                meta[col]
                .replace(phen_mapping)  # numeric mapping
                .astype(str)  # convert everything to string
                .str.strip()  # remove spaces
                .replace({"nan": "Unknown"})  # handle NaN cast to string
            )
    # extract subject/timepoint
    metrics["subject_code"], metrics["timepoint"] = zip(*metrics["filename"].map(lambda x: extract_subject_and_tp(str(x))))
    # join: pick for each trial the matching subject row from metadata and timepoint-specific columns
    ages, hys, phens, nfogs = [], [], [], []
    for _, row in metrics.iterrows():
        subj = row["subject_code"]
        tp = row["timepoint"]
        if pd.isna(subj) or pd.isna(tp):
            ages.append(np.nan); hys.append(np.nan); phens.append("Unknown"); nfogs.append("Unknown"); continue
        mr = meta[meta["First_name"].astype(str).str.upper() == subj.upper()]
        if mr.empty:
            ages.append(np.nan); hys.append(np.nan); phens.append("Unknown"); nfogs.append("Unknown"); continue
        mr0 = mr.iloc[0]
        if tp not in TIMEPOINT_MAP:
            ages.append(np.nan); hys.append(np.nan); phens.append("Unknown"); nfogs.append("Unknown"); continue
        mapping = TIMEPOINT_MAP[tp]
        ages.append(mr0.get(mapping["age_col"], np.nan))
        hys.append(mr0.get(mapping["hy_col"], np.nan))
        phens.append(mr0.get(mapping["phen_col"], "Unknown"))
        nfogs.append(mr0.get(mapping["nfog_col"], np.nan))
    metrics["age"] = ages
    metrics["hy"] = pd.to_numeric(hys, errors="coerce")
    metrics["phenotype"] = [str(x) if pd.notna(x) else "Unknown" for x in phens]
    metrics["nfog_raw"] = nfogs
    metrics["nfog_flag"] = [nfog_normalize(x) for x in nfogs]

    # ensure event_type uppercase and cohort uppercase
    metrics["event_type"] = metrics["event_type"].astype(str).str.upper()
    metrics["cohort"] = metrics["cohort"].astype(str).str.upper()

    return metrics

# -------------------------
# Plotting function
# -------------------------
def plot_ctrl_vs_pd_colored(
    df: pd.DataFrame,
    factor_col: str,
    pd_palette: Dict[str, str],
    factor_name: str,
    event_type: str,
    out_path: Path,
):
    """
    Draw side-by-side Control vs PD boxplots per algorithm.
    - For controls: box colored blue, points blue semi-transparent
    - For PD: box colored orange, points colored by subgroup according to pd_palette
    Saves PNG to out_path.
    """
    # list of algorithms to show (ordered)
    algos = safe_groups_ordered(df["algo"].unique().tolist())
    n_algos = len(algos)
    spacing = 3.0
    offset = 0.5
    width = 0.9

    fig_w = max(10, 1.2 * n_algos)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 9), sharex=True)
    fig.suptitle(f"{event_type} — {factor_name} — Control (blue) vs PD (colored-by-group)", fontsize=14)

    # iterate algorithms, build two boxes each: C at base-offset, P at base+offset
    positions = []
    box_data_f1 = []
    box_data_mae = []
    box_positions = []
    xticks = []
    for i, algo in enumerate(algos):
        base = i * spacing
        xticks.append(base)
        # control data
        sub_c = df[(df["algo"] == algo) & (df["cohort"] == "C") & (df["event_type"] == event_type)]
        c_f1 = sub_c["f1"].dropna().astype(float).to_numpy()
        c_mae = sub_c["mae"].dropna().astype(float).to_numpy()
        # pd data
        sub_p = df[(df["algo"] == algo) & (df["cohort"] == "P") & (df["event_type"] == event_type)]
        p_f1 = sub_p["f1"].dropna().astype(float).to_numpy()
        p_mae = sub_p["mae"].dropna().astype(float).to_numpy()

        # positions
        pos_c = base - offset
        pos_p = base + offset

        # append in order: control then pd (matching earlier code pattern)
        box_data_f1.append(c_f1)
        box_positions.append(pos_c)
        box_data_f1.append(p_f1)
        box_positions.append(pos_p)

        box_data_mae.append(c_mae)
        box_data_mae.append(p_mae)

    # Draw F1 boxplots
    bp1 = ax1.boxplot(box_data_f1, positions=box_positions, widths=width, patch_artist=True, showfliers=False)
    # Color the boxes alternately: control blue, pd orange
    for k, patch in enumerate(bp1["boxes"]):
        if k % 2 == 0:
            patch.set_facecolor(COLOR_CTRL)
        else:
            patch.set_facecolor(COLOR_PD_BASE)
        patch.set_alpha(0.7)
    ax1.set_ylabel("F1")
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(axis="y", alpha=0.3)

    # Draw MAE boxplots
    bp2 = ax2.boxplot(box_data_mae, positions=box_positions, widths=width, patch_artist=True, showfliers=False)
    for k, patch in enumerate(bp2["boxes"]):
        if k % 2 == 0:
            patch.set_facecolor(COLOR_CTRL)
        else:
            patch.set_facecolor(COLOR_PD_BASE)
        patch.set_alpha(0.7)
    ax2.set_ylabel("MAE (s)")
    ax2.grid(axis="y", alpha=0.3)

    # Now overlay scatter points (jittered)
    rng = np.random.default_rng(42)
    for i, algo in enumerate(algos):
        base = i * spacing
        pos_c = base - offset
        pos_p = base + offset

        # Controls: small blue dots
        sub_c = df[(df["algo"] == algo) & (df["cohort"] == "C") & (df["event_type"] == event_type)]
        x_c = [jitter_x(pos_c, scale=0.06) for _ in range(len(sub_c))]
        y_c_f1 = sub_c["f1"].astype(float).to_numpy()
        y_c_mae = sub_c["mae"].astype(float).to_numpy()
        ax1.scatter(x_c, y_c_f1, color=COLOR_CTRL, alpha=0.5, s=10, edgecolors="none")
        ax2.scatter(x_c, y_c_mae, color=COLOR_CTRL, alpha=0.5, s=10, edgecolors="none")

        # PD: color dots by subgroup (factor_col)
        sub_p = df[(df["algo"] == algo) & (df["cohort"] == "P") & (df["event_type"] == event_type)]
        if sub_p.empty:
            continue
        # determine subgroup values and colors
        group_vals = sub_p[factor_col].fillna("Unknown").astype(str).tolist()
        colors = [pd_palette.get(g, "#777777") for g in group_vals]

        x_p = [jitter_x(pos_p, scale=0.06) for _ in range(len(sub_p))]
        y_p_f1 = sub_p["f1"].astype(float).to_numpy()
        y_p_mae = sub_p["mae"].astype(float).to_numpy()
        ax1.scatter(x_p, y_p_f1, color=colors, alpha=0.9, s=18, edgecolors="k", linewidths=0.15)
        ax2.scatter(x_p, y_p_mae, color=colors, alpha=0.9, s=18, edgecolors="k", linewidths=0.15)

    # X ticks / labels
    ax2.set_xticks([i * spacing for i in range(len(algos))])
    ax2.set_xticklabels(algos, rotation=30, ha="right")

    # legend for cohorts & subgroup palette (create custom handles)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=COLOR_CTRL, label="Control (C)", alpha=0.7),
               mpatches.Patch(facecolor=COLOR_PD_BASE, label="PD (box)", alpha=0.7)]
    # subgroup handles (only unique groups in palette)
    subgroup_names = list(pd_palette.keys())
    for name in subgroup_names:
        handles.append(mpatches.Patch(facecolor=pd_palette[name], label=str(name), alpha=0.9))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.92))

    plt.tight_layout(rect=[0, 0.03, 0.94, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] {out_path}")

# -------------------------
# Statistical tests (Kruskal-Wallis + pairwise MWU Bonferroni)
# -------------------------
def pairwise_mwu_with_bonferroni(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(groups.keys())
    comparisons = []
    for a, b in combinations(names, 2):
        x = groups[a]; y = groups[b]
        x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            stat, p = np.nan, np.nan
        else:
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        comparisons.append({"g1": a, "g2": b, "stat": float(stat) if not np.isnan(stat) else np.nan,
                            "p_uncorrected": float(p) if not np.isnan(p) else np.nan})
    m = len(comparisons)
    for c in comparisons:
        if not np.isnan(c["p_uncorrected"]):
            c["p_bonf"] = min(1.0, c["p_uncorrected"] * m)
        else:
            c["p_bonf"] = np.nan
    return pd.DataFrame(comparisons)

def run_tests_by_factor(df: pd.DataFrame, factor_col: str, event_type: str) -> Dict[str, Any]:
    """
    Run Kruskal-Wallis omnibus and pairwise tests (if applicable) for F1 and MAE.
    Returns dict with results and stores CSVs under OUT_DIR.
    """
    res = {}
    df_e = df[df["event_type"] == event_type].copy()
    # restrict to PD for subgroup testing (we color PD differentially)
    df_pd = df_e[df_e["cohort"] == "P"].copy()
    if df_pd.empty:
        print(f"[INFO] No PD rows for {event_type}; skipping stats for {factor_col}")
        return res

    groups = {}
    for name, g in df_pd.groupby(factor_col):
        arr_f1 = g["f1"].dropna().astype(float).to_numpy()
        arr_mae = g["mae"].dropna().astype(float).to_numpy()
        groups[name] = {"f1": arr_f1, "mae": arr_mae}

    # prepare arrays for Kruskal
    group_names = [n for n in groups.keys() if len(groups[n]["f1"]) > 0]
    res["f1"] = {"omnibus": None, "posthoc": None}
    res["mae"] = {"omnibus": None, "posthoc": None}

    if len(group_names) >= 2:
        # F1 omnibus
        try:
            f1_args = [groups[n]["f1"] for n in group_names]
            H_f1, p_f1 = stats.kruskal(*f1_args)
            res["f1"]["omnibus"] = {"H": float(H_f1), "p": float(p_f1), "n_groups": len(group_names)}
        except Exception as e:
            res["f1"]["omnibus"] = {"H": np.nan, "p": np.nan, "error": str(e)}
        # MAE omnibus
        try:
            mae_args = [groups[n]["mae"] for n in group_names]
            H_mae, p_mae = stats.kruskal(*mae_args)
            res["mae"]["omnibus"] = {"H": float(H_mae), "p": float(p_mae), "n_groups": len(group_names)}
        except Exception as e:
            res["mae"]["omnibus"] = {"H": np.nan, "p": np.nan, "error": str(e)}

        # posthoc pairwise
        # build dict name -> array for a metric and run pairwise comparisons
        f1_groups = {n: groups[n]["f1"] for n in group_names}
        mae_groups = {n: groups[n]["mae"] for n in group_names}
        post_f1 = pairwise_mwu_with_bonferroni(f1_groups)
        post_mae = pairwise_mwu_with_bonferroni(mae_groups)
        res["f1"]["posthoc"] = post_f1
        res["mae"]["posthoc"] = post_mae

        # save csvs
        post_f1.to_csv(OUT_DIR / f"{factor_col}_{event_type}_posthoc_f1.csv", index=False)
        post_mae.to_csv(OUT_DIR / f"{factor_col}_{event_type}_posthoc_mae.csv", index=False)
    else:
        print(f"[INFO] Not enough groups for factor {factor_col} (found {len(group_names)}). No omnibus/posthoc.")

    # save omnibus summary
    with open(OUT_DIR / f"{factor_col}_{event_type}_omnibus.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    return res

# -------------------------
# Main routine
# -------------------------
def main():
    metrics = load_and_merge()

    # Focus only the three requested factors and produce 2 figs each (IC/FC)
    factors = [
        ("phenotype", PHEN_PALETTE, "MotorPhenotype"),
        ("hy", { }, "HY_score"),  # will color PD dots by HY value (we build palette on the fly)
        ("nfog_flag", FOG_PALETTE, "FOG"),
    ]

    algos = safe_groups_ordered(metrics["algo"].unique().tolist())
    print(f"Found algorithms: {algos}")

    for factor_col, palette, factor_name in factors:
        # Build palette for HY dynamically (use colormap) if empty palette passed
        if factor_col == "hy":
            # convert HY to integers and treat NaN as "Unknown"
            metrics["hy_cat"] = metrics["hy"].apply(lambda x: f"HY{int(x)}" if pd.notna(x) else "Unknown")
            # unique HY categories in PD
            hy_vals = sorted(metrics[metrics["cohort"] == "P"]["hy_cat"].dropna().unique(), key=lambda x: str(x))
            cmap = plt.get_cmap("tab10")
            hy_palette = {}
            for i, hv in enumerate(hy_vals):
                hy_palette[hv] = cmap(i % 10)
            pd_palette = hy_palette
            plot_factor_col = "hy_cat"
        else:
            pd_palette = palette
            plot_factor_col = factor_col

        for event in ["IC", "FC"]:
            out_png = OUT_DIR / f"{factor_name}_{event}_Control_vs_PD_colored.png"
            print(f"[INFO] Plotting factor={factor_col} event={event} -> {out_png}")
            plot_ctrl_vs_pd_colored(metrics, plot_factor_col, pd_palette, factor_name, event, out_png)

            # run stats (PD-only) and save results
            res = run_tests_by_factor(metrics, plot_factor_col, event)
            print(f"[INFO] Stats saved for {factor_name} {event}")

    print("All done. Figures & stats saved in:", OUT_DIR)

if __name__ == "__main__":
    main()