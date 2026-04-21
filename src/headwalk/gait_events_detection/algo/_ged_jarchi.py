import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from pyts.decomposition import SingularSpectrumAnalysis
from scipy.signal import find_peaks, argrelextrema


class GedJarchi:
    """
    Independent implementation of Jarchi et al. (2014) Gait Event Detection.

    This class follows the required output format:
    - self.ic_list_: pd.DataFrame with column "ic"
    - self.fc_list_: pd.DataFrame with column "fc" (Toe-Offs)
    - self.ic_side_: pd.Series with "L"/"R"
    - self.fc_side_: pd.Series with "L"/"R"
    """

    def __init__(self, window_size: int = 200,
                 enable_enhanced_cycles: bool = True,
                 radius_s : float = 0.05):
        self.radius_s = radius_s
        self.window_size = window_size
        self.enable_enhanced_cycles = enable_enhanced_cycles

        # SSA config: Trend [0], Dominant Oscillation [1, 2], Noise [3+]
        self.ssa_grouper = SingularSpectrumAnalysis(
            window_size=window_size,
            groups=[[0], [1, 2], np.arange(3, window_size, 1)]
        )

        # Output placeholders
        self.ic_list_ = pd.DataFrame(columns=["ic"])
        self.fc_list_ = pd.DataFrame(columns=["fc"])
        self.ic_side_ = pd.Series(dtype=str)
        self.fc_side_ = pd.Series(dtype=str)

        self._debug_data = {}

    def detect(self, data: pd.DataFrame, sampling_rate_hz: float, plot_debug: bool = True):
        """
        Detects gait events and stores them in the class attributes.
        Expected columns: 'acc_ap', 'acc_is', 'acc_ml'.
        """
        self.sampling_rate_hz = sampling_rate_hz

        if data is None or data.empty:
            return self._handle_empty_output()
        data['acc_ap'] = -data['acc_ap'] # needed because in Jarchi et al., (2014) the sign of AP axis is positive backwards, while in the Mobilise-D convention (Palmerini et al., 2020) it is positive forwards
        data['acc_ml'] = -data['acc_ml'] # needed because in Jarchi et al., (2014) the sign of ML axis is positive rightwards, while in the Mobilise-D convention (Palmerini et al., 2020) it is positive leftwards (when subject is staring forwards)
        # data['acc_is'] = -data['acc_is'] # needed because in Jarchi et al., (2014) the sign of ML axis is positive rightwards, while in the Mobilise-D convention (Palmerini et al., 2020) it is positive leftwards (when subject is staring forwards)

        # --------------------------------------------------
        # 1. SSA Preprocessing
        # --------------------------------------------------
        ap_comps = self._apply_ssa(data['acc_ap'].to_numpy())
        si_comps = self._apply_ssa(data['acc_is'].to_numpy())
        ml_comps = self._apply_ssa(data['acc_ml'].to_numpy())

        # Detrended signals + Mean Correction
        ap_sig = (ap_comps[1] + ap_comps[2]) + np.mean(data['acc_ap'])
        si_sig = (si_comps[1] + si_comps[2]) + np.mean(data['acc_is'])
        ml_sig = (ml_comps[1] + ml_comps[2]) + np.mean(data['acc_ml'])

        # AP Dominant Oscillation for interval search
        ap_dominant = ap_comps[1]

        if plot_debug:
            self._debug_data = {
                'ap_raw': data['acc_ap'].to_numpy(),
                'ap_trend': ap_comps[0],
                'ap_dominant': ap_dominant,
                'ap_detrended': ap_sig,
                'si_detrended': si_sig,
                'ml_detrended': ml_sig
            }

        # --------------------------------------------------
        # 2. IC Detection & Side Determination
        # --------------------------------------------------
        ic_raw_indices, intervals = self._detect_ic_indices(ap_dominant, ap_sig, si_sig)
        if plot_debug: self._debug_data['intervals'] = intervals

        events = self._determine_sides(ic_raw_indices, ml_sig)

        # --------------------------------------------------
        # 3. FC (Toe-Off) Detection
        # --------------------------------------------------
        if self.enable_enhanced_cycles:
            events = self._detect_fc_enhanced(events, ml_sig)
        else:
            events = self._detect_fc_simple(events, ml_sig)

        # --------------------------------------------------
        # 4. Reconstruct Output (Merge & Sort)
        # --------------------------------------------------
        self._store_results(events)

        if plot_debug:
            self._plot_debug_visualization()

        return self

    def _apply_ssa(self, signal: np.ndarray) -> np.ndarray:
        X = signal.reshape(1, -1)
        return self.ssa_grouper.fit_transform(X)[0]

    def _detect_ic_indices(self, ap_dom: np.ndarray, ap_dt: np.ndarray, si_dt: np.ndarray) -> Tuple[np.ndarray, List]:
        valleys, _ = find_peaks(-ap_dom, distance=int(0.2 * self.sampling_rate_hz))
        ics, intervals = [], []
        radius = int(self.radius_s * self.sampling_rate_hz)

        for v in valleys:
            start, end = max(0, v - radius), min(len(ap_dt), v + radius)
            intervals.append((start, end))
            product = ap_dt[start:end] * si_dt[start:end]
            ics.append(start + np.argmin(product))
        return np.array(sorted(list(set(ics)))), intervals

    def _determine_sides(self, ics: np.ndarray, ml_sig: np.ndarray) -> List[dict]:
        if len(ics) < 3:
            return [{'ic': i, 'side': 'Unknown', 'fc': np.nan} for i in ics]

        temp_sides = [None] * len(ics)
        for i in range(len(ics) - 2):
            m1 = np.mean(ml_sig[ics[i]:ics[i + 1]])
            m2 = np.mean(ml_sig[ics[i + 1]:ics[i + 2]])
            # Logic: Mean(RHC->LHC) > Mean(LHC->RHC)
            pattern = ['R', 'L'] if m1 > m2 else ['L', 'R']
            if temp_sides[i] is None: temp_sides[i] = pattern[0]
            if temp_sides[i + 1] is None: temp_sides[i + 1] = pattern[1]

        if temp_sides[-1] is None:
            temp_sides[-1] = 'L' if temp_sides[-2] == 'R' else 'R'

        return [{'ic': ics[j], 'side': temp_sides[j], 'fc': np.nan} for j in range(len(ics))]

    def _detect_fc_simple(self, events: List[dict], ml_sig: np.ndarray) -> List[dict]:
        """
        Detects Toe-Off (FC) events based on Jarchi et al. logic, adapted from Eargait's approach.

        Logic mapping to Eargait:
        - The Left Heel Contact (LHC) acts as the 'Contralateral IC' anchor.
        - LTO (Left FC): Local Maximum BEFORE the LHC (Backward search).
        - RTO (Right FC): Local Minimum AFTER the LHC (Forward search).
        """

        # 1. Pre-calculate all peaks and valleys on the full ML signal (like eargait)
        # using a width/distance heuristic to avoid high-frequency noise.
        # Note: valleys are peaks on the inverted signal.
        peaks, _ = find_peaks(ml_sig, width=int(0.1 * self.sampling_rate_hz))
        valleys, _ = find_peaks(-ml_sig, width=int(0.1 * self.sampling_rate_hz))

        # 2. Iterate through events to assign FCs relative to the anchor (LHC)
        for i in range(len(events)):
            curr = events[i]

            # Jarchi defines the pattern relative to the Left Heel Contact
            if curr['side'] == 'L':
                lhc_idx = curr['ic']

                # --- A. Detect LTO (Left Toe Off) -> Look BACKWARD ---
                # Find local max BEFORE this Left IC.
                # Valid search window: from the Previous IC up to the Current Left IC.
                start_search = events[i - 1]['ic'] if i > 0 else 0

                # Eargait logic: maximas[(i - maximas) > 0] -> Take the last one
                potential_lto = peaks[(peaks > start_search) & (peaks < lhc_idx)]

                if len(potential_lto) > 0:
                    # Take the closest peak before the IC (the last one in the array)
                    curr['fc'] = potential_lto[-1]
                else:
                    curr['fc'] = np.nan

                # --- B. Detect RTO (Right Toe Off) -> Look FORWARD ---
                # Find first local min AFTER this Left IC.
                # Valid search window: from Current Left IC up to the Next IC.
                end_search = events[i + 1]['ic'] if i < len(events) - 1 else len(ml_sig)

                # Eargait logic: minimas[(minimas - i) > 0] -> Take the first one
                potential_rto = valleys[(valleys > lhc_idx) & (valleys < end_search)]

                if len(potential_rto) > 0:
                    # Take the first valley after the IC
                    rto_val = potential_rto[0]

                    # Important: This RTO belongs to the RIGHT foot.
                    # We must assign it to the next event (if it exists and is Right side)
                    if i < len(events) - 1:
                        next_event = events[i + 1]
                        if next_event['side'] == 'R':
                            next_event['fc'] = rto_val

            # If side is Right, we don't process it as an anchor,
            # because its FC (RTO) was already handled by the previous Left event's forward look.

        return events

    def _detect_fc_enhanced(self, events: List[dict], ml_sig: np.ndarray) -> List[dict]:
        """
        SVD-based Toe-Off detection (Sections 4-6 of Jarchi et al.).
        Refined to find LTO before LHC and RTO after LHC.
        """
        if len(events) < 4:
            return self._detect_fc_simple(events, ml_sig)

        # 1. Segment cycles from IC to IC
        cycles = []
        max_len = 0
        for i in range(1, len(events)):
            seg = ml_sig[events[i - 1]['ic']: events[i]['ic']]
            cycles.append(seg)
            max_len = max(max_len, len(seg))

        # 2. Resampling for SVD Matrix
        padded_cycles = [np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(c)), c) for c in cycles]
        A = np.array(padded_cycles)

        # 3. SVD Template Extraction
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        v_dominant = Vt[0, :]

        # 4. Find features on the template
        # Identify the 'Main Valley' of the ML cycle
        main_valley_idx = np.argmin(v_dominant)

        template_peaks = find_peaks(v_dominant)[0]
        template_valleys = argrelextrema(v_dominant, np.less)[0]

        # 5. Inverse Mapping
        for i in range(1, len(events)):
            curr_event = events[i]
            actual_len = curr_event['ic'] - events[i - 1]['ic']

            if curr_event['side'] == 'L':
                # LTO: Local maximum before the main valley (LHC)
                valid_peaks = template_peaks[template_peaks < main_valley_idx]
                potential_idx = valid_peaks[-1] if len(valid_peaks) > 0 else 0
            else:
                # RTO: Local minimum after the main valley (LHC)
                valid_valleys = template_valleys[template_valleys > main_valley_idx]
                potential_idx = valid_valleys[0] if len(valid_valleys) > 0 else main_valley_idx

            # Scaling and assignment
            fc_relativo = int(potential_idx * (actual_len / max_len))
            curr_event['fc'] = events[i - 1]['ic'] + fc_relativo

        return events

    def _store_results(self, events: List[dict]):
        ic, ic_side, fc, fc_side = [], [], [], []

        for e in events:
            ic.append(e['ic'])
            ic_side.append(e['side'])
            if not np.isnan(e['fc']):
                fc.append(e['fc'])
                fc_side.append(e['side'])

        # Store ICs
        self.ic_list_ = pd.DataFrame({"ic": ic}, index=pd.RangeIndex(len(ic), name="step_id"))
        self.ic_side_ = pd.Series(ic_side, index=self.ic_list_.index, name="side")

        # Store FCs
        self.fc_list_ = pd.DataFrame({"fc": fc}, index=pd.RangeIndex(len(fc), name="step_id"))
        self.fc_side_ = pd.Series(fc_side, index=self.fc_list_.index, name="side")

    def _handle_empty_output(self):
        self.ic_list_ = pd.DataFrame(columns=["ic"])
        self.fc_list_ = pd.DataFrame(columns=["fc"])
        self.ic_side_ = pd.Series(dtype=str)
        self.fc_side_ = pd.Series(dtype=str)
        return self

    def _plot_debug_visualization(self):
        """
        Generates a comprehensive debug plot showing the SSA components,
        the product signal for IC detection, and side determination logic.
        """
        if not hasattr(self, '_debug_data') or not self._debug_data:
            print("No debug data available. Run detect() with plot_debug=True.")
            return

        d = self._debug_data
        # Map sides for plotting colors
        side_colors = {"R": "red", "L": "green", "Unknown": "gray"}

        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Jarchi Method Diagnostic - Signal Processing Steps', fontsize=16)

        # --- Subplot 1: SSA & Trend Removal ---
        axes[0].plot(d['ap_raw'], color='silver', alpha=0.6, label='Raw AP (Ear Accel)')
        axes[0].plot(d['ap_trend'], color='darkorange', linewidth=2, label='SSA Trend (Head Motion)')
        axes[0].plot(d['ap_dominant'], color='royalblue', label='Dominant Oscillation (Gait Frequency)')
        axes[0].set_title("Step 1: SSA Decomposition (AP Axis)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # --- Subplot 2: IC Detection (The Product Signal) ---
        # The core logic: ICs are found where AP and SI axes both have specific local characteristics
        product_sig = d['ap_detrended'] * d['si_detrended']
        axes[1].plot(product_sig, color='purple', label='Product Signal (AP_detrend * SI_detrend)')

        # Highlight search intervals
        for i, (start, end) in enumerate(d.get('intervals', [])):
            axes[1].axvspan(start, end, color='blue', alpha=0.1, label='Search Interval' if i == 0 else "")

        # Mark detected ICs
        if not self.ic_list_.empty:
            ics = self.ic_list_['ic'].values.astype(int)
            axes[1].scatter(ics, product_sig[ics], color='black', marker='X', s=80, label='Detected IC', zorder=5)

        axes[1].set_title("Step 2: Initial Contact Search (Local Minima of Product Signal)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # --- Subplot 3: Side Determination (ML Axis) ---
        axes[2].plot(d['ml_detrended'], color='teal', label='ML Signal (Detrended)')

        # Color intervals between ICs to show L/R logic
        ics = self.ic_list_['ic'].values.astype(int)
        sides = self.ic_side_.values
        for i in range(len(ics) - 1):
            start, end = ics[i], ics[i + 1]
            color = side_colors.get(sides[i], "gray")
            axes[2].axvspan(start, end, color=color, alpha=0.15)
            axes[2].text((start + end) / 2, np.max(d['ml_detrended']), sides[i],
                         color=color, fontweight='bold', ha='center')

        axes[2].set_title("Step 3: Side Determination (Mean ML Amplitude Logic)")
        axes[2].grid(True, alpha=0.3)

        # --- Subplot 4: Final Events on Vertical Axis ---
        axes[3].plot(d['si_detrended'], color='black', alpha=0.8, label='SI Axis (Vertical)')

        # Plot ICs
        for i, row in self.ic_list_.iterrows():
            side = self.ic_side_.iloc[i]
            axes[3].scatter(row['ic'], d['si_detrended'][int(row['ic'])],
                            color=side_colors[side], marker='v', s=120, label=f'IC {side}' if i < 2 else "")

        # Plot FCs (Toe-Offs)
        for i, row in self.fc_list_.iterrows():
            side = self.fc_side_.iloc[i]
            axes[3].scatter(row['fc'], d['si_detrended'][int(row['fc'])],
                            color=side_colors[side], marker='o', facecolors='none',
                            linewidth=2, s=100, label=f'FC {side}' if i < 2 else "")

        axes[3].set_title("Step 4: Final Event Timestamps (IC = Triangle, FC = Circle)")
        axes[3].set_xlabel("Samples")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot(self):
        """Standard plot method to visualize the final detected events on raw data."""
        if self.ic_list_.empty:
            print("No events detected to plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 4))
        # Use AP detrended for clean visualization
        data_to_plot = self._debug_data.get('ap_detrended') if hasattr(self, '_debug_data') else None

        if data_to_plot is not None:
            ax.plot(data_to_plot, color='gray', alpha=0.5, label='Signal')

            # Plot ICs
            for side, color in zip(['L', 'R'], ['green', 'red']):
                mask = self.ic_side_ == side
                ax.scatter(self.ic_list_.loc[mask, 'ic'], data_to_plot[self.ic_list_.loc[mask, 'ic'].astype(int)],
                           c=color, label=f'IC {side}', marker='v')

                # Plot FCs
                mask_fc = self.fc_side_ == side
                ax.scatter(self.fc_list_.loc[mask_fc, 'fc'], data_to_plot[self.fc_list_.loc[mask_fc, 'fc'].astype(int)],
                           edgecolors=color, facecolors='none', label=f'FC {side}', marker='o')

            ax.set_title("Detected Gait Events (Jarchi Method)")
            ax.legend()
            plt.show()