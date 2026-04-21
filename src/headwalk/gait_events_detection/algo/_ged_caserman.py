import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from typing import Optional, Tuple
from typing_extensions import Self

from .base import BaseGeDetector


class GedCaserman(BaseGeDetector):
    """
    Caserman et al. (2016) step / gait event detector using head acceleration signals.

    Convention assumed (Mobilise-D style for head/occipital sensor):
      - y-axis: acc_is (Inferior-Superior, positive upward)
      - x-axis: acc_ml (Medio-Lateral, positive to the right)

    Summary of method (as described):
      1) Detect heel strikes (IC) as positive peaks in y-axis above an adaptive threshold.
      2) Detect toe-off (FC) as the *following local peak* in y-axis after the heel strike.
      3) Adaptive step threshold is based on a moving average of the last N "event averages"
         for IC and FC (Caserman uses N=32).
      4) Laterality: classify each IC based on the x-axis value compared against the
         interquartile mean (IQM) computed over all detected steps in the trial.
         - typically: positive x deflection => right step, negative => left step.
         - here: we implement "adaptive side decision" using IQM thresholding.

    Notes / Practical choices:
      - The paper description is a bit high-level; some implementation details are not fully specified
        (e.g., exact definition of "average measurement values for toe off and heel strike").
        This implementation follows the description as closely as possible with reasonable defaults:
          * We use peak amplitudes in y for IC and FC.
          * The step threshold at time t is the mean of the last N values of:
              avg_event_amp = 0.5*(IC_amp + FC_amp)
            (computed per detected step).
          * Initial threshold is based on a warm-up window (or a percentile-based seed).
      - FC detection: "following local peak" after IC.
        We implement FC as the next positive peak in y after IC within a max search window.

    Outputs (same contract as IcdTcn/IcdFang):
      self.ic_list_ : pd.DataFrame with column 'ic' (sample indices)
      self.fc_list_ : pd.DataFrame with column 'fc' (sample indices)
      self.ic_side_ : pd.Series with values 'L','R','U'
      self.fc_side_ : pd.Series with values 'L','R','U' (inherits side from IC)

    Reference:
      Caserman et al., 2016 (Step detection using head acceleration; adaptive threshold + IQM laterality).
    """

    def __init__(
        self,
        *,
        # Adaptive threshold settings
        history_len: int = 32,                 # Caserman: 32 most recent averages
        init_threshold_quantile: float = 0.85, # seed threshold before history fills
        min_step_duration_s: float = 0.35,     # minimum time between IC peaks
        fc_search_max_s: float = 0.6,          # max time after IC to search FC
        # Peak detection helpers
        prominence: Optional[float] = None,    # optional peak prominence in y
        # Laterality
        iqm_margin: float = 0.0,               # optional dead-zone around IQM
    ) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.init_threshold_quantile = float(init_threshold_quantile)
        self.min_step_duration_s = float(min_step_duration_s)
        self.fc_search_max_s = float(fc_search_max_s)
        self.prominence = prominence
        self.iqm_margin = float(iqm_margin)

        # Public outputs
        self.ic_list_: Optional[pd.DataFrame] = None
        self.ic_side_: Optional[pd.Series] = None
        self.fc_list_: Optional[pd.DataFrame] = None
        self.fc_side_: Optional[pd.Series] = None

        # Debug buffers (optional)
        self._debug_: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        acc_is_col: str = "acc_is",  # y-axis
        acc_ml_col: str = "acc_ml",  # x-axis
        plot_debug: bool = True,
        **kwargs,
    ) -> Self:
        fs = float(sampling_rate_hz)

        y = data[acc_is_col].to_numpy(dtype=float)  # IS (y-axis)
        x = data[acc_ml_col].to_numpy(dtype=float)  # ML (x-axis)

        min_dist = max(1, int(self.min_step_duration_s * fs))
        fc_max = max(1, int(self.fc_search_max_s * fs))

        # Candidate positive peaks in y (heel strikes are a subset, chosen by adaptive threshold)
        peak_kwargs = {"distance": min_dist}
        if self.prominence is not None:
            peak_kwargs["prominence"] = self.prominence
        cand_peaks, props = find_peaks(y, **peak_kwargs)

        # Seed threshold (until we have enough history)
        if len(y) > 0:
            seed_thr = float(np.quantile(y, self.init_threshold_quantile))
        else:
            seed_thr = 0.0

        # Adaptive threshold history: store per-step "average measurement" as described
        # (we interpret it as average amplitude of IC and FC peaks).
        avg_hist: list[float] = []

        ic_idx: list[int] = []
        fc_idx: list[int] = []
        ic_amp: list[float] = []
        fc_amp: list[float] = []
        x_at_ic: list[float] = []

        current_thr = seed_thr

        # Iterate candidate peaks chronologically
        for p in cand_peaks:
            # adaptive step threshold gate
            if y[p] <= current_thr:
                continue

            # heel strike (IC) at p
            # find following local peak in y as toe off (FC)
            # Caserman: "toe off event is defined as the following local peak."
            # We choose the next positive peak in y within [p+1, p+fc_max]
            # using local maxima search by scanning peaks again.
            w_end = min(len(y) - 1, p + fc_max)
            # Find next peak after p among cand_peaks
            next_candidates = cand_peaks[(cand_peaks > p) & (cand_peaks <= w_end)]
            if len(next_candidates) == 0:
                # no FC found in window -> skip this step (or keep IC only)
                # here we skip to stay consistent (IC/FC paired steps)
                continue
            p_fc = int(next_candidates[0])

            ic_idx.append(int(p))
            fc_idx.append(p_fc)
            ic_amp.append(float(y[p]))
            fc_amp.append(float(y[p_fc]))
            x_at_ic.append(float(x[p]))

            # update threshold history
            avg_val = 0.5 * (y[p] + y[p_fc])
            avg_hist.append(float(avg_val))
            if len(avg_hist) > self.history_len:
                avg_hist = avg_hist[-self.history_len:]

            # moving average threshold of last N averages
            current_thr = float(np.mean(avg_hist)) if len(avg_hist) > 0 else seed_thr

        ic_idx = np.asarray(ic_idx, dtype=int)
        fc_idx = np.asarray(fc_idx, dtype=int)

        # Laterality with IQM on x-values at IC
        ic_sides = self._assign_sides_iqm(np.asarray(x_at_ic, dtype=float))

        # FC side is opposite of the IC that precedes it (as in your convention request elsewhere),
        # but Caserman describes step as left/right based on x at IC; FC belongs to same step.
        # For Caserman's walking step, FC should correspond to the same foot as IC -> same side.
        # We'll keep FC = IC side (most consistent with "step laterality").
        fc_sides = ic_sides.copy()

        # Standardized output formatting (like IcdTcn/IcdFang)
        self.ic_list_ = pd.DataFrame({"ic": ic_idx}, index=pd.RangeIndex(len(ic_idx), name="step_id"))
        self.fc_list_ = pd.DataFrame({"fc": fc_idx}, index=pd.RangeIndex(len(fc_idx), name="step_id"))
        self.ic_side_ = pd.Series(ic_sides, index=self.ic_list_.index, name="side")
        self.fc_side_ = pd.Series(fc_sides, index=self.fc_list_.index, name="side")

        # store debug info
        self._debug_ = dict(
            fs=fs,
            y=y,
            x=x,
            cand_peaks=cand_peaks,
            seed_thr=seed_thr,
            thr_trace=None,  # could store if you want per-sample; we store per-step below
            ic_idx=ic_idx,
            fc_idx=fc_idx,
            ic_amp=np.asarray(ic_amp, dtype=float),
            fc_amp=np.asarray(fc_amp, dtype=float),
            x_at_ic=np.asarray(x_at_ic, dtype=float),
            iqm=self._iqm(np.asarray(x_at_ic, dtype=float)) if len(x_at_ic) else np.nan,
        )

        if plot_debug:
            self._plot_debug()

        return self

    # ------------------------------------------------------------------
    # Laterality helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iqm(values: np.ndarray) -> float:
        """
        Interquartile mean:
          - discard bottom 25% and top 25%
          - average the remaining
        """
        if values.size == 0:
            return float("nan")
        v = np.sort(values)
        n = len(v)
        lo = int(np.floor(0.25 * n))
        hi = int(np.ceil(0.75 * n))
        core = v[lo:hi] if hi > lo else v
        return float(np.mean(core)) if core.size else float(np.mean(v))

    def _assign_sides_iqm(self, x_at_ic: np.ndarray) -> np.ndarray:
        """
        Adaptive side decision:
          compare each x_at_ic against IQM of x_at_ic for the trial.

        Convention:
          - positive deflection tends to be Right, negative tends to be Left (Caserman statement).
          - We implement:
              if x_at_ic > IQM + margin => 'R'
              elif x_at_ic < IQM - margin => 'L'
              else => 'U' (ambiguous)
        """
        if x_at_ic.size == 0:
            return np.asarray([], dtype=str)

        iqm = self._iqm(x_at_ic)
        sides = []
        for v in x_at_ic:
            if v > iqm + self.iqm_margin:
                sides.append("R")
            elif v < iqm - self.iqm_margin:
                sides.append("L")
            else:
                sides.append("U")
        return np.asarray(sides, dtype=str)

    # ------------------------------------------------------------------
    # Debug plot
    # ------------------------------------------------------------------
    def _plot_debug(self) -> None:
        """
        Debug visualization of intermediate steps:
          1) y (acc_is) with candidate peaks + accepted IC/FC + adaptive threshold (step-level)
          2) x (acc_ml) with IC markers colored by side + IQM line
        """
        if not self._debug_:
            return

        y = self._debug_["y"]
        x = self._debug_["x"]
        fs = self._debug_["fs"]
        cand = self._debug_["cand_peaks"]
        ic = self._debug_["ic_idx"]
        fc = self._debug_["fc_idx"]
        x_at_ic = self._debug_["x_at_ic"]
        iqm = self._debug_["iqm"]
        seed_thr = self._debug_["seed_thr"]

        t = np.arange(len(y)) / fs

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # --- Top: y-axis with peaks ---
        ax = axes[0]
        ax.plot(t, y, label="acc_is (y-axis)", linewidth=1.2)
        if len(cand) > 0:
            ax.plot(t[cand], y[cand], "o", markersize=3, alpha=0.35, label="candidate y-peaks")

        if len(ic) > 0:
            ax.plot(t[ic], y[ic], "g*", markersize=10, label="IC (heel strike)")
        if len(fc) > 0:
            ax.plot(t[fc], y[fc], "r^", markersize=7, label="FC (toe off)")

        ax.axhline(seed_thr, color="k", linestyle="--", alpha=0.4, label="seed threshold")

        ax.set_title("Caserman 2016 — Step detection on y-axis (acc_is)")
        ax.set_ylabel("acc_is")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

        # --- Bottom: x-axis + laterality ---
        ax = axes[1]
        ax.plot(t, x, label="acc_ml (x-axis)", linewidth=1.2)
        ax.axhline(iqm, color="k", linestyle="--", alpha=0.6, label="IQM (x at IC)")

        if self.ic_list_ is not None and len(self.ic_list_) > 0:
            ic_sides = self.ic_side_.to_numpy(dtype=str)

            l_mask = ic_sides == "L"
            r_mask = ic_sides == "R"
            u_mask = ic_sides == "U"

            if np.any(l_mask):
                idx = ic[l_mask]
                ax.plot(t[idx], x[idx], "g*", markersize=10, label="IC Left")
            if np.any(r_mask):
                idx = ic[r_mask]
                ax.plot(t[idx], x[idx], "r*", markersize=10, label="IC Right")
            if np.any(u_mask):
                idx = ic[u_mask]
                ax.plot(t[idx], x[idx], "y*", markersize=10, label="IC Unknown")

        ax.set_title("Laterality decision on x-axis (acc_ml) using IQM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("acc_ml")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()