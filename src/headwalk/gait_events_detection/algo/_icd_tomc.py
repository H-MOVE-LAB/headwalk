import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Any, Optional
from typing_extensions import Self, Unpack

from .base import BaseIcDetector


class IcdTomc(BaseIcDetector):
    """
    Gait event detection based on Adaptive Frequency Oscillators (AFO).
    Simplified offline version using global peak detection on reconstructed signals.
    """

    def __init__(
            self,
            n_osc: int = 4,
            k0: float = 0.2,
            k_phi: float = 3.0,
            omega0: float = 0.9,
    ):
        super().__init__()
        self.n_osc = n_osc
        self.k0 = k0
        self.k_phi = k_phi
        self.k_alpha = k0 # like k0
        self.k_omega = k_phi # like k_phi
        self.omega0 = omega0

        self.ic_list_: Optional[pd.DataFrame] = None
        self.ic_side_: Optional[pd.Series] = None
        self.fc_list_: Optional[pd.DataFrame] = None
        self.fc_side_: Optional[pd.Series] = None

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_column: str = "acc_ml",
            plot_debug: bool = True,
            **_: Unpack[dict[str, Any]],
    ) -> Self:

        fs = sampling_rate_hz
        dt = 1.0 / fs
        # Note: Input signal inversion as per your current implementation logic
        acc_ml = -data[acc_column].to_numpy()
        n_samples = len(acc_ml)

        # --- 1. AFO State Initialization ---
        alpha0 = 0.0
        alpha = np.array([1.0, 0.2, 0.5, 0.1][: self.n_osc])
        phi = np.zeros(self.n_osc)
        omega_rad = 2 * np.pi * self.omega0
        u_hat = np.zeros(n_samples)

        # --- 2. AFO Online Integration ---
        for k in range(n_samples):
            u = acc_ml[k]
            u_hat[k] = alpha0 + np.sum(alpha * np.sin(phi))
            e = u - u_hat[k]

            alpha0 += self.k0 * e * dt
            omega_rad += self.k_omega * e * np.cos(phi[0]) * dt

            for i in range(self.n_osc):
                phi[i] += (i + 1) * omega_rad * dt + self.k_phi * e * np.cos(phi[i]) * dt
                alpha[i] += self.k_alpha * e * np.sin(phi[i]) * dt

        # --- 3. Simplified Global Peak/Valley Detection ---
        # Find all positive peaks and negative valleys in the reconstructed signal
        peaks_pos, _ = find_peaks(u_hat)
        peaks_neg, _ = find_peaks(-u_hat)

        # Combine into a single sorted event list
        events = []
        for idx in peaks_pos:
            events.append({"idx": idx, "type": "peak", "val": u_hat[idx]})
        for idx in peaks_neg:
            events.append({"idx": idx, "type": "valley", "val": u_hat[idx]})

        events = sorted(events, key=lambda x: x["idx"])

        ic_list = []
        for i in range(1, len(events)):
            prev = events[i - 1]
            curr = events[i]

            # Tomc Heuristic: IC occurs at the valley immediately preceding a positive peak
            # Case 1: Standard (Negative valley -> Positive peak)
            if curr["type"] == "peak" and curr["val"] > 0 and \
                    prev["type"] == "valley" and prev["val"] < 0:
                ic_list.append(prev["idx"])

            # Case 2: Double peak/Shallow valley (Positive valley -> Positive peak)
            elif curr["type"] == "peak" and curr["val"] > 0 and \
                    prev["type"] == "valley" and prev["val"] >= 0:
                ic_list.append(curr["idx"])

        # --- 4. Store Standardized Results ---
        ic_idx = np.array(ic_list, dtype=int)

        self.ic_list_ = pd.DataFrame(
            {"ic": ic_idx},
            index=pd.RangeIndex(len(ic_idx), name="step_id"),
        )
        self.ic_side_ = pd.Series(['U'] * len(ic_idx), index=self.ic_list_.index, name="side")

        # Empty FC list for consistency
        self.fc_list_ = pd.DataFrame(
            {"fc": np.array([], dtype=int)},
            index=pd.Index([], name="step_id"),
        )
        self.fc_side_ = pd.Series([], index=self.fc_list_.index, name="side", dtype=object)

        if plot_debug:
            self._plot_debug(acc_ml, u_hat, ic_idx, fs)

        return self

    def _plot_debug(self, acc_ml, u_hat, ic_idx, fs):
        t = np.arange(len(acc_ml)) / fs
        plt.figure(figsize=(12, 5))
        plt.plot(t, acc_ml, label="Inverted Driver (-acc_ml)", color="gray", alpha=0.4)
        plt.plot(t, u_hat, label="AFO Reconstruction", color="tab:blue", linewidth=2)

        if len(ic_idx) > 0:
            plt.vlines(ic_idx / fs, plt.ylim()[0], plt.ylim()[1],
                       colors="tab:red", linestyles="--", alpha=0.8, label="IC (Valley/Peak)")

        plt.title("IcdTomc: Simplified Offline AFO Detection")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()