import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from .base import BaseIcDetector


class IcdTomc(BaseIcDetector):
    """
    Gait event detection based on Adaptive Frequency Oscillators (AFO),
    following Tomc et al. (2023).

    The algorithm uses a pool of adaptive oscillators driven by the
    mediolateral acceleration of the head (acc_ml). After synchronization,
    peaks and valleys of the estimated signal are used to detect gait events.

    References
    ----------
    Tomc et al., 2023
    """

    def __init__(
        self,
        n_osc: int = 4,
        k0: float = 0.2,
        k_phi: float = 3.0,
        k_alpha: float = 0.2,
        k_omega: float = 3.0,
        omega0: float = 0.9,
    ):
        """
        Parameters
        ----------
        n_oscillators
            Number of adaptive frequency oscillators.
        k0, k_phi, k_alpha, k_omega
            AFO tuning gains (see Tomc et al., 2023).
        omega_init_hz
            Initial fundamental frequency (Hz).
        """

        self.n_osc = n_osc
        self.k0 = k0
        self.k_phi = k_phi
        self.k_alpha = k_alpha
        self.k_omega = k_omega

        self.omega0 = 2 * np.pi * omega0

    def detect(
        self,
        data: pd.DataFrame,
        remove_gravity: bool = False,
        window_seconds: float = 6.0,
        peak_update_seconds: float = 1.5,
        sampling_rate_hz: float = 128.0,
    ):
        """
        Detect initial contact (IC) and final contact (FC) events.

        Parameters
        ----------
        data
            DataFrame containing IMU signals. Must include 'acc_ml'.
        remove_gravity
            If True, gravity is removed before using acc_ml
            (requires sensor orientation estimation).
        window_seconds
            Length of the sliding window used for peak detection.
        peak_update_seconds
            Interval at which peak detection is performed.

        Returns
        -------
        self
        """
        acc_ml = data["acc_ml"].to_numpy()

        if remove_gravity:
            acc_ml = self._remove_gravity(data, sampling_rate_hz=sampling_rate_hz)[:, 1]  # mediolateral axis

        dt = 1.0 / sampling_rate_hz
        n_samples = acc_ml.shape[0]

        # --- Initialize AFO states ---
        alpha0 = 0.0
        alpha = np.array([1.0, 0.2, 0.5, 0.1][: self.n_osc])
        phi = np.zeros(self.n_osc)
        omega = self.omega0

        # --- Storage ---
        u_hat = np.zeros(n_samples)
        phi1 = np.zeros(n_samples)

        # --- AFO integration ---
        for k in range(n_samples):
            u = acc_ml[k]

            u_hat[k] = alpha0 + np.sum(alpha * np.sin(phi))
            e = u - u_hat[k]

            # State updates
            alpha0 += self.k0 * e * dt
            omega += self.k_omega * e * np.cos(phi[0]) * dt

            for i in range(self.n_osc):
                phi[i] += (i + 1) * omega * dt + self.k_phi * e * np.cos(phi[i]) * dt
                alpha[i] += self.k_alpha * e * np.sin(phi[i]) * dt

            phi1[k] = phi[0]

        # --- Peak detection on estimated signal ---
        win = int(window_seconds * sampling_rate_hz)
        step = int(peak_update_seconds * sampling_rate_hz)

        ic_list = []
        # fc_list = []

        for start in range(0, n_samples - win, step):
            segment = u_hat[start : start + win]

            peaks_pos, _ = find_peaks(segment)
            peaks_neg, _ = find_peaks(-segment)

            events = []

            for idx in peaks_pos:
                events.append(
                    {
                        "idx": idx,
                        "type": "peak",
                        "sign": np.sign(segment[idx]),
                    }
                )

            for idx in peaks_neg:
                events.append(
                    {
                        "idx": idx,
                        "type": "valley",
                        "sign": np.sign(segment[idx]),
                    }
                )

            # Sort events by time
            events = sorted(events, key=lambda x: x["idx"])

            ic_candidates = []

            for i in range(1, len(events)):
                prev_evt = events[i - 1]
                curr_evt = events[i]

                # Case 1: negative valley → positive peak
                if (
                        curr_evt["type"] == "peak"
                        and curr_evt["sign"] == 1
                        and prev_evt["type"] == "valley"
                        and prev_evt["sign"] == -1
                ):
                    ic_candidates.append(prev_evt["idx"])

                # Case 2: positive peak → negative valley
                if (
                        prev_evt["type"] == "valley"
                        and prev_evt["sign"] == 1
                        and curr_evt["type"] == "peak"
                        and curr_evt["sign"] == 1
                ):
                    ic_candidates.append(curr_evt["idx"])

            if len(ic_candidates) > 0:
                ic_idx = ic_candidates[-1] + start
                ic_list.append(ic_idx)

        self.ic_list_ = np.array(ic_list, dtype=int)
        fc_list = np.empty(np.shape(ic_list))
        fc_list[:] = np.nan
        # --------------------------------------------------
        # 4. Store results
        # --------------------------------------------------
        self.ic_list_ = pd.DataFrame(
            {"ic": np.array(ic_list, dtype=int)},
            index=pd.RangeIndex(len(ic_list), name="step_id"),
        )

        self.fc_list_ = pd.DataFrame(
            {"fc": np.array(fc_list, dtype=int)},
            index=pd.RangeIndex(len(fc_list), name="step_id"),
        )

        t = np.arange(n_samples) / sampling_rate_hz

        plt.figure(figsize=(12, 4))
        plt.plot(t, acc_ml, label="Driver signal (acc_ml)", alpha=0.6)
        plt.plot(t, u_hat, label="AFO reconstructed signal", linewidth=2)

        for i, ic in enumerate(self.ic_list_["ic"]):
            plt.axvline(
                ic / sampling_rate_hz,
                linestyle="--",
                color="tab:red",
                alpha=0.7,
                label="IC" if i == 0 else None,
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration")
        plt.title("IcdTomc – AFO reconstruction and detected gait events")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return self

    def _remove_gravity(self, data: pd.DataFrame, sampling_rate_hz: float) -> np.ndarray:
        """
        Estimate orientation and remove gravity contribution.

        This is intentionally kept simple and consistent with
        the other ICD implementations.
        """
        acc = data[["acc_x", "acc_y", "acc_z"]].to_numpy()
        gravity = np.mean(acc[: int(2 * sampling_rate_hz)], axis=0)
        gravity = gravity / np.linalg.norm(gravity)
        return acc - gravity
