import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Any
from typing_extensions import Self
from .base import BaseIcDetector


class IcdJiang(BaseIcDetector):
    """
    Jiang Initial Contact detector based on Signal Vector Magnitude (SVM).

    Description:
    ------------
    This algorithm detects Initial Contacts (ICs) by analyzing the Signal Vector
    Magnitude (SVM) of 3-axis acceleration. It utilizes a recursive low-pass
    filter (Exponential Moving Average) and an adaptive threshold based on the
    signal's mean and standard deviation to identify gait peaks.

    Algorithm Pipeline:
    1. Calculation of the Signal Vector Magnitude (SVM).
    2. Recursive low-pass filtering (EMA) to attenuate high-frequency noise.
    3. Adaptive thresholding: Threshold = Mean + Std + C.
    4. Peak detection (local maxima) that exceeds the adaptive threshold.
    5. Temporal refinement based on minimum step duration.

    Author: Paolo Tasca
    Date: 19 February 2026

    References:
    -----------
    [1] Jiang, W., et al. (2024). Short Step Length Estimation for Parkinson's
        Disease Patients by Using Fusion Data From Camera-IMU in Smart Glasses.
        IEEE Transactions on Biomedical Engineering, 71(7), 2265-2275.
    """

    def __init__(
            self,
            fc_hz: float = 3.2, #optimized: 3.2, #original: 4.0,
            c_constant: float = 0.0, #optimized: 0.0, #original: 0.0,
            min_step_duration_s: float = 0.4
    ) -> None:
        """
        Initialize the detector with optimizable hyperparameters.

        Args:
            fc_hz: Cut-off frequency for the recursive filter.
            c_constant: Adaptive threshold offset constant 'C'.
            min_step_duration_s: Minimum time between consecutive ICs.
        """
        super().__init__()
        self.fc_hz = fc_hz
        self.c_constant = c_constant
        self.min_step_duration_s = min_step_duration_s

        # Public output attributes
        self.ic_list_: Optional[pd.DataFrame] = None
        self.ic_side_: Optional[pd.Series] = None
        self.fc_list_: Optional[pd.DataFrame] = None
        self.fc_side_: Optional[pd.Series] = None

    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_columns: list[str] = ["acc_is", "acc_ml", "acc_ap"],
            plot_debug: bool = True,
            **kwargs
    ) -> Self:
        """
        Detect Initial Contacts from multi-axis acceleration data.

        Args:
            data: DataFrame containing the gait signals.
            sampling_rate_hz: Sampling frequency of the data.
            acc_columns: List of column names representing the 3-axis acceleration.
            plot_debug: If True, generates a plot of the SVM and detections.

        Returns:
            self: The instance with populated ic_list_ and ic_side_.
        """
        fs = sampling_rate_hz
        ts = 1.0 / fs

        # 1. Calculate Signal Vector Magnitude (SVM)
        acc_coeffs = data[acc_columns].to_numpy()
        svm = np.sqrt(np.sum(acc_coeffs ** 2, axis=1))

        # 2. Recursive Low-Pass Filtering (EMA)
        # alpha = (2*pi * fc * ts) / (2*pi * fc * ts + 1)
        w_c = 2 * np.pi * self.fc_hz
        alpha = (w_c * ts) / (w_c * ts + 1)

        filtered_svm = np.zeros_like(svm)
        filtered_svm[0] = svm[0]
        for i in range(1, len(svm)):
            filtered_svm[i] = filtered_svm[i - 1] + alpha * (svm[i] - filtered_svm[i - 1])

        # 3. Adaptive Thresholding
        mean_val = np.mean(filtered_svm)
        std_val = np.std(filtered_svm)
        threshold = mean_val + std_val + self.c_constant

        # 4. Peak Detection (Local Maxima + Threshold)
        ic_idx = []
        for i in range(1, len(filtered_svm) - 1):
            if filtered_svm[i - 1] < filtered_svm[i] > filtered_svm[i + 1]:
                if filtered_svm[i] > threshold:
                    ic_idx.append(i)

        # 5. Temporal Refinement
        if len(ic_idx) > 0:
            min_dist_samples = int(self.min_step_duration_s * fs)
            refined_ic = [ic_idx[0]]
            for j in range(1, len(ic_idx)):
                if ic_idx[j] - refined_ic[-1] >= min_dist_samples:
                    refined_ic.append(ic_idx[j])
            ic_idx = np.array(refined_ic)
        else:
            ic_idx = np.array([], dtype=int)

        # --- Standardized Output Formatting ---

        # IC Results
        self.ic_list_ = pd.DataFrame({"ic": ic_idx}, index=pd.RangeIndex(len(ic_idx), name="step_id"))
        self.ic_side_ = pd.Series(['U'] * len(ic_idx), index=self.ic_list_.index, name="side")

        # FC Results (Not natively supported by Jiang SVM)
        self.fc_list_ = pd.DataFrame(columns=["fc"])
        self.fc_side_ = pd.Series(dtype=str, name="side")

        if plot_debug:
            self._plot_debug(svm, filtered_svm, threshold, ic_idx)

        return self

    def _plot_debug(self, svm, filtered_svm, threshold, ic_idx):
        """Internal helper for debugging Jiang's SVM processing."""
        plt.figure(figsize=(12, 5))
        plt.plot(svm, label='Raw SVM', color='lightgray', alpha=0.6)
        plt.plot(filtered_svm, label='Filtered SVM (EMA)', color='darkblue')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold (C={self.c_constant})')

        if len(ic_idx) > 0:
            plt.scatter(ic_idx, filtered_svm[ic_idx], color='green', marker='x', label='Detected ICs')

        plt.title('Jiang Algorithm: SVM & Adaptive Threshold')
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (m/s²)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show()