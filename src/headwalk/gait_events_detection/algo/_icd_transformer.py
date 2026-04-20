from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, find_peaks
from typing_extensions import Self, Unpack
from dataclasses import dataclass
import math

from .base import BaseIcDetector


class IcdTransformer(BaseIcDetector):

    def __init__(
        self,
        artifact_dir = Path(__file__).resolve().parents[1] / "models"/ "portable_artifacts" / "transformer_icd_fold_6",
        device="auto",
        target_fs: float = 100.0,
        lowpass_hz: float = 20.0,
        butter_order: int = 4,
        window_size: int = 200,
        prob_threshold: float = 0.39,
        min_peak_distance: int = 25,
        offset: float = 0.028, # optimized: 0.028, # best working (empirical); 0.03, #original: 0.0,
    ):
        # DO NOT MODIFY PARAMETERS (tpcp requirement)
        self.artifact_dir = artifact_dir
        self.device = device
        self.target_fs = target_fs
        self.lowpass_hz = lowpass_hz
        self.butter_order = butter_order
        self.window_size = window_size
        self.prob_threshold = prob_threshold
        self.min_peak_distance = min_peak_distance
        self.offset = offset

        self._model = None
        self._torch_device = None
        self.metadata_ = {}

    # ------------------------------------------------------------
    # Preparation (all logic moved here)
    # ------------------------------------------------------------
    def _prepare(self):

        self.artifact_dir_ = Path(self.artifact_dir)

        # Defaults (applied here, NOT in __init__)
        self.target_fs_ = 100.0 if self.target_fs is None else float(self.target_fs)
        self.lowpass_hz_ = 20.0 if self.lowpass_hz is None else float(self.lowpass_hz)
        self.butter_order_ = 4 if self.butter_order is None else int(self.butter_order)
        self.window_size_ = 200 if self.window_size is None else int(self.window_size)
        self.prob_threshold_ = 0.24 if self.prob_threshold is None else float(self.prob_threshold)
        self.min_peak_distance_ = 25 if self.min_peak_distance is None else int(self.min_peak_distance)
        self.offset_ = self.offset

        self._load_preprocessing()
        self._load_metadata()
        self._load_model()

    # ------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------
    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _load_json_if_exists(self, path):
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _load_preprocessing(self):
        prep = self._load_json_if_exists(self.artifact_dir_ / "preprocessing.json")
        if not prep:
            return

        self.target_fs_ = prep.get("target_fs", self.target_fs_)
        self.lowpass_hz_ = prep.get("lowpass_hz", self.lowpass_hz_)
        self.butter_order_ = prep.get("butter_order", self.butter_order_)
        self.window_size_ = prep.get("window_size", self.window_size_)
        self.prob_threshold_ = prep.get("prob_threshold", self.prob_threshold_)
        self.min_peak_distance_ = prep.get("min_peak_distance", self.min_peak_distance_)

    def _load_metadata(self):
        self.metadata_ = self._load_json_if_exists(self.artifact_dir_ / "metadata.json")

    def _load_model(self):

        if self._model is not None:
            return

        cfg_dict = json.loads((self.artifact_dir_ / "config.json").read_text())
        cfg = TransformerConfig(**cfg_dict)

        dev = self._resolve_device()
        self._torch_device = dev

        model = build_transformer_encoder(cfg)
        state = torch.load(self.artifact_dir_ / "model.pt", map_location=dev)
        model.load_state_dict(state, strict=True)
        model.to(dev).eval()

        self._model = model

    # ------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------
    def _apply_lowpass(self, X, fs):
        nyq = 0.5 * fs
        b, a = butter(self.butter_order_, self.lowpass_hz_ / nyq, btype="low")
        return filtfilt(b, a, X, axis=0)

    def _detect_peaks(self, prob: np.ndarray) -> np.ndarray:
        peaks, _ = find_peaks(prob, height=self.prob_threshold, distance=self.min_peak_distance)
        return peaks

    def _conditional_zscore(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        Xn = (X - mean) / std
        if np.all(np.max(np.abs(Xn), axis=0) < np.max(np.abs(X), axis=0)):
            return Xn
        return X

    def _window_signal(self, X):
        windows = []
        for i in range(0, len(X) - self.window_size_ + 1, self.window_size_):
            win = self._conditional_zscore(X[i:i + self.window_size_])
            windows.append(win)
        return np.asarray(windows, dtype=np.float32)

    def _resample_if_needed(self, X: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
        if np.isclose(fs, self.target_fs):
            return X, fs
        n_new = int(len(X) * self.target_fs / fs)
        X_rs = resample(X, n_new, axis=0)
        return X_rs, self.target_fs

    def _reconstruct_output(self, Y, total_len):
        Yc = np.zeros((total_len, 4), dtype=np.float32)
        idx = 0
        for i in range(Y.shape[0]):
            Yc[idx:idx + self.window_size_] = Y[i]
            idx += self.window_size_
        return Yc

    def _plot_debug(
            self,
            raw_acc_is: np.ndarray,
            filtered_acc_is: np.ndarray,
            resampled_acc_is: np.ndarray,
            Yc: np.ndarray,
            ic_indices: np.ndarray,
            ic_sides: np.ndarray,
    ):
        """
        Debug plot identical structure to IcdTcn.
        """

        t = np.arange(len(raw_acc_is)) / self.sampling_rate_hz
        t_rs = np.arange(len(resampled_acc_is)) / self.target_fs_

        fig = plt.figure(figsize=(14, 12))
        ax1 = plt.subplot(4, 1, 1)
        ax2 = plt.subplot(4, 1, 2, sharex=ax1)
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)

        fig.suptitle(
            f'Debug Pipeline: Transformer - {self.artifact_dir_.name}',
            fontsize=16
        )

        # --- 1) Filtering ---
        ax1.plot(t, raw_acc_is, color='gray', alpha=0.4, label='Raw')
        ax1.plot(t, filtered_acc_is, color='blue',
                 label=f'Filtered ({self.lowpass_hz_}Hz LP)')
        ax1.set_title('Step 1: Filtering (Original Sampling Rate)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # --- 2) Resampling ---
        ax2.plot(t_rs, resampled_acc_is, color='purple',
                 label=f'Resampled to {self.target_fs_}Hz')
        ax2.set_title('Step 2: Resampling')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # --- 3) Likelihood ---
        ax3.plot(t_rs, Yc[:, 0], label='L-IC Prob', color='green', alpha=0.7)
        ax3.plot(t_rs, Yc[:, 2], label='R-IC Prob', color='red', alpha=0.7)
        ax3.axhline(y=self.prob_threshold_, color='black',
                    linestyle=':', label='Threshold')
        ax3.set_title('Step 3: Model Likelihood (Target Domain)')
        ax3.set_ylabel('Probability')
        ax3.set_ylim([-0.05, 1.05])
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # --- 4) Final Events ---
        ax4.plot(t_rs, resampled_acc_is, color='black',
                 alpha=0.6, label='Processed Signal')

        l_mask = (ic_sides == 'L')
        r_mask = (ic_sides == 'R')

        if np.any(l_mask):
            idx_l = ic_indices[l_mask]
            ax4.plot(t_rs[idx_l], resampled_acc_is[idx_l],
                     'go', label='Left IC', markersize=8)

        if np.any(r_mask):
            idx_r = ic_indices[r_mask]
            ax4.plot(t_rs[idx_r], resampled_acc_is[idx_r],
                     'ro', label='Right IC', markersize=8)

        ax4.set_title('Step 4: Final Events (Target Domain)')
        ax4.set_xlabel('Samples (at Target Frequency)')
        ax4.set_ylabel('Acc (m/s^2)')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def detect(
            self,
            data: pd.DataFrame,
            *,
            sampling_rate_hz: float,
            acc_columns: list[str] = ["acc_is", "acc_ml", "acc_ap"],
            gyr_columns: list[str] = ["gyr_is", "gyr_ml", "gyr_ap"],
            plot_debug: bool = True,
            **_: Unpack[dict[str, Any]],
    ) -> Self:

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Prepare model + params
        self._prepare()

        # --------------------------------------------------------------
        # 1) RAW INPUT
        # --------------------------------------------------------------
        raw_acc_is = data[acc_columns[0]].to_numpy()

        acc = data[acc_columns].to_numpy()
        gyr = data[gyr_columns].to_numpy()
        X = np.concatenate([acc, gyr], axis=1)

        # --------------------------------------------------------------
        # 2) LOWPASS FILTERING (Original FS domain)
        # --------------------------------------------------------------
        X_filtered = self._apply_lowpass(X, sampling_rate_hz)
        filtered_acc_is = X_filtered[:, 0].copy()

        # --------------------------------------------------------------
        # 3) RESAMPLING (Target FS domain)
        # --------------------------------------------------------------
        X_rs, fs_rs = self._resample_if_needed(X_filtered, sampling_rate_hz)
        resampled_acc_is = X_rs[:, 0].copy()

        # --------------------------------------------------------------
        # 4) WINDOWING + CONDITIONAL Z-SCORE
        # --------------------------------------------------------------
        X_win = self._window_signal(X_rs)

        if len(X_win) == 0:
            self.ic_list_ = pd.DataFrame(columns=["ic"])
            self.fc_list_ = pd.DataFrame(columns=["fc"])
            self.ic_side_ = pd.Series(dtype=str)
            self.fc_side_ = pd.Series(dtype=str)
            return self

        # --------------------------------------------------------------
        # 5) MODEL PREDICTION (Likelihood)
        # --------------------------------------------------------------
        with torch.no_grad():
            x = torch.from_numpy(X_win).to(self._torch_device)
            Y_pred = self._model(x).cpu().numpy()

        total_len = len(X_win) * self.window_size_
        Yc = self._reconstruct_output(Y_pred, total_len)

        # Align resampled signal to prediction length
        if len(resampled_acc_is) > total_len:
            resampled_acc_is = resampled_acc_is[:total_len]

        # --------------------------------------------------------------
        # 6) PEAK DETECTION (Target FS domain)
        # --------------------------------------------------------------
        L_ic = self._detect_peaks(Yc[:, 0])
        R_ic = self._detect_peaks(Yc[:, 2])
        L_fc = self._detect_peaks(Yc[:, 1])
        R_fc = self._detect_peaks(Yc[:, 3])

        ic, ic_side = [], []
        fc, fc_side = [], []

        for i in L_ic:
            ic.append(i)
            ic_side.append("L")
        for i in R_ic:
            ic.append(i)
            ic_side.append("R")

        for i in L_fc:
            fc.append(i)
            fc_side.append("L")
        for i in R_fc:
            fc.append(i)
            fc_side.append("R")

        ic = np.asarray(ic) + int(self.offset_*self.target_fs_)
        ic_side = np.asarray(ic_side)
        fc = np.asarray(fc) + int(self.offset_*self.target_fs_)
        fc_side = np.asarray(fc_side)

        # --------------------------------------------------------------
        # 7) DEBUG PLOT (Target FS domain)
        # --------------------------------------------------------------
        if plot_debug:
            self._plot_debug(
                raw_acc_is,
                filtered_acc_is,
                resampled_acc_is,
                Yc,
                ic,
                ic_side,
            )

        # --------------------------------------------------------------
        # 8) RESAMPLE EVENTS BACK TO ORIGINAL FS
        # --------------------------------------------------------------
        resample_ratio = sampling_rate_hz / self.target_fs_
        ic = (ic * resample_ratio + int(self.offset_*self.target_fs_)).astype(int)
        fc = (fc * resample_ratio + int(self.offset_*self.target_fs_)).astype(int)

        # --------------------------------------------------------------
        # 9) SORT + STORE RESULTS
        # --------------------------------------------------------------
        order_ic = np.argsort(ic)
        ic, ic_side = ic[order_ic], ic_side[order_ic]

        order_fc = np.argsort(fc)
        fc, fc_side = fc[order_fc], fc_side[order_fc]

        self.ic_list_ = pd.DataFrame(
            {"ic": ic},
            index=pd.RangeIndex(len(ic), name="step_id"),
        )
        self.fc_list_ = pd.DataFrame(
            {"fc": fc},
            index=pd.RangeIndex(len(fc), name="step_id"),
        )
        self.ic_side_ = pd.Series(ic_side, index=self.ic_list_.index, name="side")
        self.fc_side_ = pd.Series(fc_side, index=self.fc_list_.index, name="side")

        return self


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (no learnable parameters).
    """
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TransformerEncoderICD(nn.Module):
    """
    Encoder-only Transformer for many-to-many event heatmap regression.

    Input:
      x: (B, T, C_in)  e.g., (B, 200, 6)

    Output:
      y: (B, T, C_out) e.g., (B, 200, 4) in [0,1] if use_sigmoid=True
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        use_sigmoid: bool = True,
        pos_encoding: str = "sin",  # "sin" or "none"
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.input_proj = nn.Linear(in_channels, d_model)

        if pos_encoding == "sin":
            self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len)
        elif pos_encoding == "none":
            self.pos_enc = nn.Identity()
        else:
            raise ValueError("pos_encoding must be 'sin' or 'none'")

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)     # (B,T,D)
        h = self.pos_enc(h)        # (B,T,D)
        h = self.encoder(h)        # (B,T,D)
        y = self.head(h)           # (B,T,out)

        if self.use_sigmoid:
            y = torch.sigmoid(y)   # (B,T,out) in [0,1]
        return y


@dataclass(frozen=True)
class TransformerConfig:
    """
    Simple config container used by the factory.
    """
    in_channels: int = 6
    out_channels: int = 4
    seq_len: int = 200
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    use_sigmoid: bool = True
    pos_encoding: str = "sin"
    norm_first: bool = True


def build_transformer_encoder(cfg: TransformerConfig) -> nn.Module:
    """
    Factory function for the encoder-only transformer architecture.
    """
    return TransformerEncoderICD(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
        use_sigmoid=cfg.use_sigmoid,
        pos_encoding=cfg.pos_encoding,
        norm_first=cfg.norm_first,
    )
