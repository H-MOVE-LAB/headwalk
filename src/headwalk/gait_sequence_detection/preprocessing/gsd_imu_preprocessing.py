"""
Unified IMU preprocessing utilities for head-worn GSD datasets.

This module implements the WearGaitPD-like preprocessing chain to be reused
inside all dataset construction scripts:

raw Acc/Gyr
-> optional fixed sensor-to-Mobilise-D-like axis transform
-> optional/automatic accelerometer unit conversion
-> NaN-aware gravity alignment to axis 0
-> IMU-only quality mask based on long NaN gaps
-> numerical gap filling for filtering/M3 compatibility
-> Brodie/Buckley M3 continuous pitch-roll correction
-> gravity removal
-> zero-phase 15 Hz low-pass filtering
-> X = [acc_final, gyr_final]

Important design rule
---------------------
Numerical gap filling is used only to make the full signal compatible with M3
and filtfilt. It does not make the interpolated samples automatically reliable.
Long original NaN gaps are preserved through imu_quality and must be used later
during window construction to reject windows.

The module does not implement WearGaitPD walkway quality. It only implements the
shared IMU preprocessing that can be applied consistently across datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import butter, filtfilt, welch


GRAVITY_MS2 = 9.80665
DEFAULT_LOW_PASS_CUTOFF_HZ = 15.0
DEFAULT_IMU_GAP_THRESHOLD_FRAMES = 7


@dataclass
class GSDPreprocessingResult:
    """Container returned by preprocess_head_imu_trial()."""

    X: np.ndarray
    acc_final: np.ndarray
    gyr_final: np.ndarray
    acc_aligned_raw: np.ndarray
    gyr_aligned_raw: np.ndarray
    acc_aligned_filled: np.ndarray
    gyr_aligned_filled: np.ndarray
    rotation_matrix: np.ndarray
    mean_static_acc_raw: np.ndarray
    mean_static_acc_aligned: np.ndarray
    gravity_vector: np.ndarray
    imu_quality: np.ndarray
    quality_debug_info: Dict[str, Any]
    preprocessing_debug_info: Dict[str, Any]


def _as_float_2d(name: str, values: np.ndarray, expected_n_cols: int = 3) -> np.ndarray:
    """Convert input to a finite-compatible 2D float array with 3 columns."""

    array = np.asarray(values, dtype=float)

    if array.ndim != 2 or array.shape[1] != expected_n_cols:
        raise ValueError(
            f"{name} must have shape (n_samples, {expected_n_cols}), "
            f"but found {array.shape}."
        )

    return array


def apply_fixed_axis_transform(
    acc: np.ndarray,
    gyr: np.ndarray,
    transform_matrix: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply an optional fixed 3x3 axis transformation to Acc and Gyr.

    This function does not decide the matrix. Dataset-specific construction
    scripts must provide it only when the raw sensor convention is known.

    Example for Tobii HUCS -> Mobilise-D-like, if empirically verified:
        X_new =  Y_tobii
        Y_new = -X_tobii
        Z_new =  Z_tobii
    """

    acc = _as_float_2d("acc", acc)
    gyr = _as_float_2d("gyr", gyr)

    if transform_matrix is None:
        return acc.copy(), gyr.copy()

    matrix = np.asarray(transform_matrix, dtype=float)

    if matrix.shape != (3, 3):
        raise ValueError(
            "transform_matrix must have shape (3, 3), "
            f"but found {matrix.shape}."
        )

    acc_out = (matrix @ acc.T).T
    gyr_out = (matrix @ gyr.T).T

    return acc_out, gyr_out


def _find_first_valid_static_block_from_mask(
    static_reference_mask: np.ndarray,
    acc: np.ndarray,
    fs: float,
    static_duration_s: float,
    min_valid_static_samples: int,
) -> np.ndarray:
    """
    Return indices of the first continuous static block usable for alignment.
    """

    mask = np.asarray(static_reference_mask, dtype=bool)

    if mask.ndim != 1 or len(mask) != len(acc):
        raise ValueError(
            "static_reference_mask must be a 1D Boolean array with the same "
            "length as acc."
        )

    n_static = int(round(static_duration_s * fs))
    n_static = max(n_static, min_valid_static_samples)

    finite_acc_mask = np.isfinite(acc).all(axis=1)
    valid_mask = mask & finite_acc_mask
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < n_static:
        raise ValueError(
            "Not enough valid static samples to estimate gravity alignment. "
            f"Required at least {n_static}, found {len(valid_indices)}."
        )

    split_points = np.where(np.diff(valid_indices) > 1)[0] + 1
    blocks = np.split(valid_indices, split_points)

    for block in blocks:
        if len(block) >= n_static:
            return block[:n_static]

    raise ValueError(
        "Static samples exist, but no continuous static block is long enough "
        "for gravity alignment."
    )


def select_static_reference_indices(
    acc: np.ndarray,
    fs: float,
    static_duration_s: float = 1.0,
    static_reference_mask: Optional[np.ndarray] = None,
    min_valid_static_samples: int = 10,
) -> np.ndarray:
    """
    Select samples used to estimate gravity direction.

    If static_reference_mask is provided, the first continuous valid static block
    is used. Otherwise, the first static_duration_s seconds are used, matching
    the WearGaitPD construction logic.
    """

    acc = _as_float_2d("acc", acc)

    if static_reference_mask is not None:
        return _find_first_valid_static_block_from_mask(
            static_reference_mask=static_reference_mask,
            acc=acc,
            fs=fs,
            static_duration_s=static_duration_s,
            min_valid_static_samples=min_valid_static_samples,
        )

    n_static = int(round(static_duration_s * fs))
    n_static = min(n_static, len(acc))

    if n_static <= 0:
        raise ValueError("Static reference duration is zero or the trial is empty.")

    candidate_indices = np.arange(n_static, dtype=int)
    finite_mask = np.isfinite(acc[candidate_indices]).all(axis=1)
    finite_indices = candidate_indices[finite_mask]

    if len(finite_indices) < min_valid_static_samples:
        raise ValueError(
            "Not enough finite samples in the initial static reference to "
            "estimate gravity alignment."
        )

    return finite_indices


def detect_acceleration_unit_from_static_reference(
    acc: np.ndarray,
    reference_indices: np.ndarray,
) -> Tuple[str, float]:
    """
    Detect whether acceleration is likely stored in g or m/s^2.

    Returns
    -------
    unit : str
        "g", "m/s2", or "unknown".
    static_norm : float
        Norm of the mean static acceleration vector.
    """

    acc = _as_float_2d("acc", acc)
    reference_indices = np.asarray(reference_indices, dtype=int)

    acc_ref = acc[reference_indices]
    finite_mask = np.isfinite(acc_ref).all(axis=1)

    if not np.any(finite_mask):
        return "unknown", np.nan

    mean_static_acc = np.mean(acc_ref[finite_mask], axis=0)
    static_norm = float(np.linalg.norm(mean_static_acc))

    if 0.7 <= static_norm <= 1.3:
        return "g", static_norm

    if 7.0 <= static_norm <= 13.0:
        return "m/s2", static_norm

    return "unknown", static_norm


def convert_acceleration_unit(
    acc: np.ndarray,
    reference_indices: np.ndarray,
    input_unit: str = "auto",
    output_unit: str = "g",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert acceleration to the requested output unit.

    Recommended project setting:
        output_unit="g"

    because WearGaitPD/POLITO constructed datasets were found to operate in a
    normalized acceleration scale. The M3 gravity vector is estimated after unit
    conversion, so gravity removal remains unit-consistent.
    """

    acc = _as_float_2d("acc", acc)
    input_unit = str(input_unit).lower().replace(" ", "")
    output_unit = str(output_unit).lower().replace(" ", "")

    aliases = {
        "ms2": "m/s2",
        "m/s^2": "m/s2",
        "mps2": "m/s2",
        "g-units": "g",
        "gunits": "g",
    }

    input_unit = aliases.get(input_unit, input_unit)
    output_unit = aliases.get(output_unit, output_unit)

    detected_unit, static_norm = detect_acceleration_unit_from_static_reference(
        acc=acc,
        reference_indices=reference_indices,
    )

    if input_unit == "auto":
        effective_input_unit = detected_unit
    else:
        effective_input_unit = input_unit

    if effective_input_unit not in ["g", "m/s2"]:
        raise ValueError(
            "Could not determine acceleration unit. Pass input_unit='g' or "
            "input_unit='m/s2' explicitly. "
            f"Detected unit: {detected_unit}, static norm: {static_norm:.6f}."
        )

    if output_unit not in ["g", "m/s2"]:
        raise ValueError("output_unit must be either 'g' or 'm/s2'.")

    scale_factor = 1.0

    if effective_input_unit == "m/s2" and output_unit == "g":
        scale_factor = 1.0 / GRAVITY_MS2
    elif effective_input_unit == "g" and output_unit == "m/s2":
        scale_factor = GRAVITY_MS2

    acc_converted = acc * scale_factor

    debug_info = {
        "input_unit_requested": input_unit,
        "input_unit_detected": detected_unit,
        "input_unit_effective": effective_input_unit,
        "output_unit": output_unit,
        "static_reference_norm_before_unit_conversion": static_norm,
        "acc_scale_factor_applied": scale_factor,
    }

    return acc_converted, debug_info


def rotation_matrix_from_vectors(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
    """Compute the proper rotation matrix aligning source_vector to target_vector."""

    source = np.asarray(source_vector, dtype=float)
    target = np.asarray(target_vector, dtype=float)

    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)

    if source_norm < 1e-12 or target_norm < 1e-12:
        return np.eye(3)

    source = source / source_norm
    target = target / target_norm

    v = np.cross(source, target)
    c = np.dot(source, target)
    s = np.linalg.norm(v)

    if s < 1e-12 and c > 0:
        return np.eye(3)

    if s < 1e-12 and c < 0:
        orthogonal = np.array([1.0, 0.0, 0.0])

        if abs(source[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0])

        v = np.cross(source, orthogonal)
        v = v / np.linalg.norm(v)

        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

        return np.eye(3) + 2.0 * (vx @ vx)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


def align_imu_to_gravity_omit_nan(
    acc: np.ndarray,
    gyr: np.ndarray,
    fs: float,
    static_duration_s: float = 1.0,
    static_reference_mask: Optional[np.ndarray] = None,
    target_axis: int = 0,
    min_valid_static_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NaN-aware gravity alignment preserving missing rows.

    The mean gravity direction is estimated from the selected static reference,
    ignoring non-finite accelerometer rows. The raw signal is not interpolated
    before alignment; NaNs remain NaNs after rotation and are filled later only
    for numerical preprocessing.
    """

    acc = _as_float_2d("acc", acc)
    gyr = _as_float_2d("gyr", gyr)

    if len(acc) != len(gyr):
        raise ValueError("acc and gyr must have the same number of samples.")

    reference_indices = select_static_reference_indices(
        acc=acc,
        fs=fs,
        static_duration_s=static_duration_s,
        static_reference_mask=static_reference_mask,
        min_valid_static_samples=min_valid_static_samples,
    )

    acc_reference = acc[reference_indices]
    finite_reference_mask = np.isfinite(acc_reference).all(axis=1)

    mean_static_acc_raw = np.mean(acc_reference[finite_reference_mask], axis=0)
    gravity_norm = np.linalg.norm(mean_static_acc_raw)

    if not np.isfinite(gravity_norm) or gravity_norm < 1e-12:
        raise ValueError("Invalid gravity vector estimated from static reference.")

    target_vector = np.zeros(3, dtype=float)
    target_vector[target_axis] = gravity_norm

    R = rotation_matrix_from_vectors(
        source_vector=mean_static_acc_raw,
        target_vector=target_vector,
    )

    acc_aligned = acc @ R.T
    gyr_aligned = gyr @ R.T
    mean_static_acc_aligned = R @ mean_static_acc_raw

    return (
        acc_aligned,
        gyr_aligned,
        R,
        mean_static_acc_raw,
        mean_static_acc_aligned,
        reference_indices,
    )


def compute_imu_nan_quality_mask(
    X_raw: np.ndarray,
    gap_threshold_frames: int = DEFAULT_IMU_GAP_THRESHOLD_FRAMES,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute an IMU-only quality mask from original NaN gaps.

    A sample is marked as low quality only if it belongs to a contiguous NaN
    block whose length is >= gap_threshold_frames. This mirrors the IMU-quality
    part of the WearGaitPD/Anderson screening while intentionally excluding any
    walkway-contact logic.
    """

    X_raw = np.asarray(X_raw, dtype=float)

    if X_raw.ndim != 2:
        raise ValueError("X_raw must have shape (n_samples, n_channels).")

    n_samples = X_raw.shape[0]
    is_nan = np.isnan(X_raw).any(axis=1)
    labeled_nans, n_nan_blocks = ndimage.label(is_nan)

    imu_quality = np.ones(n_samples, dtype=bool)
    long_nan_blocks = []

    if n_nan_blocks > 0:
        nan_slices = ndimage.find_objects(labeled_nans)

        for block_id, sl in enumerate(nan_slices, start=1):
            start = int(sl[0].start)
            stop = int(sl[0].stop)
            length = stop - start

            if length >= gap_threshold_frames:
                imu_quality[sl] = False
                long_nan_blocks.append({
                    "block_id": block_id,
                    "start": start,
                    "stop": stop,
                    "length": length,
                })

    debug_info = {
        "gap_threshold_frames": int(gap_threshold_frames),
        "n_nan_blocks": int(n_nan_blocks),
        "n_long_nan_blocks": int(len(long_nan_blocks)),
        "n_low_quality_samples": int(np.sum(~imu_quality)),
        "n_high_quality_samples": int(np.sum(imu_quality)),
        "long_nan_blocks": long_nan_blocks,
    }

    return imu_quality, debug_info


def fill_all_gaps_after_alignment_for_preprocessing(X_aligned: np.ndarray) -> np.ndarray:
    """
    Fill all remaining NaNs after gravity alignment for M3/filtfilt only.

    Reliability must still be controlled later through imu_quality and labels.
    """

    X_df = pd.DataFrame(np.asarray(X_aligned, dtype=float))

    if (X_df.notna().sum(axis=0) == 0).any():
        raise ValueError(
            "At least one IMU channel is entirely NaN. Full-trial interpolation "
            "is not possible."
        )

    try:
        X_filled = X_df.interpolate(
            method="spline",
            order=3,
            axis=0,
            limit_direction="both",
        )
    except Exception:
        X_filled = X_df.interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )

    if X_filled.isna().any().any():
        X_filled = X_df.interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        X_filled = X_filled.bfill().ffill()

    if X_filled.isna().any().any():
        raise ValueError(
            "NaNs remain after full gap filling. M3 and low-pass filtering "
            "cannot be applied safely."
        )

    return X_filled.to_numpy(dtype=float)


def apply_zero_phase_lowpass(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filtering."""

    data = np.asarray(data, dtype=float)

    if data.ndim not in [1, 2]:
        raise ValueError("data must be 1D or 2D.")

    if np.isnan(data).any():
        raise ValueError("Low-pass filtering received NaN values.")

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyquist

    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        raise ValueError(
            f"Invalid cutoff_hz={cutoff_hz} for fs={fs}. Normalized cutoff "
            "must be between 0 and 1."
        )

    b, a = butter(N=order, Wn=normalized_cutoff, btype="lowpass")

    return filtfilt(b, a, data, axis=0)


def estimate_step_frequency_from_acc(
    acc: np.ndarray,
    fs: float,
    vertical_axis_idx: int = 0,
) -> float:
    """Estimate dominant gait frequency Fo from vertical + best horizontal PSD."""

    acc = _as_float_2d("acc", acc)
    n_samples = len(acc)

    if n_samples < fs:
        return 1.8

    freqs, psd_all = welch(
        acc,
        fs=fs,
        nperseg=min(n_samples, 1024),
        axis=0,
    )

    gait_band_mask = (freqs >= 0.5) & (freqs <= 3.0)

    if not np.any(gait_band_mask):
        return 1.8

    horizontal_axes = [idx for idx in range(3) if idx != vertical_axis_idx]
    horizontal_powers = [
        np.nansum(psd_all[gait_band_mask, axis_idx])
        for axis_idx in horizontal_axes
    ]
    best_horizontal_axis = horizontal_axes[int(np.argmax(horizontal_powers))]

    combined_psd = psd_all[:, vertical_axis_idx] + psd_all[:, best_horizontal_axis]
    gait_freqs = freqs[gait_band_mask]
    gait_psd = combined_psd[gait_band_mask]

    if len(gait_psd) == 0 or np.all(np.isnan(gait_psd)):
        return 1.8

    return float(gait_freqs[int(np.argmax(gait_psd))])


def apply_brodie_continuous_tilt_correction(
    acc: np.ndarray,
    fs: float,
    gravity_vector: np.ndarray,
    vertical_axis_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Apply Brodie/Buckley M3 continuous pitch-roll correction.

    Steps:
    1. estimate step frequency Fo;
    2. low-pass acceleration at Fo/4 to obtain LFA;
    3. compute sample-wise rotation aligning LFA to gravity_vector;
    4. apply the rotation to acceleration;
    5. subtract gravity_vector to obtain linear acceleration.
    """

    acc = _as_float_2d("acc", acc)
    gravity_vector = np.asarray(gravity_vector, dtype=float)

    if gravity_vector.shape != (3,):
        raise ValueError("gravity_vector must have shape (3,).")

    step_frequency_hz = estimate_step_frequency_from_acc(
        acc=acc,
        fs=fs,
        vertical_axis_idx=vertical_axis_idx,
    )

    m3_cutoff_hz = float(np.clip(step_frequency_hz / 4.0, 0.15, 1.0))

    lfa = apply_zero_phase_lowpass(
        data=acc,
        fs=fs,
        cutoff_hz=m3_cutoff_hz,
        order=4,
    )

    acc_corrected = np.zeros_like(acc)

    for sample_idx in range(len(acc)):
        R_i = rotation_matrix_from_vectors(
            source_vector=lfa[sample_idx],
            target_vector=gravity_vector,
        )
        acc_corrected[sample_idx] = R_i @ acc[sample_idx]

    acc_linear = acc_corrected - gravity_vector

    return acc_linear, acc_corrected, step_frequency_hz, m3_cutoff_hz


def apply_preprocessing_full_trial_after_alignment(
    acc_aligned_filled: np.ndarray,
    gyr_aligned_filled: np.ndarray,
    fs: float,
    gravity_vector: np.ndarray,
    lowpass_cutoff_hz: float = DEFAULT_LOW_PASS_CUTOFF_HZ,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply M3 correction, gravity removal and final low-pass filtering."""

    acc_aligned_filled = _as_float_2d("acc_aligned_filled", acc_aligned_filled)
    gyr_aligned_filled = _as_float_2d("gyr_aligned_filled", gyr_aligned_filled)

    if np.isnan(acc_aligned_filled).any() or np.isnan(gyr_aligned_filled).any():
        raise ValueError(
            "Full-trial preprocessing received NaN values. Run full gap filling "
            "after gravity alignment first."
        )

    acc_linear, acc_corrected, step_frequency_hz, m3_cutoff_hz = (
        apply_brodie_continuous_tilt_correction(
            acc=acc_aligned_filled,
            fs=fs,
            gravity_vector=gravity_vector,
            vertical_axis_idx=0,
        )
    )

    acc_final = apply_zero_phase_lowpass(
        data=acc_linear,
        fs=fs,
        cutoff_hz=lowpass_cutoff_hz,
        order=4,
    )

    gyr_final = apply_zero_phase_lowpass(
        data=gyr_aligned_filled,
        fs=fs,
        cutoff_hz=lowpass_cutoff_hz,
        order=4,
    )

    debug_info = {
        "step_frequency_hz": float(step_frequency_hz),
        "m3_cutoff_hz": float(m3_cutoff_hz),
        "final_lowpass_cutoff_hz": float(lowpass_cutoff_hz),
        "gravity_vector": np.asarray(gravity_vector, dtype=float).tolist(),
    }

    return acc_final, gyr_final, debug_info


def preprocess_head_imu_trial(
    acc_raw: np.ndarray,
    gyr_raw: np.ndarray,
    fs: float,
    static_duration_s: float = 1.0,
    static_reference_mask: Optional[np.ndarray] = None,
    fixed_axis_transform: Optional[np.ndarray] = None,
    acc_input_unit: str = "auto",
    acc_output_unit: str = "g",
    gyr_scale_factor: float = 1.0,
    gap_threshold_frames: int = DEFAULT_IMU_GAP_THRESHOLD_FRAMES,
    lowpass_cutoff_hz: float = DEFAULT_LOW_PASS_CUTOFF_HZ,
    min_valid_static_samples: int = 10,
) -> GSDPreprocessingResult:
    """
    Apply the unified WearGaitPD-like preprocessing chain to one trial.

    Parameters
    ----------
    acc_raw, gyr_raw
        Raw accelerometer and gyroscope arrays, shape (n_samples, 3).
    fs
        Sampling frequency in Hz. The current GSD construction uses 100 Hz.
    static_reference_mask
        Optional Boolean mask identifying static samples. If provided, the first
        continuous static block is used for gravity alignment. If omitted, the
        first static_duration_s seconds are used, as in WearGaitPD.
    fixed_axis_transform
        Optional 3x3 matrix to convert a dataset-specific sensor convention to
        the Mobilise-D-like convention before gravity alignment. Use None when
        the construction already provides VT/ML/AP-like axes.
    acc_input_unit
        "auto", "g", or "m/s2". With "auto", the static-reference norm is used.
    acc_output_unit
        "g" or "m/s2". Recommended project setting is "g" to match the current
        WearGaitPD-like feature space.
    gyr_scale_factor
        Optional multiplicative factor for gyroscope harmonization. It is kept
        explicit because gyroscope units cannot be inferred from static gravity.
    """

    acc_raw = _as_float_2d("acc_raw", acc_raw)
    gyr_raw = _as_float_2d("gyr_raw", gyr_raw)

    if len(acc_raw) != len(gyr_raw):
        raise ValueError("acc_raw and gyr_raw must have the same number of samples.")

    # The IMU quality mask is computed from original missing-data regions before
    # any interpolation can hide the dropout pattern.
    X_quality_raw = np.concatenate([acc_raw, gyr_raw], axis=1)
    imu_quality, quality_debug_info = compute_imu_nan_quality_mask(
        X_raw=X_quality_raw,
        gap_threshold_frames=gap_threshold_frames,
    )

    acc_sensor, gyr_sensor = apply_fixed_axis_transform(
        acc=acc_raw,
        gyr=gyr_raw,
        transform_matrix=fixed_axis_transform,
    )

    gyr_sensor = gyr_sensor * float(gyr_scale_factor)

    reference_indices_for_unit = select_static_reference_indices(
        acc=acc_sensor,
        fs=fs,
        static_duration_s=static_duration_s,
        static_reference_mask=static_reference_mask,
        min_valid_static_samples=min_valid_static_samples,
    )

    acc_unit_converted, unit_debug_info = convert_acceleration_unit(
        acc=acc_sensor,
        reference_indices=reference_indices_for_unit,
        input_unit=acc_input_unit,
        output_unit=acc_output_unit,
    )

    (
        acc_aligned_raw,
        gyr_aligned_raw,
        R,
        mean_static_acc_raw,
        mean_static_acc_aligned,
        reference_indices_for_alignment,
    ) = align_imu_to_gravity_omit_nan(
        acc=acc_unit_converted,
        gyr=gyr_sensor,
        fs=fs,
        static_duration_s=static_duration_s,
        static_reference_mask=static_reference_mask,
        target_axis=0,
        min_valid_static_samples=min_valid_static_samples,
    )

    X_aligned_raw = np.concatenate([acc_aligned_raw, gyr_aligned_raw], axis=1)
    X_aligned_filled = fill_all_gaps_after_alignment_for_preprocessing(X_aligned_raw)

    acc_aligned_filled = X_aligned_filled[:, 0:3]
    gyr_aligned_filled = X_aligned_filled[:, 3:6]

    gravity_vector = mean_static_acc_aligned.copy()

    acc_final, gyr_final, preprocessing_debug_info = apply_preprocessing_full_trial_after_alignment(
        acc_aligned_filled=acc_aligned_filled,
        gyr_aligned_filled=gyr_aligned_filled,
        fs=fs,
        gravity_vector=gravity_vector,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )

    X = np.concatenate([acc_final, gyr_final], axis=1)

    mean_static_acc_norm = mean_static_acc_aligned / np.linalg.norm(mean_static_acc_aligned)

    preprocessing_debug_info.update(unit_debug_info)
    preprocessing_debug_info.update({
        "static_duration_s": float(static_duration_s),
        "static_reference_mode": "mask_first_valid_block" if static_reference_mask is not None else "first_n_seconds",
        "n_static_reference_samples_for_unit": int(len(reference_indices_for_unit)),
        "n_static_reference_samples_for_alignment": int(len(reference_indices_for_alignment)),
        "mean_static_acc_raw_after_unit_conversion": mean_static_acc_raw.tolist(),
        "mean_static_acc_aligned": mean_static_acc_aligned.tolist(),
        "mean_static_acc_aligned_norm": mean_static_acc_norm.tolist(),
        "fixed_axis_transform_applied": fixed_axis_transform is not None,
        "gyr_scale_factor": float(gyr_scale_factor),
    })

    return GSDPreprocessingResult(
        X=X.astype(np.float32),
        acc_final=acc_final,
        gyr_final=gyr_final,
        acc_aligned_raw=acc_aligned_raw,
        gyr_aligned_raw=gyr_aligned_raw,
        acc_aligned_filled=acc_aligned_filled,
        gyr_aligned_filled=gyr_aligned_filled,
        rotation_matrix=R,
        mean_static_acc_raw=mean_static_acc_raw,
        mean_static_acc_aligned=mean_static_acc_aligned,
        gravity_vector=gravity_vector,
        imu_quality=imu_quality,
        quality_debug_info=quality_debug_info,
        preprocessing_debug_info=preprocessing_debug_info,
    )


def window_is_accepted_by_unified_preprocessing(
    labels: np.ndarray,
    imu_quality: np.ndarray,
    start: int,
    window_size: int,
    none_label_value: int = -1,
) -> bool:
    """
    Return True only if a window has high IMU quality and no none labels.

    This implements the intended external-validation rule:
    - reject windows touching long original NaN gaps;
    - reject windows with at least one none-labelled sample.
    """

    labels = np.asarray(labels)
    imu_quality = np.asarray(imu_quality, dtype=bool)

    stop = start + window_size

    if stop > len(labels) or stop > len(imu_quality):
        return False

    if not np.all(imu_quality[start:stop]):
        return False

    if np.any(labels[start:stop] == none_label_value):
        return False

    return True
