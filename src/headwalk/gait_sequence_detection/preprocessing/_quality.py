"""
Quality-mask utilities for Gait Sequence Detection.

The module separates two concepts:

1. IMU quality
   Rejects samples belonging to long original missing-data regions.
   Short gaps are kept because they can be interpolated during preprocessing.

2. Optional instrumented-walkway / instrumented-mat quality
   Rejects samples where foot-contact information indicates unreliable
   walkway/mat support, for datasets/tasks where this information is available
   and meaningful.

The final quality mask is:
- imu_quality only, when use_walkway_quality=False;
- imu_quality AND walkway_quality, when use_walkway_quality=True.

Quality masks do not modify labels or signals directly. They are intended to be
used later to reject unreliable windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.ndimage as ndimage


@dataclass(frozen=True)
class GSDQualityMasks:
    """Container for sample-wise quality masks."""

    imu_quality: np.ndarray
    walkway_quality: Optional[np.ndarray]
    final_quality: np.ndarray
    debug_info: dict


def _as_2d_float_array(name: str, x: np.ndarray) -> np.ndarray:
    """Convert input to a finite-shape 2D float array."""

    arr = np.asarray(x, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")

    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty.")

    return arr


def _as_1d_float_array(name: str, x: np.ndarray, n_samples: int) -> np.ndarray:
    """Convert input to a 1D float array with expected length."""

    arr = np.asarray(x, dtype=float).reshape(-1)

    if len(arr) != n_samples:
        raise ValueError(
            f"{name} length mismatch: expected {n_samples}, got {len(arr)}."
        )

    return arr


def compute_imu_nan_quality_mask(
    X_raw: np.ndarray,
    *,
    gap_threshold_frames: int = 7,
) -> tuple[np.ndarray, dict]:
    """
    Compute IMU quality from original missing-data regions.

    Parameters
    ----------
    X_raw
        Raw IMU array, usually shape (n_samples, 6):
        Acc X/Y/Z + Gyr X/Y/Z.

    gap_threshold_frames
        Contiguous NaN blocks with length >= this value are rejected.
        Shorter gaps remain valid because they can be interpolated during
        preprocessing.

    Returns
    -------
    imu_quality
        Boolean mask, True = usable sample.

    debug_info
        Diagnostic information about detected NaN blocks.
    """

    X_raw = _as_2d_float_array("X_raw", X_raw)

    if gap_threshold_frames <= 0:
        raise ValueError("gap_threshold_frames must be positive.")

    n_samples = X_raw.shape[0]

    is_nan = np.isnan(X_raw).any(axis=1)

    labeled_nans, n_nan_blocks = ndimage.label(is_nan)

    imu_quality = np.ones(n_samples, dtype=bool)

    rejected_nan_blocks = 0
    rejected_nan_samples = 0
    nan_block_lengths = []

    if n_nan_blocks > 0:
        nan_slices = ndimage.find_objects(labeled_nans)

        for sl in nan_slices:
            if sl is None:
                continue

            start = int(sl[0].start)
            stop = int(sl[0].stop)
            block_len = stop - start
            nan_block_lengths.append(block_len)

            if block_len >= gap_threshold_frames:
                imu_quality[start:stop] = False
                rejected_nan_blocks += 1
                rejected_nan_samples += block_len

    debug_info = {
        "n_samples": int(n_samples),
        "gap_threshold_frames": int(gap_threshold_frames),
        "n_nan_blocks": int(n_nan_blocks),
        "nan_block_lengths": [int(v) for v in nan_block_lengths],
        "n_rejected_nan_blocks": int(rejected_nan_blocks),
        "n_rejected_nan_samples": int(rejected_nan_samples),
        "n_imu_quality_samples": int(np.sum(imu_quality)),
    }

    return imu_quality, debug_info


def compute_walkway_contact_quality_mask(
    l_foot_contact: np.ndarray,
    r_foot_contact: np.ndarray,
    *,
    n_samples: int,
    edge_buffer_frames: int = 50,
) -> tuple[np.ndarray, dict]:
    """
    Compute sample-wise quality from instrumented walkway/mat foot contacts.

    The contact signals are interpreted as Boolean-like:
    - 1 = foot detected in contact;
    - 0 = foot not detected in contact.

    The valid walkway/mat region is based on:
    - the dynamic interval between the first and last contact transition;
    - either-foot-ground-contact = left contact OR right contact;
    - removal of an edge buffer at the start and end of each valid region.

    This function should be used only when the raw trial really contains
    meaningful, sample-aligned left/right foot-contact signals.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if edge_buffer_frames < 0:
        raise ValueError("edge_buffer_frames cannot be negative.")

    l_foot_contact = _as_1d_float_array(
        "l_foot_contact",
        l_foot_contact,
        n_samples,
    )
    r_foot_contact = _as_1d_float_array(
        "r_foot_contact",
        r_foot_contact,
        n_samples,
    )

    l_foot_contact = np.nan_to_num(l_foot_contact, nan=0.0).astype(int)
    r_foot_contact = np.nan_to_num(r_foot_contact, nan=0.0).astype(int)

    l_transitions = np.abs(
        np.diff(l_foot_contact, prepend=l_foot_contact[0])
    )
    r_transitions = np.abs(
        np.diff(r_foot_contact, prepend=r_foot_contact[0])
    )

    any_transition = (l_transitions > 0) | (r_transitions > 0)
    transition_indices = np.where(any_transition)[0]

    walkway_quality = np.zeros(n_samples, dtype=bool)

    if len(transition_indices) == 0:
        debug_info = {
            "use_walkway_quality": True,
            "n_contact_transitions": 0,
            "first_dynamic_event": None,
            "last_dynamic_event": None,
            "edge_buffer_frames": int(edge_buffer_frames),
            "n_walkway_quality_samples": 0,
        }
        return walkway_quality, debug_info

    first_dynamic_event = int(transition_indices[0])
    last_dynamic_event = int(transition_indices[-1])

    movement_mask = np.zeros(n_samples, dtype=bool)
    movement_mask[first_dynamic_event:last_dynamic_event + 1] = True

    either_foot_ground_contact = (
        (l_foot_contact == 1)
        |
        (r_foot_contact == 1)
    )

    walkway_active = movement_mask & either_foot_ground_contact
    walkway_quality = walkway_active.copy()

    active_int = walkway_active.astype(int)
    diff_active = np.diff(active_int, prepend=0)

    starts = np.where(diff_active == 1)[0]
    ends = np.where(diff_active == -1)[0]

    if len(active_int) > 0 and active_int[-1] == 1:
        ends = np.append(ends, n_samples)

    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)

        walkway_quality[
            start:min(start + edge_buffer_frames, n_samples)
        ] = False

        walkway_quality[
            max(0, end - edge_buffer_frames):end
        ] = False

    debug_info = {
        "use_walkway_quality": True,
        "n_contact_transitions": int(len(transition_indices)),
        "first_dynamic_event": first_dynamic_event,
        "last_dynamic_event": last_dynamic_event,
        "edge_buffer_frames": int(edge_buffer_frames),
        "n_walkway_quality_samples": int(np.sum(walkway_quality)),
    }

    return walkway_quality, debug_info


def compute_gsd_quality_masks(
    X_raw: np.ndarray,
    *,
    l_foot_contact: Optional[np.ndarray] = None,
    r_foot_contact: Optional[np.ndarray] = None,
    gap_threshold_frames: int = 7,
    edge_buffer_frames: int = 50,
    use_walkway_quality: bool = False,
) -> GSDQualityMasks:
    """
    Compute final sample-wise quality masks for GSD.

    If use_walkway_quality=False:
        final_quality = imu_quality
        walkway_quality = None

    If use_walkway_quality=True:
        final_quality = imu_quality AND walkway_quality
        l_foot_contact and r_foot_contact are required.
    """

    X_raw = _as_2d_float_array("X_raw", X_raw)
    n_samples = X_raw.shape[0]

    imu_quality, imu_debug_info = compute_imu_nan_quality_mask(
        X_raw,
        gap_threshold_frames=gap_threshold_frames,
    )

    debug_info = {
        "use_walkway_quality": bool(use_walkway_quality),
        **imu_debug_info,
    }

    if not use_walkway_quality:
        final_quality = imu_quality.copy()
        debug_info.update({
            "n_contact_transitions": None,
            "first_dynamic_event": None,
            "last_dynamic_event": None,
            "edge_buffer_frames": None,
            "n_walkway_quality_samples": None,
            "n_final_quality_samples": int(np.sum(final_quality)),
        })

        return GSDQualityMasks(
            imu_quality=imu_quality,
            walkway_quality=None,
            final_quality=final_quality,
            debug_info=debug_info,
        )

    if l_foot_contact is None or r_foot_contact is None:
        raise ValueError(
            "use_walkway_quality=True requires both l_foot_contact and r_foot_contact."
        )

    walkway_quality, walkway_debug_info = compute_walkway_contact_quality_mask(
        l_foot_contact=l_foot_contact,
        r_foot_contact=r_foot_contact,
        n_samples=n_samples,
        edge_buffer_frames=edge_buffer_frames,
    )

    final_quality = imu_quality & walkway_quality

    debug_info.update(walkway_debug_info)
    debug_info["n_final_quality_samples"] = int(np.sum(final_quality))

    return GSDQualityMasks(
        imu_quality=imu_quality,
        walkway_quality=walkway_quality,
        final_quality=final_quality,
        debug_info=debug_info,
    )


def compute_weargait_quality_masks(
    X_raw: np.ndarray,
    l_foot_contact: Optional[np.ndarray],
    r_foot_contact: Optional[np.ndarray],
    gap_threshold_frames: int = 7,
    edge_buffer_frames: int = 50,
    use_walkway_quality: bool = True,
):
    """
    Backward-compatible wrapper matching the old WearGaitPD construction API.

    Returns
    -------
    imu_quality, walkway_quality, final_quality, debug_info
    """

    masks = compute_gsd_quality_masks(
        X_raw=X_raw,
        l_foot_contact=l_foot_contact,
        r_foot_contact=r_foot_contact,
        gap_threshold_frames=gap_threshold_frames,
        edge_buffer_frames=edge_buffer_frames,
        use_walkway_quality=use_walkway_quality,
    )

    return (
        masks.imu_quality,
        masks.walkway_quality,
        masks.final_quality,
        masks.debug_info,
    )


def find_walkway_contact_columns(
    columns: Sequence[str],
    *,
    left_candidates: Sequence[str] = (
        "L Foot Contact",
        "left_foot_contact",
        "l_foot_contact",
    ),
    right_candidates: Sequence[str] = (
        "R Foot Contact",
        "right_foot_contact",
        "r_foot_contact",
    ),
) -> tuple[Optional[str], Optional[str]]:
    """
    Find left/right foot-contact columns using case-insensitive matching.

    Returns
    -------
    left_column, right_column
        Column names as present in the input table, or None if not found.
    """

    normalized_to_original = {
        str(col).strip().lower(): str(col)
        for col in columns
    }

    left_column = None
    right_column = None

    for candidate in left_candidates:
        key = str(candidate).strip().lower()
        if key in normalized_to_original:
            left_column = normalized_to_original[key]
            break

    for candidate in right_candidates:
        key = str(candidate).strip().lower()
        if key in normalized_to_original:
            right_column = normalized_to_original[key]
            break

    return left_column, right_column


def window_is_accepted_by_final_quality(
    labels: np.ndarray,
    final_quality: np.ndarray,
    start: int,
    window_size: int,
    none_label_value: int = -1,
) -> bool:
    """
    Return True only if a GSD window is fully valid.

    A window is accepted only when:
    1. it lies completely inside the signal;
    2. all samples have final_quality == True;
    3. no sample has the none / invalid label.

    Notes
    -----
    final_quality is expected to be:
    - imu_quality for datasets/tasks without instrumented walkway quality;
    - imu_quality AND walkway_quality for datasets/tasks where
      use_walkway_quality=True.

    This function is intended for labelled dataset construction and validation.
    For completely raw, unlabelled inference, labels may not be available; in
    that case only the quality-mask part can be applied by the caller.
    """

    labels = np.asarray(labels)
    final_quality = np.asarray(final_quality, dtype=bool)

    stop = int(start) + int(window_size)

    if start < 0:
        return False

    if stop > len(labels) or stop > len(final_quality):
        return False

    if not np.all(final_quality[start:stop]):
        return False

    if np.any(labels[start:stop] == none_label_value):
        return False

    return True


def window_is_accepted_by_quality_only(
    final_quality: np.ndarray,
    start: int,
    window_size: int,
) -> bool:
    """
    Return True only if all samples in an unlabelled window are high quality.

    This variant is useful for raw inference, where no sample-wise labels exist.
    It does not check none labels because they are unavailable.
    """

    final_quality = np.asarray(final_quality, dtype=bool)

    stop = int(start) + int(window_size)

    if start < 0:
        return False

    if stop > len(final_quality):
        return False

    return bool(np.all(final_quality[start:stop]))
