"""
This module contains functions for signal and frame alignment.

Its goal is to make sensor signals comparable across acquisitions by
handling reference-frame consistency and orientation-related preprocessing.

Typical operations include:
- vector normalization
- gravity-based alignment
- axis reorientation
- rotation of tri-axial signals
- gravity removal after alignment

These functions are especially useful when methods assume a specific
sensor orientation or reference-frame convention.
"""

import numpy as np


def normalize_vector(v):
    """
    Normalize a vector to unit norm.

    Parameters
    v : array-like
        Input vector

    Returns
    v_unit : np.ndarray
        Normalized vector
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return v / norm


def estimate_gravity_vector(acc_signal):
    """
    Estimate the gravity direction from the mean accelerometer signal.

    Parameters
    acc_signal: np.ndarray
        Accelerometer signal of shape (N, 3)

    Returns
    gravity_vector: np.ndarray
        Mean gravity direction estimate, shape (3,1)
    """
    acc_signal = np.asarray(acc_signal, dtype=float)

    # Verify acc_signal dimensions (N,3): N samples acquired for each of the three axes.
    if acc_signal.ndim != 2 or acc_signal.shape[1] != 3:
        raise ValueError("acc_signal must have shape (N, 3).")

    # Gravity vector defined as mean value of acceleration signal for each axis (column).
    gravity_vector = np.mean(acc_signal, axis=0)
    return gravity_vector


def skew_symmetric(v):
    """
    Build the skew-symmetric matrix associated with a 3D vector.
    """
    v = np.asarray(v, dtype=float)

    # Compute antisymmetric matrix associated to v vector. It is a standardized structure.
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def rotation_matrix_from_vectors(source_vector, target_vector):
    """
    Compute the rotation matrix that aligns source_vector to target_vector.

    Parameters
    source_vector: array-like with initial direction
    target_vector: array-like with desired direction

    Returns
    R: rotation matrix of shape (3, 3).
    """
    # Normalize vectors applied on source and target vectors.
    a = normalize_vector(source_vector)
    b = normalize_vector(target_vector)

    # cross-product and dot-product computed between a and b.
    cross = np.cross(a, b)
    dot = np.dot(a, b)

    # If dot product outputs 1, it means that target has the same direction as original vector.
    # In this case, rotation matrix is defined as eye matrix.
    if np.isclose(dot, 1.0):
        return np.eye(3)

    # Manage exception whether dot product equals -1.
    if np.isclose(dot, -1.0):
        raise ValueError("180-degree rotation case is ambiguous and requires a chosen rotation axis.")

    # K and R serve the purpose of building rotation matrix R that allows rotating
    # vectors from initial coordinate system into gravity-aligned one.
    K = skew_symmetric(cross)

    # Rodrigues formula to define rotation matrix R.
    # It is computationally efficient and does not require
    # iterative optimization or orientation tracking.
    # Alternatives were quaternions, Euler angles or Madgwick/Kalman: these three options
    # are very refined and complex to implement.
    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + dot)) # K @ K computes matrices product.

    return R


def apply_rotation(signal, rotation_matrix):
    """
    Apply a 3D rotation matrix to a tri-axial signal.

    Parameters
    signal: signal of shape (N, 3) provided as ndarray
    rotation_matrix: rotation matrix of shape (3, 3) provided as ndarray

    Returns
    rotated_signal: rotated signal of shape (N, 3) as ndarray
    """
    signal = np.asarray(signal, dtype=float)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)

    # Manage exception whether shape not equals (N,3).
    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError("signal must have shape (N, 3).")

    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3).")

    return signal @ rotation_matrix.T


def align_signal_to_gravity(acc_signal, reference_axis=np.array([0.0, 0.0, 1.0])):
    """
    Align an accelerometer signal so that its mean gravity direction matches
    a chosen reference axis.

    Parameters
    acc_signal: accelerometer signal of shape (N, 3) provided as ndarray
    reference_axis: desired gravity-aligned axis provided as array

    Returns
    aligned_acc: rotated accelerometer signal (ndarray)
    R: rotation matrix used for alignment (ndarray)
    gravity_vector: estimated gravity vector before alignment
    """
    gravity_vector = estimate_gravity_vector(acc_signal)
    R = rotation_matrix_from_vectors(gravity_vector, reference_axis)
    aligned_acc = apply_rotation(acc_signal, R)

    return aligned_acc, R, gravity_vector


def align_trial_signals_to_gravity(acc_signal, gyr_signal=None, mag_signal=None,
                                   reference_axis=np.array([0.0, 0.0, 1.0])):
    """
    Align all tri-axial signals of one trial using the gravity direction
    estimated from the accelerometer.

    Parameters
    acc_signal: accelerometer signal of shape (N, 3), ndarray
    gyr_signal: gyroscope signal of shape (N, 3), np.ndarray or None
    mag_signal: magnetometer signal of shape (N, 3), np.ndarray or None
    reference_axis: desired gravity-aligned axis, array-like

    Returns
    aligned_data: dictionary containing aligned signals and alignment information
    """
    # align signal to gravity outputs aligned acceleration and rotation matrix R that
    # serves a purpose of allow gyro and magnetometer signals rotation.
    aligned_acc, R, gravity_vector = align_signal_to_gravity(
        acc_signal=acc_signal,
        reference_axis=reference_axis
    )

    aligned_data = {
        "Acc": aligned_acc,
        "RotationMatrix": R,
        "EstimatedGravity": gravity_vector
    }

    if gyr_signal is not None:
        aligned_data["Gyr"] = apply_rotation(gyr_signal, R)

    if mag_signal is not None:
        aligned_data["Mag"] = apply_rotation(mag_signal, R)

    return aligned_data


def remove_gravity_component(acc_signal, gravity_axis=2):
    """
    Remove the constant gravity component from an already aligned accelerometer signal.

    Parameters
    acc_signal: accelerometer signal of shape (N, 3), provided as ndarray
    gravity_axis: axis along which gravity is expected after alignment, provided as integer.
    in this case, gravity_axis is 2 since the gravity direction is expected to be along z axis.

    Returns
    acc_without_gravity: accelerometer signal with gravity removed on the selected axis, ndarray
    """
    acc_signal = np.asarray(acc_signal, dtype=float)

    if acc_signal.ndim != 2 or acc_signal.shape[1] != 3:
        raise ValueError("acc_signal must have shape (N, 3).")

    acc_without_gravity = acc_signal.copy()

    # All the row along the column defined using gravity axis (2 = z), then detrend.
    # The idea is: after alignment, gravity should be basically entirely on one axis.
    # This axis contains a constant component around g. Subtracting mean value from each
    # axis helps retain dynamic components of signal.
    acc_without_gravity[:, gravity_axis] = acc_without_gravity[:, gravity_axis] - np.mean(
        acc_without_gravity[:, gravity_axis]
    )

    return acc_without_gravity