"""
This module contains reusable preprocessing functions for IMU signals.

Its goal is to transform raw inertial data into a clean and consistent
representation suitable for downstream gait analysis methods.

Typical operations include:
- signal validation
- filtering
- normalization
- resampling
- windowing
- timestamp handling
- simple signal transformations

These functions are method-independent and can be reused across different
pipeline blocks and algorithmic approaches.
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt, iirnotch, resample_poly
from math import gcd


def validate_signal(signal):
    """
    Validate and convert the input signal to a NumPy array.

    The expected format is:
    - 1D array for single-channel signals
    - 2D array of shape (N, C) for multichannel signals
    """
    signal = np.asarray(signal)

    # ValueError if the signal dimension is neither one between 1 and 2.
    if signal.ndim not in [1, 2]:
        raise ValueError("Signal must be 1D or 2D.")

    return signal


def remove_mean(signal):
    """
    Remove the mean value from the signal channel-wise.
    """
    # Accept only signals 1D or 2D.
    # Signal is organized in a NxC structure with N samples for each of the C axes.
    signal = validate_signal(signal)

    # if 1D:
    if signal.ndim == 1:
        return signal - np.mean(signal)
    # else:
    # axis=0 computes column-wise mean. axis=1 computes row-wise mean.
    # keepdims=True makes the mean maintaining the shape (1,C),
    # allowing automatic difference (since -row_vector is easy to compute).
    return signal - np.mean(signal, axis=0, keepdims=True)


def zscore_normalization(signal):
    """
    Apply z-score normalization channel-wise.
    """
    signal = validate_signal(signal)

    if signal.ndim == 1:
        # Compute signal std.
        std = np.std(signal)
        # z-score normalization: subtracts the mean and divides by std.
        # if std = 0 (constant signal), returns signal, avoiding zero-division.
        return (signal - np.mean(signal)) / std if std > 0 else signal

    # column-wise mean (each axis X, Y, Z) and std.
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    # Avoid zero-division forcing std = 1 whether std==0.
    std[std == 0] = 1.0

    return (signal - mean) / std # zero-score normalization


def minmax_normalization(signal, feature_range=(0, 1)):
    """
    Apply min-max normalization channel-wise.
    Aim at converting signal range from an original interval into a new one (0,1).
    Dangerous since it has high outlier-sensibility.
    """
    signal = validate_signal(signal)
    # min_val = 0, max_val = 1.
    min_val, max_val = feature_range

    if signal.ndim == 1:
        # Extract the min and max values from signal.
        s_min = np.min(signal)
        s_max = np.max(signal)
        # If constant signal, normalized signal equals to original signal.
        if s_max == s_min:
            return signal
        # Else, signal is normalized subtracting its minimum and dividing
        # by the range (s_max-s_min). Then, multiply this value for the
        # new feature range and adds min_val to center the signal around the correct value.
        return (signal - s_min) / (s_max - s_min) * (max_val - min_val) + min_val

    s_min = np.min(signal, axis=0, keepdims=True)
    s_max = np.max(signal, axis=0, keepdims=True)
    denom = s_max - s_min
    # Avoid zero-division.
    denom[denom == 0] = 1.0

    # min-max scaling.
    return (signal - s_min) / denom * (max_val - min_val) + min_val


def robust_scaling(signal):
    """
    Apply robust scaling channel-wise using median and IQR.
    """
    signal = validate_signal(signal)

    if signal.ndim == 1:
        # median computation.
        median = np.median(signal)
        q1 = np.percentile(signal, 25) # 25-th percentile
        q3 = np.percentile(signal, 75) # 75-th percentile
        iqr = q3 - q1 # interquartile range definition
        # Signal is robust-scaled subtracting its median value and
        # dividing by iqr (if greater than zero, or rather not constant),
        # otherwise, normalization outputs original signal.
        return (signal - median) / iqr if iqr > 0 else signal

    median = np.median(signal, axis=0, keepdims=True)
    q1 = np.percentile(signal, 25, axis=0, keepdims=True)
    q3 = np.percentile(signal, 75, axis=0, keepdims=True)
    iqr = q3 - q1
    # Avoids zero-division.
    iqr[iqr == 0] = 1.0

    return (signal - median) / iqr


def butter_filter(signal, fs, cutoff, btype="low", order=4):
    """
    Apply a Butterworth filter channel-wise.

    Parameters:
    signal: array-like
        Input signal, shape (N,1) or (N, C)
    fs: float
        Sampling frequency
    cutoff: float or tuple
        Cutoff frequency in Hz
    btype: str
        'low', 'high', or 'band'
    order: int
        Filter order
    """
    signal = validate_signal(signal)

    # Nyquist frequency definition as a half of sampling frequency.
    nyq = 0.5 * fs

    # Cutoff frequency normalization dividing by f_Nyquist.
    # In the case of low- or high-pass filter, there is one f_cut.
    if btype in ["low", "high"]:
        wn = cutoff / nyq
    # In the case of band-pass filter, there are two f_cut that define a 2-element list.
    elif btype == "band":
        wn = [cutoff[0] / nyq, cutoff[1] / nyq]
    else: # Error in btype setting.
        raise ValueError("btype must be 'low', 'high', or 'band'.")

    # Extract b and a (Butterworth filter coefficients).
    b, a = butter(order, wn, btype=btype)

    # Zero-phase filtering applied forward and backward along each signal channel
    return filtfilt(b, a, signal, axis=0)


def lowpass_filter(signal, fs, cutoff, order=4):
    """
    Apply a low-pass Butterworth filter.
    """
    return butter_filter(signal, fs, cutoff, btype="low", order=order)


def highpass_filter(signal, fs, cutoff, order=4):
    """
    Apply a high-pass Butterworth filter.
    """
    return butter_filter(signal, fs, cutoff, btype="high", order=order)


def bandpass_filter(signal, fs, low_cutoff, high_cutoff, order=4):
    """
    Apply a band-pass Butterworth filter.
    """
    return butter_filter(signal, fs, (low_cutoff, high_cutoff), btype="band", order=order)


def notch_filter(signal, fs, notch_freq, quality_factor=30):
    """
    Apply a notch filter channel-wise to remove a narrowband interference.
    """
    signal = validate_signal(signal)
    # Nyquist frequency definition.
    nyq = 0.5 * fs
    # Notch frequency normalization using Nyquist frequency.
    w0 = notch_freq / nyq
    # Extract filter coefficients b and a for iirnotch filter.
    b, a = iirnotch(w0, quality_factor)
    # Zero-phase filtering applied forward and backward along each signal channel
    return filtfilt(b, a, signal, axis=0)


def median_filter(signal, kernel_size=3):
    """
    Apply a median filter channel-wise.
    """
    signal = validate_signal(signal)

    if signal.ndim == 1:
        # median-filtering signal using specified kernel_size.
        return medfilt(signal, kernel_size=kernel_size)

    # Initialize filtered signal.
    filtered = np.zeros_like(signal)
    # Column-wise (X,Y,Z) median filtering on signal.
    for ch in range(signal.shape[1]):
        # signal[:, ch] isolates a column,
        # filtered[:, ch] saves the result within the same column in filtered signal.
        filtered[:, ch] = medfilt(signal[:, ch], kernel_size=kernel_size)

    return filtered


def compute_signal_magnitude(signal):
    """
    Compute the vector magnitude of a tri-axial signal.

    Expected input shape: (N, 3)
    """
    signal = validate_signal(signal)

    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError("Signal magnitude requires shape (N, 3).")

    # Column-wise 2-norm (default selected norm).
    return np.linalg.norm(signal, axis=1)


def resample_signal(signal, original_fs, target_fs):
    """
    Resample a signal from original_fs to target_fs using polyphase resampling.
    """
    signal = validate_signal(signal)

    # If original sampling frequencies equals target one,
    # resampling is not necessary.
    if original_fs == target_fs:
        return signal

    # gcd (greatest common divisor) between two frequencies.
    ratio_gcd = gcd(int(original_fs), int(target_fs))

    # target_fs / original_fs = up / down
    # e.g. from 100 Hz to 50 Hz: gcd(50, 100) = 50
    # up = 50/50 = 1, down = 100/50 = 2 -> so it is splitting
    # in half the original frequency. 50:100 = 1:2
    up = int(target_fs // ratio_gcd)
    down = int(original_fs // ratio_gcd)

    # resample_poly is a function that firstly upsamples multiplying
    # by up (x1) and then downsamples dividing by down (÷2). This is
    # a necessary compromise that allows resampling when the ratio between
    # original and target frequencies is not integer; resample_poly leverages
    # common divisors between the two frequencies and, since it is not possible
    # to multiply by for example 0.6, firstly it multiplies by 3 and then divides by 5.
    return resample_poly(signal, up, down, axis=0)


def create_sliding_windows(signal, window_size, step_size):
    """
    Create sliding windows from a 1D or 2D signal.

    Parameters
    ----------
    signal : array-like
        Input signal of shape (N,1) or (N, C)
    window_size : int
        Number of samples in each window
    step_size : int
        Number of samples between consecutive windows

    Returns
    -------
    windows : np.ndarray
        Shape:
        - (num_windows, window_size)- for 1D signals
        - (num_windows, window_size, C) for 2D signals
    """
    signal = validate_signal(signal)
    # Input signals are provided in 2 possible format but both
    # have samples row-wise organized.
    n_samples = signal.shape[0]

    if window_size > n_samples:
        raise ValueError("window_size cannot be greater than signal length.")

    windows = []
    # start is the first sample identified per each window: it is selected
    # in the range (0, n_samples - window_size + 1) using step_size as offset
    # between consecutive windows.
    for start in range(0, n_samples - window_size + 1, step_size):
        # end is the final sample identified per each window: start + window_size.
        end = start + window_size
        # windows list is updated with signal samples between start and end samples.
        windows.append(signal[start:end])

    return np.array(windows)


def create_time_windows(time_vector, window_size, step_size):
    """
    Create sliding windows from a time vector using create_sliding_window
    function defined above.
    """
    time_vector = np.asarray(time_vector).squeeze()

    if time_vector.ndim != 1:
        raise ValueError("Time vector must be 1D.")

    return create_sliding_windows(time_vector, window_size, step_size)