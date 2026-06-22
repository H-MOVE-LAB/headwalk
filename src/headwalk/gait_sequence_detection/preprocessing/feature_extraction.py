import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
import pywt

LABEL_NAME_MAP = {
    0: "static",
    1: "walking",
    2: "ascending",
    3: "descending",
}

def compute_basic_stats(x, prefix):
    """Compute basic time-domain statistical features for one signal."""
    x = np.asarray(x)

    features = {}

    features[f"{prefix}_mean"] = np.mean(x)
    features[f"{prefix}_std"] = np.std(x)
    features[f"{prefix}_var"] = np.var(x)
    features[f"{prefix}_rms"] = np.sqrt(np.mean(x ** 2))
    features[f"{prefix}_max"] = np.max(x)
    features[f"{prefix}_min"] = np.min(x)
    features[f"{prefix}_median"] = np.median(x)
    features[f"{prefix}_mad"] = np.mean(np.abs(x - np.mean(x)))
    features[f"{prefix}_iqr"] = np.percentile(x, 75) - np.percentile(x, 25)
    features[f"{prefix}_abs_dev"] = np.mean(np.abs(x))
    features[f"{prefix}_skewness"] = skew(x)
    features[f"{prefix}_kurtosis"] = kurtosis(x)

    features[f"{prefix}_p10"] = np.percentile(x, 10)
    features[f"{prefix}_p25"] = np.percentile(x, 25)
    features[f"{prefix}_p50"] = np.percentile(x, 50)
    features[f"{prefix}_p70"] = np.percentile(x, 70)
    features[f"{prefix}_p90"] = np.percentile(x, 90)

    features[f"{prefix}_energy"] = np.sum(x ** 2)
    features[f"{prefix}_power"] = np.mean(x ** 2)

    hist, _ = np.histogram(x, bins=20, density=True)
    hist = hist + 1e-12
    features[f"{prefix}_entropy"] = entropy(hist)

    features[f"{prefix}_zero_crossing_rate"] = np.mean(np.diff(np.sign(x)) != 0)

    return features


def compute_frequency_features(x, fs, prefix):
    """Compute FFT and spectral-domain features for one signal."""
    x = np.asarray(x)
    features = {}

    freqs = rfftfreq(len(x), d=1 / fs)
    fft_values = np.abs(rfft(x))

    power_spectrum = fft_values ** 2

    if len(power_spectrum) > 1:
        dominant_idx = np.argmax(power_spectrum[1:]) + 1
        features[f"{prefix}_dominant_freq"] = freqs[dominant_idx]

        sorted_idx = np.argsort(power_spectrum[1:]) + 1
        features[f"{prefix}_second_dominant_freq"] = freqs[sorted_idx[-2]] if len(sorted_idx) > 1 else np.nan

        features[f"{prefix}_peak_spectral_coeff"] = np.max(fft_values[1:])
    else:
        features[f"{prefix}_dominant_freq"] = np.nan
        features[f"{prefix}_second_dominant_freq"] = np.nan
        features[f"{prefix}_peak_spectral_coeff"] = np.nan

    features[f"{prefix}_fft_energy"] = np.sum(power_spectrum)
    features[f"{prefix}_spectral_energy"] = np.sum(power_spectrum)
    features[f"{prefix}_spectral_power"] = np.mean(power_spectrum)
    features[f"{prefix}_sum_squared_spectral_coeff"] = np.sum(fft_values ** 2)

    ps = power_spectrum + 1e-12
    ps_norm = ps / np.sum(ps)
    features[f"{prefix}_spectral_entropy"] = entropy(ps_norm)

    lf_mask = (freqs >= 0.1) & (freqs < 3)
    hf_mask = freqs >= 3

    features[f"{prefix}_lf_spectral_coeff_sum"] = np.sum(power_spectrum[lf_mask])
    features[f"{prefix}_hf_spectral_coeff_sum"] = np.sum(power_spectrum[hf_mask])

    return features


def compute_wavelet_features(x, prefix, wavelet="db4", level=3):
    """Compute simple wavelet coefficient statistics."""
    x = np.asarray(x)
    features = {}

    coeffs = pywt.wavedec(x, wavelet, level=level)

    for i, c in enumerate(coeffs):
        features[f"{prefix}_wavelet_L{i}_mean"] = np.mean(c)
        features[f"{prefix}_wavelet_L{i}_std"] = np.std(c)
        features[f"{prefix}_wavelet_L{i}_energy"] = np.sum(c ** 2)
        features[f"{prefix}_wavelet_L{i}_sum"] = np.sum(c)

    features[f"{prefix}_wavelet_total_energy"] = np.sum([np.sum(c ** 2) for c in coeffs])
    features[f"{prefix}_wavelet_total_sum"] = np.sum([np.sum(c) for c in coeffs])

    return features


def compute_axis_correlations(window, channel_names):
    """Compute correlations between accelerometer or gyroscope axes."""
    features = {}

    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            xi = window[:, i]
            xj = window[:, j]

            corr = np.corrcoef(xi, xj)[0, 1]
            features[f"corr_{channel_names[i]}_{channel_names[j]}"] = corr

    return features


def compute_window_features(window, fs, channel_names):
    """
    Compute all features for a single window.

    Expected window shape:
    samples x channels

    Example channel_names:
    ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"]
    """
    features = {}

    for i, ch in enumerate(channel_names):
        x = window[:, i]

        features.update(compute_basic_stats(x, ch))
        features.update(compute_frequency_features(x, fs, ch))
        features.update(compute_wavelet_features(x, ch))

    acc = window[:, 0:3]
    gyr = window[:, 3:6]

    acc_norm = np.linalg.norm(acc, axis=1)
    gyr_norm = np.linalg.norm(gyr, axis=1)

    features.update(compute_basic_stats(acc_norm, "acc_norm"))
    features.update(compute_frequency_features(acc_norm, fs, "acc_norm"))

    features.update(compute_basic_stats(gyr_norm, "gyr_norm"))
    features.update(compute_frequency_features(gyr_norm, fs, "gyr_norm"))

    features["ap_rms_norm_rms_ratio"] = (
            np.sqrt(np.mean(window[:, 0] ** 2)) /
            (np.sqrt(np.mean(acc_norm ** 2)) + 1e-12)
    )

    features.update(compute_axis_correlations(acc, ["acc_AP", "acc_ML", "acc_VT"]))
    features.update(compute_axis_correlations(gyr, ["gyr_AP", "gyr_ML", "gyr_VT"]))

    features["movement_intensity"] = np.mean(acc_norm)
    features["energy_expenditure_proxy"] = np.sum(acc_norm ** 2)

    return features


def extract_features_from_dataset(dataset, fs, output_csv):
    """
    Extract features from a window-based dataset and save them to CSV.

    Expected dataset format:
    list of dictionaries, where each element contains at least:
    - "X": window signal, shape samples x channels
    - "label": window label

    Optional metadata:
    - "subject"
    - "task"
    - "trial"
    - "window_start"
    - "window_end"
    """

    channel_names = ["acc_AP", "acc_ML", "acc_VT", "gyr_AP", "gyr_ML", "gyr_VT"]

    rows = []

    for idx, item in enumerate(dataset):
        # Window signal: shape = samples x channels
        window = item["Signal"]

        row = compute_window_features(window, fs, channel_names)

        # Metadata extraction from INDIVI window structure
        row["label"] = item.get("WindowLabel", None)
        row["label_name"] = LABEL_NAME_MAP.get(item.get("WindowLabel", None), "unknown")

        row["subject"] = item.get("Subject", None)
        row["task"] = item.get("Task", None)
        row["trial"] = item.get("Trial", None)

        row["window_start"] = item.get("WindowStartIdx", None)
        row["window_end"] = item.get("WindowEndIdx", None)

        row["window_start_time"] = item.get("WindowStartTime", None)
        row["window_end_time"] = item.get("WindowEndTime", None)

        row["window_index"] = idx
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"[INFO] Feature dataset saved to: {output_csv}")
    print(f"[INFO] Shape: {df.shape}")

    return df