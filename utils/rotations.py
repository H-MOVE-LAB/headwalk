import numpy as np


def align_imu_to_gravity(
    acc: np.ndarray,
    gyr: np.ndarray,
    sampling_rate_hz: float,
    static_duration_s: float,
    gravity_ideal: np.ndarray = np.array([1.0, 0.0, 0.0]),
):
    """
    Reorient IMU accelerometer and gyroscope signals so that gravity is aligned
    with an ideal reference direction.

    The rotation is estimated from the mean acceleration measured during an
    initial static period, assuming the sensor is not moving and acceleration
    is dominated by gravity.

    Parameters
    ----------
    acc : ndarray, shape (N, 3)
        Accelerometer data in sensor frame [m/s^2 or g].
    gyr : ndarray, shape (N, 3)
        Gyroscope data in sensor frame [rad/s].
    sampling_rate_hz : float
        Sampling frequency of the signals in Hz.
    static_duration_s : float
        Duration (in seconds) of the initial static window used to estimate gravity.
    gravity_ideal : ndarray, shape (3,), optional
        Ideal gravity direction in the target reference frame.
        Default is [1, 0, 0], meaning gravity aligned with the x-axis.

    Returns
    -------
    acc_aligned : ndarray, shape (N, 3)
        Accelerometer data reoriented to the ideal gravity frame.
    gyr_aligned : ndarray, shape (N, 3)
        Gyroscope data reoriented to the ideal gravity frame.
    R : ndarray, shape (3, 3)
        Rotation matrix applied to the signals.

    # ------------------------------------------------------------
    # Example IMU data (replace with your real signals)
    # ------------------------------------------------------------
    sampling_rate_hz = 128.0          # Hz
    static_duration_s = 2.0            # first 2 seconds assumed static

    N = 1280  # total number of samples (10 s at 128 Hz)

    # Fake example signals
    # Accelerometer measures gravity mainly along sensor y-axis
    acc = np.zeros((N, 3))
    acc[:, 1] = 9.81                   # gravity along y (misaligned)

    # Small random noise
    acc += 0.05 * np.random.randn(N, 3)

    # Gyroscope (almost static)
    gyr = 0.01 * np.random.randn(N, 3)

    # ------------------------------------------------------------
    # Align IMU signals to gravity
    # ------------------------------------------------------------
    acc_aligned, gyr_aligned, R = align_imu_to_gravity(
        acc=acc,
        gyr=gyr,
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=static_duration_s,
        gravity_ideal=np.array([1.0, 0.0, 0.0])  # gravity along x-axis
    )
    """

    def normalize(v):
        return v / np.linalg.norm(v)

    # ------------------------------------------------------------------
    # 1. Estimate measured gravity from initial static window
    # ------------------------------------------------------------------
    n_static = int(static_duration_s * sampling_rate_hz)
    if n_static < 1:
        raise ValueError("Static duration too short for the given sampling rate.")

    g_measured = normalize(acc[:n_static].mean(axis=0))
    g_ideal = normalize(gravity_ideal)

    # ------------------------------------------------------------------
    # 2. Compute rotation axis and angle (angle-axis representation)
    # ------------------------------------------------------------------
    axis = np.cross(g_measured, g_ideal)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # Gravity already aligned
        R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(g_measured, g_ideal), -1.0, 1.0))

        # ------------------------------------------------------------------
        # 3. Rodrigues' rotation formula
        # ------------------------------------------------------------------
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])

        R = (
            np.eye(3)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K)
        )

    # ------------------------------------------------------------------
    # 4. Apply rotation to signals
    # ------------------------------------------------------------------
    acc_aligned = (R @ acc.T).T # A @ B is equivalent to np.matmul(A, B)
    gyr_aligned = (R @ gyr.T).T

    return acc_aligned, gyr_aligned, R

