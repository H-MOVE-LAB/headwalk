"""
Utilities package for head-worn IMU data.
"""

from .rotations import align_imu_to_gravity

__all__ = [
    "align_imu_to_gravity"
]