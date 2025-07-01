"""
Pooling and post-processing of segmented angle data.
"""

import numpy as np


def pool_segments(segments: np.ndarray) -> np.ndarray:
    """
    Apply average pooling to each segment of angles.

    Args:
        segments (np.ndarray): Array of segments to pool.

    Returns:
        np.ndarray: Pooled feature array.
    """
    return np.array([np.mean(seg, axis=0) for seg in segments])
