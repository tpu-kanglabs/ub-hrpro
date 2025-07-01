"""
Functions for splitting angle sequences into segments.
"""
import numpy as np


def create_segments(data, segment_length=16, stride=16):
    """
    Segment a 1D or 2D numpy array into overlapping windows.

    Args:
        data (np.ndarray): Input sequence of angles.
        segment_length (int): Number of frames per segment.
        stride (int): Step size between segments.

    Returns:
        np.ndarray: Array of shape (num_segments, segment_length, ...).
    """
    total_length = len(data)
    max_valid = ((total_length - segment_length) // stride + 1) * stride + segment_length - stride
    trimmed = data[:max_valid]
    segments = [trimmed[i:i + segment_length] for i in range(0, len(trimmed) - segment_length + 1, stride)]
    return np.array(segments)
