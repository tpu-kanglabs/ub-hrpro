"""
Utility functions for hand-angle computation.
"""

import math

import numpy as np


def calculate_angle(v1, v2):
    """
    Calculate the angle in degrees between two vectors using their dot product and magnitudes.

    Args:
        v1 (array-like): First vector.
        v2 (array-like): Second vector.

    Returns:
        float: Angle between v1 and v2 in degrees.

    Raises:
        ValueError: If either vector has zero magnitude.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Zero-length vector encountered when calculating angle.")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(np.arccos(cos_theta))


def get_joint_angle(landmarks, a, b, c):
    """
    Compute the 3D angle at landmark b formed by points a-b-c.

    Args:
        landmarks (list): List of landmark objects with x, y, z attributes.
        a (int): Index of first point.
        b (int): Index of vertex point.
        c (int): Index of third point.

    Returns:
        float: Angle in degrees.
    """
    p1, p2, p3 = landmarks[a], landmarks[b], landmarks[c]
    v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
    v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
    return calculate_angle(v1, v2)


def get_palm_orientation_angle(landmarks):
    """
    Compute the angle between the palm normal and the middle-finger direction.

    Defines the palm plane using wrist (0), index base (5), and pinky base (17).
    Returns the angle between the plane normal and the vector from wrist to middle finger base (9).
    """
    v1 = np.array(
        [
            landmarks[5].x - landmarks[0].x,
            landmarks[5].y - landmarks[0].y,
            landmarks[5].z - landmarks[0].z,
        ]
    )
    v2 = np.array(
        [
            landmarks[17].x - landmarks[0].x,
            landmarks[17].y - landmarks[0].y,
            landmarks[17].z - landmarks[0].z,
        ]
    )
    normal = np.cross(v1, v2)
    hand_dir = np.array(
        [
            landmarks[9].x - landmarks[0].x,
            landmarks[9].y - landmarks[0].y,
            landmarks[9].z - landmarks[0].z,
        ]
    )
    return calculate_angle(normal, hand_dir)


def extract_hand_angles(landmarks):
    """
    Extract 19 joint angles plus palm orientation from hand landmarks.

    Args:
        landmarks (list): List of 21 3D landmark points.

    Returns:
        list: A list of 20 angles in degrees.
    """
    triplets = [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (0, 5, 6),
        (5, 6, 7),
        (6, 7, 8),
        (0, 9, 10),
        (9, 10, 11),
        (10, 11, 12),
        (0, 13, 14),
        (13, 14, 15),
        (14, 15, 16),
        (0, 17, 18),
        (17, 18, 19),
        (18, 19, 20),
        (1, 0, 5),
        (5, 0, 17),
        (9, 0, 13),
        (1, 0, 17),
    ]
    angles = [get_joint_angle(landmarks, a, b, c) for a, b, c in triplets]
    angles.append(get_palm_orientation_angle(landmarks))
    return angles
