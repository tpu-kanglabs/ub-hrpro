"""
Wrapper for creating the MediaPipe hand detector.
"""

from mediapipe.tasks.python import BaseOptions, vision


def create_hand_detector(model_path: str, num_hands: int = 1):
    """
    Instantiate a MediaPipe HandLandmarker with given model.

    Args:
        model_path (str): Path to .task file.
        num_hands (int): Maximum number of hands to detect.

    Returns:
        HandLandmarker: Initialized detector instance.
    """
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path), num_hands=num_hands
    )
    return vision.HandLandmarker.create_from_options(options)
