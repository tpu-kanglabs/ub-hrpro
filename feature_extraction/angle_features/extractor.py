"""
Core video processing to extract per-frame hand angles.
"""

import cv2
import mediapipe as mp

from .utils import extract_hand_angles


def extract_angles_from_video(video_path: str, detector) -> list:
    """
    Process a video and extract joint angles for each frame.

    Args:
        video_path (str): Path to video file.
        detector: MediaPipe HandLandmarker instance.

    Returns:
        list: List of angle vectors per frame, or zeros if no hand detected.
    """
    cap = cv2.VideoCapture(video_path)
    frames_angles = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            angles = extract_hand_angles(result.hand_landmarks[0])
        else:
            angles = [0.0] * 20
        frames_angles.append(angles)
    cap.release()
    return frames_angles
