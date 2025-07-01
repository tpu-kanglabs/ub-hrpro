"""
Command-line interface for extracting and saving hand-angle features.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .detector import create_hand_detector
from .extractor import extract_angles_from_video
from .processor import pool_segments
from .segment import create_segments


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand joint angles from videos and save as .npy features"
    )
    parser.add_argument("--model_path", type=str, default="hand_landmarker.task",
                        help="Path to the MediaPipe hand-landmarker model file")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing input videos (.mp4)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for saving output .npy files")
    parser.add_argument("--pool", action="store_true",
                        help="Apply average pooling over segments")
    parser.add_argument("--segment_length", type=int, default=16,
                        help="Number of frames per segment")
    parser.add_argument("--stride", type=int, default=16,
                        help="Frame stride between segments")
    args = parser.parse_args()

    detector = create_hand_detector(args.model_path)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() == ".mp4"]
    for video in tqdm(video_files, desc="Processing videos"):
        angles = extract_angles_from_video(str(video), detector)
        segments = create_segments(angles, segment_length=args.segment_length, stride=args.stride)
        features = pool_segments(segments) if args.pool else np.array(segments)
        print(f"Feature shape for {video.name}: {features.shape}")
        np.save(output_dir / f"{video.stem}.npy", features)


if __name__ == "__main__":
    main()