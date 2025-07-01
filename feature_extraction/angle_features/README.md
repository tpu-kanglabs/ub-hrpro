# angle_features

Extract angles from a folder of videos

```bash
uv run main.py \
  --video_dir /path/to/videos \
  --model_path /path/to/hand_landmarker.task \
  --output_dir ./output \
  [--pool] [--segment_length 16] [--stride 16]

```

You can download `hand_landmarker.task` [at here](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
