# Required Files for Dataset Construction

This directory must contain the following three files to enable dataset construction and evaluation:

* `gt_full.json`: JSON file containing full ground truth annotations.
* `split_train.txt`: List of video IDs used for training.
* `split_test.txt`: List of video IDs used for testing.

These files are subject to future updates and improvements for better usability.

## `split_train.txt` / `split_test.txt`

Each line in the text files represents a single video ID. For example:

```
filename1
filename2
filename3
```

These files define the training and testing splits used to partition the dataset.

## `gt_full.json`

This JSON file stores annotation metadata for all videos. Its format follows the structure below:

```json
{
  "version": "fingerspeling",
  "database": {
    "filename1": {
      "subset": "Validation",
      "duration": 3.92,
      "fps": 60.0,
      "annotations": [
        {
          "label": "ta",
          "segment": [0.0, 0.4],
          "segment(frames)": [0, 23.5],
          "label_id": 15
        },
        ...
      ]
    },
    "filename2": {
      ...
    }
  }
}
```

Each entry under `database` corresponds to a video, identified by `video_id`. For each video, the following information is provided:

* `subset`: Indicates whether the video is used for training, validation, or testing.
* `duration`: Length of the video in seconds.
* `fps`: Frame rate of the video.
* `annotations`: A list of labeled action segments. Each annotation includes:

  * `label`: The class label (e.g., "ta", "i", etc.).
  * `segment`: Start and end times (in seconds).
  * `segment(frames)`: Start and end frame indices (can be fractional).
  * `label_id`: Integer ID corresponding to the label.

This annotation format allows frame-accurate supervision based on both temporal and frame-index information.
