# Required Files for Dataset Construction

This directory must contain a CSV file (`point_gaussian.csv`) containing annotation information to construct the dataset. The CSV file must follow the format shown below:

| class | class\_index | start\_frame | stop\_frame | video\_id  | point |
| ----- | ------------ | ------------ | ----------- | ---------- | ----- |
| ta    | 15           | 0            | 23          | filename   | 11    |
| ti    | 16           | 35           | 78          | filename   | 56    |
| tu    | 17           | 82           | 129         | filename   | 105   |
| te    | 18           | 144          | 171         | filename   | 157   |
| to    | 19           | 189          | 235         | filename   | 212   |
| a     | 0            | 112          | 180         | filename   | 146   |
| i     | 1            | 212          | 259         | filename   | 235   |
| u     | 2            | 274          | 303         | filename   | 288   |

* `class`: Class label (e.g., "ta", "i", etc.)
* `class_index`: Integer index corresponding to the class
* `start_frame`: Starting frame of the action segment
* `stop_frame`: Ending frame of the action segment
* `video_id`: Identifier of the video (e.g., filename without extension)
* `point`: **Central frame** of the action segment
