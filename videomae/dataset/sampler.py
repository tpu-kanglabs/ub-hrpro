import json
import math
import os

from torch.utils.data import WeightedRandomSampler


def load_annotations(ann_path):
    with open(ann_path, "r") as f:
        return json.load(f)["database"]


def _compute_weight(start, end, annotations):
    overlap = 0
    for ann in annotations:
        a0, a1 = ann["segment(frames)"]
        overlap += max(0, min(end, a1) - max(start, a0))
    return overlap / (end - start)


def build_snippets_and_sampler(annotations, clip_len, skip_list_path=None):
    # 1) skip_list をロード
    skip_videos = set()
    if skip_list_path is not None and os.path.isfile(skip_list_path):
        with open(skip_list_path, "r") as f:
            skip_videos = {line.strip() for line in f if line.strip()}

    snippets = []
    weights = []
    for video, info in annotations.items():
        if video in skip_videos:
            continue
        fps = info.get("fps", 30.0)
        duration = info.get("duration", 0)
        video_len = int(duration * fps)
        half = clip_len // 2

        for ann in info["annotations"]:
            start_f, end_f = ann["segment(frames)"]
            center = int((start_f + end_f) / 2)
            label = ann["label_id"]
            offsets = [-half // 2, 0, half // 2]
            for off in offsets:
                c = max(half, min(center + off, video_len - half))
                snippets.append((video, c, label))

                # weight 計算
                w0 = _compute_weight(c - half, c + half, info["annotations"])
                sigma = half / 2
                w = w0 * math.exp(-(off**2) / (2 * sigma**2))
                weights.append(w)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return snippets, sampler
