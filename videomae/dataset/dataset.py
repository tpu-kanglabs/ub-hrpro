import io
import os
from functools import partial

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from .augmentations import VideoAugmentation

try:
    from petrel_client.client import Client

    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False


def _video_loader(video_path, client=None):
    if client is not None and "s3:" in video_path:
        video_path = io.BytesIO(client.get(video_path))

    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    return vr


def get_video_loader(
    use_petrel_backend: bool = True, enable_mc: bool = True, conf_path: str = None
):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    return partial(_video_loader, client=_client)


class SnippetDataset(Dataset):
    def __init__(
        self,
        snippets,
        data_path,
        annotations,
        clip_len,
        aug1,
        aug2,
    ):
        """
        snippets: list of tuples (video_name, center, label)
        data_path: directory containing video files
        annotations: dict mapping video_name to metadata
        clip_len: number of frames per clip
        aug1, aug2: spatial transform specs for two augmentations
        """

        # Filter out skipped videos
        self.snippets = snippets
        self.data_path = data_path
        self.annotations = annotations
        self.clip_len = clip_len
        self.aug1 = VideoAugmentation(size=(224, 224), spatial_transforms=aug1)
        self.aug2 = VideoAugmentation(size=(224, 224), spatial_transforms=aug2)
        self.loader_fn = get_video_loader(use_petrel_backend=False)

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        # Retrieve filtered snippet list
        video, center, label = self.snippets[idx]

        base = os.path.join(self.data_path, video)
        video_path = None
        for ext in [".mp4"]:
            p = base + ext
            if os.path.isfile(p):
                video_path = p
                break
        if video_path is None:
            raise FileNotFoundError(f"No video found for {video}")

        vr = self.loader_fn(video_path)
        info = self.annotations[video]
        video_len = int(info["duration"] * info.get("fps", 30.0))
        half = self.clip_len // 2

        starts = center - half
        ends = center + half
        pad_pre = max(0, -starts)
        pad_post = max(0, ends - video_len)
        starts = max(0, starts)
        ends = min(video_len, ends)

        frames = np.arange(starts, ends)
        data = vr.get_batch(frames).asnumpy()
        vid = torch.from_numpy(data)

        if pad_pre > 0 or pad_post > 0:
            pad_frames = []
            if pad_pre > 0:
                first = vid[0:1].repeat(pad_pre, 1, 1, 1)
                pad_frames.append(first)
            pad_frames.append(vid)
            if pad_post > 0:
                last = vid[-1:].repeat(pad_post, 1, 1, 1)
                pad_frames.append(last)
            vid = torch.cat(pad_frames, dim=0)

        assert vid.shape[0] == self.clip_len, f"Vid len {vid.shape[0]}!=clip_len"
        v1 = self.aug1(vid)
        v2 = self.aug2(vid)
        return v1, v2, torch.tensor(label, dtype=torch.long)
