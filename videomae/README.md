# Point-Supervised Contrastive Learning

## Usage

### 1. Training

To train the model, run:

```bash
uv run main.py \
    --data_path "path/to/video" \
    --ann_path "path/to/gt_full.json" \
    --ckpt "vit_b_k710_dl_from_giant.pth" \
    --skip_list "path/to/split_test.txt" \
    --epochs 30
```
The `vit_b_k710_dl_from_giant.pth` checkpoint is provided by [VideoMAE v2](https://github.com/OpenGVLab/VideoMAEv2) and distilled from a ViT-G model trained on Kinetics-710. Please download it [here](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/distill).

### 2. Feature Extraction

To extract temporal features for downstream tasks, use:

```bash
uv run extract_tad_feature.py \
    --data_path "path/to/video" \
    --ckpt_path "path/to/checkpoint"
```
