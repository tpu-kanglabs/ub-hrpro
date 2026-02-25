# HR-Pro

Point-supervised Temporal Action Localization for Syllable localization.

This code is based on https://github.com/pipixin321/HR-Pro

## Usage

```bash
uv run main.py --cfg ub-moji --stage 1 --mode train
uv run main.py --cfg ub-moji --stage 1 --mode test
uv run main.py --cfg ub-moji --stage 2 --mode train
uv run main.py --cfg ub-moji --stage 2 --mode test
```

## Checkpoint

The checkpoint can be downloaded from [Hugging Face](https://huggingface.co/kanglabs/ub-hrpro/tree/main/hrpro). Place the downloaded `ckpt/` directory in the same directory level as this README file.

```
.
|-- cfgs/
|-- ckpt/
|-- dataset/
`-- eval/
```
