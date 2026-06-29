# Point-Supervised Japanese Fingerspelling Localization via HR-Pro and Contrastive Learning

Official PyTorch implementation of our paper at **ICCV 2025** [1st Multimodal Sign Language Recognition (MSLR) Workshop](https://multimodal-sign-language-recognition.github.io/ICCV-2025/)

This repository presents a point-supervised temporal localization pipeline for Japanese fingerspelling. We enhance [HR-Pro](https://arxiv.org/abs/2308.12608) with three key components:

- A transformer-based encoder ([VideoMAE v2](https://arxiv.org/abs/2303.16727))
- SimCLR-style point-supervised contrastive learning (Point-Sup. CL)
- Joint angle features derived from MediaPipe Hands


## Directory Structure

```bash
.
├── feature_extraction # Feature extraction pipelines (Angular/I3D)
├── hrpro # HR-Pro-based temporal localization with point annotations
└── videomae # Point-supervised contrastive learning with VideoMAE v2
```

Each directory includes its own `README.md` with detailed instructions for setup and execution.

## Requirements

- [uv](https://github.com/astral-sh/uv) (or your preferred package manager)
- CUDA>=12.4
- OpenCV

## Dataset

We have released [**ub-MOJI**](https://huggingface.co/datasets/kanglabs/ub-MOJI), a Japanese fingerspelling video dataset with point-level annotations, available via Hugging Face.

## Results

Localization performance (mean Average Precision across tIoU thresholds) on the [**ub-MOJI**](https://huggingface.co/datasets/kanglabs/ub-MOJI) dataset:


| Model                                 | mAP@0.1–0.5 | mAP@0.3–0.7 | mAP@0.1–0.7 |
|---------------------------------------|------------:|------------:|------------:|
| I3D (RGB + Flow)                      | 57.6%       | 50.8%       | 52.9%       |
| I3D + Angular                         | 90.8%       | **80.4%**   | **84.0%**   |
| VideoMAE v2                           | 62.9%       | 56.5%       | 58.6%       |
| VideoMAE v2 + Point-Sup. CL           | **93.4%**   | 78.9%       | 83.6%       |
| VideoMAE v2 + Point-Sup. CL + Angular | 90.9%       | 79.6%       | 83.7%       |

## Checkpoints

Download our trained checkpoints from [Hugging Face](https://huggingface.co/kanglabs/ub-hrpro).

## Contributing

For questions or collaborations, feel free to open an Issue or Pull Request.

## License

This code is released under the [MIT License](./LICENSE).  
Please refer to the dataset repository for dataset-specific licensing terms.

## Authors

- [Ryota Murai](https://github.com/rmuraix) (Code Owner)
- Naoto Tsuta
- [Duk Shin](https://researchmap.jp/shinduk?lang=en)
- [Yousun Kang](https://researchmap.jp/yskang)

## Citation

```bibtex
@InProceedings{Murai_2025_ICCV,
    author    = {Murai, Ryota and Tsuta, Naoto and Shin, Duk and Kang, Yousun},
    title     = {Point-Supervised Japanese Fingerspelling Localization via HR-Pro and Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {4975-4982}
    doi       = {10.1109/ICCVW69036.2025.00516},
}
```
