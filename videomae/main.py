import argparse
import random

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import numpy as np
import torch
import wandb

from .train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--lr_model", type=float, default=1e-4, help="learning rate for backbone model"
    )
    parser.add_argument(
        "--lr_proj", type=float, default=1e-4, help="learning rate for projector"
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_list",
        type=str,
        required=True,
        help="path to txt file listing video names to skip",
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    wandb.init(project="ub_hrpro_pretrain", config=vars(args))
    train(args)


if __name__ == "__main__":
    main()
