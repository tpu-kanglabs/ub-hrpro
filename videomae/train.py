import torch
import torch.nn.functional as F
import wandb
from dataset import (
    SnippetDataset,
    build_snippets_and_sampler,
    get_augmentations,
    load_annotations,
)
from models import build_model
from schedulefree import RAdamScheduleFree
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_loader(
    snippets, annotations, data_path, clip_len, aug1, aug2, batch_size, sampler
):
    dataset = SnippetDataset(snippets, data_path, annotations, clip_len, aug1, aug2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )


def update_memory_bank(model, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor):
    """
    Update model.memory_bank and memory_labels circular buffer with new features.
    """
    bs = z1.size(0)
    ptr = model.memory_ptr
    end = ptr + bs * 2

    new_feats = torch.cat([z1, z2], dim=0).cpu()
    new_lbls = torch.cat([labels, labels], dim=0).cpu()
    bank = model.memory_bank
    labels_bank = model.memory_labels
    bank_size = bank.size(0)

    if end <= bank_size:
        bank[ptr:end] = F.normalize(new_feats, dim=1)
        labels_bank[ptr:end] = new_lbls
    else:
        first = bank_size - ptr
        bank[ptr:] = F.normalize(new_feats[:first], dim=1)
        labels_bank[ptr:] = new_lbls[:first]
        tail = end % bank_size
        bank[:tail] = F.normalize(new_feats[first:], dim=1)
        labels_bank[:tail] = new_lbls[first:]

    model.memory_ptr = end % bank_size


def sample_bank_negatives(model, num_negs: int, device: torch.device):
    bank_size = model.memory_bank.size(0)
    neg_inds = torch.randint(0, bank_size, (num_negs,), device="cpu")
    feats = model.memory_bank[neg_inds].to(device)
    lbls = model.memory_labels[neg_inds].to(device)
    return F.normalize(feats, dim=1), lbls


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    bank_feats: torch.Tensor,
    bank_lbls: torch.Tensor,
    temperature: float,
):
    """
    Compute supervised contrastive loss combining batch and memory bank.
    """
    # Combine
    features = torch.cat([z1, z2, bank_feats], dim=0)
    lbls = torch.cat([labels, labels, bank_lbls], dim=0)
    features = F.normalize(features, dim=1)

    # similarity
    sim = torch.matmul(features, features.T) / temperature
    max_sim, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - max_sim.detach()

    # positive mask
    mask = (lbls.unsqueeze(1) == lbls.unsqueeze(0)).float().to(features.device)
    mask_no_self = mask - torch.eye(mask.size(0), device=mask.device)

    exp_logits = torch.exp(logits) * (1 - torch.eye(mask.size(0), device=mask.device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    pos_counts = mask_no_self.sum(1)
    safe_pos = pos_counts.clone().clamp(min=1)
    mean_log_prob = (mask_no_self * log_prob).sum(1) / safe_pos

    valid = pos_counts > 0
    if valid.any():
        return -mean_log_prob[valid].mean()
    return torch.tensor(0.0, device=features.device)


class SimCLRTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        annotations = load_annotations(args.ann_path)
        snippets, sampler = build_snippets_and_sampler(
            annotations, args.clip_len, args.skip_list
        )
        aug1, aug2 = get_augmentations((224, 224))
        self.loader = build_loader(
            snippets,
            annotations,
            args.data_path,
            args.clip_len,
            aug1,
            aug2,
            args.batch_size,
            sampler,
        )

        self.model, self.projector = build_model(args, args.clip_len)
        self.model.to(self.device)
        self.scaler = GradScaler()
        self.optimizer = RAdamScheduleFree(
            [
                {"params": self.model.parameters(), "lr": args.lr_model},
                {"params": self.projector.parameters(), "lr": args.lr_proj},
            ]
        )

    def feat_cp(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_()
        return checkpoint(self.model.forward_features, x, use_reentrant=False)

    def train_epoch(self, epoch: int):
        total_loss = 0.0
        self.optimizer.train()

        for v1, v2, labels in tqdm(self.loader, desc=f"Epoch {epoch}"):
            v1, v2, labels = (
                v1.to(self.device),
                v2.to(self.device),
                labels.to(self.device),
            )

            with autocast(device_type="cuda", dtype=torch.float16):
                h1 = self.feat_cp(v1)
                h2 = self.feat_cp(v2)
                z1, z2 = self.projector(h1), self.projector(h2)

                # update memory
                update_memory_bank(self.model, z1, z2, labels)

                # sample negatives
                bank_feats, bank_lbls = sample_bank_negatives(
                    self.model, min(self.model.memory_bank.size(0), 4096), self.device
                )

                # loss
                loss = contrastive_loss(
                    z1, z2, labels, bank_feats, bank_lbls, self.args.temperature
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.loader)

    def train(self):
        for epoch in range(self.args.epochs):
            avg_loss = self.train_epoch(epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated() / 1024**2
            wandb.log({"epoch": epoch, "loss": avg_loss, "lr": lr, "mem": mem})

            # checkpoint
            if (epoch + 1) % self.args.save_interval == 0 or (
                epoch + 1
            ) == self.args.epochs:
                ckpt = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "epoch": epoch + 1,
                    "config": vars(self.args),
                }
                path = f"simclr_t_epoch{epoch + 1}.pth"
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")


def train(args):
    """
    Entrypoint for training SimCLR with memory bank and supervised contrastive loss.
    """
    trainer = SimCLRTrainer(args)
    trainer.train()
