import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401


def build_model(args, clip_len):
    # load backbone
    model = create_model(
        args.model_name,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=clip_len,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    for key in ["model_state_dict", "model", "module"]:
        if key in ckpt:
            state = ckpt[key]
            break
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.cuda().train()
    # projector
    projector = nn.Sequential(
        nn.Linear(model.num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512, affine=False),
    ).cuda()
    # label-aware memory bank
    bank_size = 65536
    feat_dim = 512
    # initialize memory bank and labels
    memory_bank = F.normalize(torch.randn(bank_size, feat_dim, device="cpu"), dim=1)
    memory_labels = -torch.ones(bank_size, dtype=torch.long, device="cpu")
    # register buffers on model
    model.register_buffer("memory_bank", memory_bank)
    model.register_buffer("memory_labels", memory_labels)
    model.memory_ptr = 0
    return model, projector
