import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def get_augmentations(image_size):
    # Two distinct augmentations tuned for sign language videos
    spatial1 = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),
            transforms.RandomRotation(degrees=5),
            transforms.RandomGrayscale(p=0.1),
        ]
    )
    spatial2 = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            ),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.6),
            transforms.RandomRotation(degrees=10),
            transforms.RandomSolarize(threshold=160, p=0.1),
        ]
    )
    return spatial1, spatial2


class VideoAugmentation:
    def __init__(self, size=(224, 224), spatial_transforms=None):
        self.size = size
        self.spatial = spatial_transforms or transforms.Compose([])

    def __call__(self, vid):
        vid = vid.permute(3, 0, 1, 2).float() / 255.0
        vid = F.interpolate(
            vid,
            scale_factor=float(self.size[0]) / min(vid.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        frames = vid.permute(1, 0, 2, 3)
        augmented = []
        for f in frames:
            img = transforms.ToPILImage()(f)
            img = self.spatial(img)
            augmented.append(transforms.ToTensor()(img))
        return torch.stack(augmented, dim=0).permute(1, 0, 2, 3)
