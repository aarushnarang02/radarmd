"""Image transform pipelines built on MONAI.

Chest X-rays are single-channel. We repeat to 3 channels so ImageNet-pretrained
backbones (DenseNet-121, ConvNeXt) can be used directly, and normalize with
ImageNet statistics for the same reason. Training adds mild geometric/intensity
augmentation; validation/test is deterministic.
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    Resize,
    ScaleIntensity,
    ToTensor,
)

# ImageNet channel statistics (after scaling to [0,1] and repeating to 3ch).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _repeat_to_3ch(x):
    """Repeat a single grayscale channel to 3 channels (C,H,W)."""
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def build_transforms(image_size: int = 224, train: bool = True) -> Compose:
    """Return a MONAI Compose for the given split.

    Input to the pipeline is a HxW (or HxWx1) numpy array in the raw PNG range;
    output is a normalized float32 tensor of shape (3, image_size, image_size).
    """
    common_head = [
        EnsureChannelFirst(channel_dim="no_channel"),  # HxW -> 1xHxW
        ScaleIntensity(minv=0.0, maxv=1.0),  # PNG range -> [0,1]
        Resize(spatial_size=(image_size, image_size)),
    ]

    if train:
        aug = [
            RandFlip(prob=0.5, spatial_axis=1),  # horizontal only; vertical flips are unphysical
            RandAffine(
                prob=0.5,
                rotate_range=0.087,  # ~5 degrees
                translate_range=(0.05 * image_size, 0.05 * image_size),
                scale_range=(0.05, 0.05),
                padding_mode="border",
            ),
            RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),
        ]
    else:
        aug = []

    tail = [
        _repeat_to_3ch,
        NormalizeIntensity(
            subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True
        ),
        ToTensor(track_meta=False),
    ]

    return Compose(common_head + aug + tail)
