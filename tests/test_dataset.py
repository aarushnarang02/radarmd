"""Dataset and transform output shapes/dtypes."""

import torch

from radarmd.data.constants import NUM_CLASSES
from radarmd.data.dataset import ChestXrayDataset, image_key, index_images
from radarmd.data.labels import load_metadata


def test_index_images_finds_all(image_dir, metadata_csv):
    idx = index_images(image_dir)
    df = load_metadata(metadata_csv)
    # Index is keyed by stem, so compare stems.
    assert {image_key(n) for n in df["image"]}.issubset(set(idx.keys()))


def test_dataset_item_shapes(image_dir, metadata_csv):
    df = load_metadata(metadata_csv)
    idx = index_images(image_dir)
    ds = ChestXrayDataset(df, idx, image_size=128, train=True)
    img, label = ds[0]
    assert img.shape == (3, 128, 128)
    assert img.dtype == torch.float32
    assert label.shape == (NUM_CLASSES,)


def test_eval_transform_is_deterministic(image_dir, metadata_csv):
    df = load_metadata(metadata_csv)
    idx = index_images(image_dir)
    ds = ChestXrayDataset(df, idx, image_size=96, train=False)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_transform_normalization_reasonable(image_dir, metadata_csv):
    # After ImageNet normalization, values should be roughly centered, not [0,1].
    df = load_metadata(metadata_csv)
    idx = index_images(image_dir)
    ds = ChestXrayDataset(df, idx, image_size=64, train=False)
    img, _ = ds[0]
    assert img.min() < 0.0  # normalization pushed some values negative
