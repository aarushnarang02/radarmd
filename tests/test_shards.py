"""Shard packing/extraction round-trip and integration with the dataset."""

from pathlib import Path

from PIL import Image

from radarmd.data.dataset import ChestXrayDataset, index_images
from radarmd.data.labels import load_metadata
from radarmd.data.shards import extract_shards, pack_shards


def test_pack_creates_expected_shard_count(image_dir, tmp_path):
    paths = sorted(Path(image_dir).glob("*.png"))
    shards = pack_shards(paths, tmp_path / "shards", size=64, shard_size=10)
    expected = (len(paths) + 9) // 10
    assert len(shards) == expected
    assert all(p.exists() and p.suffix == ".tar" for p in shards)


def test_pack_extract_roundtrip_resizes(image_dir, tmp_path):
    paths = sorted(Path(image_dir).glob("*.png"))
    pack_shards(paths, tmp_path / "shards", size=48, shard_size=1000)
    n = extract_shards(tmp_path / "shards", tmp_path / "extracted")
    assert n == len(paths)
    jpgs = list((tmp_path / "extracted").glob("*.jpg"))
    assert len(jpgs) == len(paths)
    # Images were resized to 48x48 and are loadable.
    with Image.open(jpgs[0]) as im:
        assert im.size == (48, 48)


def test_extracted_shards_feed_dataset(image_dir, metadata_csv, tmp_path):
    # End-to-end: pack -> extract -> dataset resolves rows via stem matching.
    paths = sorted(Path(image_dir).glob("*.png"))
    pack_shards(paths, tmp_path / "shards", size=64, shard_size=1000)
    extract_shards(tmp_path / "shards", tmp_path / "extracted")

    df = load_metadata(metadata_csv)
    idx = index_images(tmp_path / "extracted")
    ds = ChestXrayDataset(df, idx, image_size=64, train=False)
    assert len(ds) == len(df)  # every row matched a JPEG by stem
    img, _ = ds[0]
    assert img.shape == (3, 64, 64)


def test_extract_rejects_path_traversal(tmp_path):
    import io
    import tarfile

    bad = tmp_path / "shards"
    bad.mkdir()
    with tarfile.open(bad / "images_0000.tar", "w") as tf:
        data = b"not really an image"
        info = tarfile.TarInfo(name="../escape.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    # Flattening to basename means the traversal name is neutralized, not written
    # outside dest; assert nothing landed outside the extraction dir.
    extract_shards(bad, tmp_path / "out")
    assert not (tmp_path / "escape.jpg").exists()
    assert (tmp_path / "out" / "escape.jpg").exists()
