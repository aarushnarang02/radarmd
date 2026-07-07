"""Resize the full dataset and pack it into tar shards for Colab.

The full NIH dataset is 112k PNGs spread across a dozen folders. Two problems on
Colab: (1) the raw 1024px images are ~42GB, far more than we need at 320px, and
(2) Google Drive chokes on 112k tiny files. The fix is to resize every image to a
fixed training resolution and pack them into a handful of ``.tar`` shards
(~4k images each). Shards live on Drive; a training run copies them to the VM's
local SSD and extracts once, then reads normal files at full speed.

Shards are plain tars of JPEGs (stdlib ``tarfile``) so no special runtime
dependency is needed to read them — extraction yields a flat image directory the
existing :class:`~radarmd.data.dataset.ChestXrayDataset` consumes directly.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable
from pathlib import Path

from PIL import Image


def _resize_encode(path: Path, size: int, quality: int) -> bytes:
    """Load an image, convert to grayscale, resize to ``size``, JPEG-encode."""
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pack_shards(
    image_paths: Iterable[Path],
    dest_dir: str | Path,
    size: int = 320,
    shard_size: int = 4096,
    quality: int = 90,
    prefix: str = "images",
) -> list[Path]:
    """Resize images and write them into ``.tar`` shards.

    Each image is stored in the tar under its **original filename but with a
    ``.jpg`` extension**, so downstream code must match on the stem. Returns the
    list of shard paths written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    shards: list[Path] = []
    current: tarfile.TarFile | None = None
    count = 0
    shard_idx = 0

    def _open_shard(idx: int) -> tarfile.TarFile:
        p = dest / f"{prefix}_{idx:04d}.tar"
        shards.append(p)
        return tarfile.open(p, "w")

    try:
        for path in image_paths:
            if current is None or count == shard_size:
                if current is not None:
                    current.close()
                current = _open_shard(shard_idx)
                shard_idx += 1
                count = 0
            data = _resize_encode(Path(path), size, quality)
            info = tarfile.TarInfo(name=Path(path).with_suffix(".jpg").name)
            info.size = len(data)
            current.addfile(info, io.BytesIO(data))
            count += 1
    finally:
        if current is not None:
            current.close()

    return shards


def extract_shards(shard_dir: str | Path, dest_dir: str | Path) -> int:
    """Extract every ``.tar`` shard in ``shard_dir`` into a flat ``dest_dir``.

    Returns the number of image files extracted. Intended to run once on the
    Colab VM's local disk before training.
    """
    shard_dir = Path(shard_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for shard in sorted(shard_dir.glob("*.tar")):
        with tarfile.open(shard, "r") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            # filter=data guards against path-traversal members (Py 3.12+).
            for m in members:
                _safe_extract_member(tf, m, dest)
            n += len(members)
    return n


def _safe_extract_member(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    # Flatten to basename and refuse anything that would escape dest.
    target = (dest / Path(member.name).name).resolve()
    if not str(target).startswith(str(dest.resolve())):
        raise ValueError(f"Unsafe tar member path: {member.name}")
    with tf.extractfile(member) as src, open(target, "wb") as out:
        out.write(src.read())
