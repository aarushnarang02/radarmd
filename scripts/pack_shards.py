"""Resize the full NIH dataset and pack it into tar shards for Colab.

Run this once (on Colab, after downloading the full dataset) to turn 112k raw
1024px PNGs into a handful of resized JPEG shards that live on Google Drive.
Training runs then copy the shards to the VM's local SSD and extract once.

Usage (on Colab):
    python scripts/pack_shards.py \
        --src /content/data --dest /content/drive/MyDrive/radarmd/shards \
        --size 320 --shard-size 4096
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from radarmd.data.dataset import _IMAGE_EXTS
from radarmd.data.shards import pack_shards


def _find_images(src: Path) -> list[Path]:
    paths: list[Path] = []
    for p in src.rglob("*"):
        if p.suffix.lower() in _IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="directory with the raw dataset images")
    ap.add_argument("--dest", required=True, help="output directory for .tar shards")
    ap.add_argument("--size", type=int, default=320, help="target square resolution")
    ap.add_argument("--shard-size", type=int, default=4096, help="images per shard")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    args = ap.parse_args()

    src = Path(args.src)
    images = _find_images(src)
    if not images:
        raise SystemExit(f"No images found under {src}")
    print(f"Found {len(images)} images. Packing to {args.size}px JPEG shards ...")

    t0 = time.time()
    shards = pack_shards(
        images,
        args.dest,
        size=args.size,
        shard_size=args.shard_size,
        quality=args.quality,
    )
    dt = time.time() - t0
    total_mb = sum(p.stat().st_size for p in shards) / 1e6
    print(f"Wrote {len(shards)} shards ({total_mb:.0f} MB) in {dt:.0f}s to {args.dest}")


if __name__ == "__main__":
    main()
