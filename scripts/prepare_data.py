"""Download NIH ChestX-ray14 metadata and (locally) the sample image set.

Local development uses the Kaggle "sample" dataset (~5.6k images, ~4GB), which
is enough to smoke-test the full pipeline on an M1 Mac. Full training happens on
Colab against the complete 112k-image dataset (see notebooks/colab_train.ipynb).

Usage:
    uv run python scripts/prepare_data.py --dest data/

Requires Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KEY env).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Kaggle dataset slugs.
SAMPLE_SLUG = "nih-chest-xrays/sample"  # ~5.6k images + metadata subset
FULL_SLUG = "nih-chest-xrays/data"  # full 112k dataset (used on Colab)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_tree(item, target)
        elif not target.exists():
            shutil.copy2(item, target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data", help="destination data directory")
    ap.add_argument(
        "--full",
        action="store_true",
        help="download the full 112k dataset instead of the sample (Colab only)",
    )
    args = ap.parse_args()

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "kagglehub not installed. Run: uv pip install -e '.[data]'"
        ) from exc

    slug = FULL_SLUG if args.full else SAMPLE_SLUG
    print(f"Downloading '{slug}' via kagglehub (this can take a while)...")
    cached = Path(kagglehub.dataset_download(slug))
    print(f"Downloaded to Kaggle cache: {cached}")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Materializing dataset into {dest.resolve()} ...")
    _copy_tree(cached, dest)

    csvs = list(dest.rglob("*.csv"))
    pngs = list(dest.rglob("*.png"))
    print(f"Done. {len(pngs)} images, {len(csvs)} CSV files under {dest}.")
    for c in csvs:
        print(f"  - {c.relative_to(dest)}")


if __name__ == "__main__":
    main()
