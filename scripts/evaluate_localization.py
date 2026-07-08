"""Score Grad-CAM localization against the 880 NIH ground-truth boxes.

Loads a trained checkpoint, runs Grad-CAM for each annotated pathology on the
boxed images, and reports pointing-game accuracy and IoU@{0.1,0.25,0.5}
localization accuracy per class and overall.

Usage:
    uv run python scripts/evaluate_localization.py \
        --checkpoint outputs/checkpoints/best.ckpt \
        --data-dir data --bbox-csv data/BBox_List_2017.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from radarmd.data.bboxes import load_bboxes, validate_bbox_labels
from radarmd.data.dataset import image_key, index_images
from radarmd.data.transforms import build_transforms
from radarmd.interpret.evaluate import evaluate_localization
from radarmd.interpret.gradcam import GradCAMExplainer
from radarmd.training.module import ChestXrayClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="trained Lightning checkpoint")
    ap.add_argument("--data-dir", required=True, help="directory with the boxed images")
    ap.add_argument("--bbox-csv", required=True, help="BBox_List_2017.csv")
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--cam-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/localization.csv")
    args = ap.parse_args()

    model = ChestXrayClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    explainer = GradCAMExplainer(model.model)

    bboxes = load_bboxes(args.bbox_csv)
    validate_bbox_labels(bboxes)
    index = index_images(args.data_dir)
    # Only score boxes whose image we actually have locally.
    bboxes = bboxes[bboxes["image"].map(lambda n: image_key(n) in index)].reset_index(drop=True)
    print(f"Scoring {len(bboxes)} boxes across {bboxes['label'].nunique()} pathologies.")

    transform = build_transforms(image_size=args.image_size, train=False)

    def load_image(name: str) -> torch.Tensor:
        path = index[image_key(name)]
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        return transform(arr)

    result = evaluate_localization(
        explainer, bboxes, load_image, cam_threshold=args.cam_threshold
    )
    frame = result.summary_frame()

    # Overall (box-weighted) rows.
    total = int(frame["n"].sum())
    overall = {"pathology": "OVERALL", "n": total}
    overall["pointing"] = float(
        np.average(frame["pointing"], weights=frame["n"])
    )
    for t in result.iou_thresholds:
        overall[f"iou@{t}"] = float(np.average(frame[f"iou@{t}"], weights=frame["n"]))

    print(frame.to_string(index=False))
    print("-" * 60)
    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in overall.items()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
