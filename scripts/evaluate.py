"""Evaluate a trained checkpoint: tune thresholds on val, report on test.

Produces per-class AUROC / AP / sensitivity / specificity / F1 at operating
points chosen with a sensitivity floor on serious findings, and checks the
"missed diagnoses under 8%" target.

Usage:
    uv run python scripts/evaluate.py \
        --checkpoint outputs/checkpoints/best.ckpt \
        --data-dir data --metadata data/Data_Entry_2017.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radarmd.data.datamodule import ChestXrayDataModule
from radarmd.evaluation.inference import collect_predictions
from radarmd.evaluation.metrics import per_class_report
from radarmd.evaluation.report import format_report
from radarmd.evaluation.thresholds import (
    DEFAULT_SENSITIVITY_FLOOR,
    select_operating_points,
    thresholds_vector,
)
from radarmd.training.module import ChestXrayClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--floor", type=float, default=DEFAULT_SENSITIVITY_FLOOR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/evaluation")
    args = ap.parse_args()

    model = ChestXrayClassifier.load_from_checkpoint(args.checkpoint, map_location=args.device)

    dm = ChestXrayDataModule(
        data_dir=args.data_dir,
        metadata_csv=args.metadata,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()

    print("Collecting validation predictions to tune thresholds ...")
    val_probs, val_labels = collect_predictions(model, dm.val_dataloader(), args.device)
    points = select_operating_points(val_probs, val_labels, floor=args.floor)
    thr = thresholds_vector(points)

    print("Collecting test predictions ...")
    test_probs, test_labels = collect_predictions(model, dm.test_dataloader(), args.device)
    report = per_class_report(test_probs, test_labels, thr)

    text = format_report(report, sensitivity_floor=args.floor, split_name="test")
    print("\n" + text)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "per_class_metrics.csv", index=False)
    (out / "report.md").write_text(text)
    with open(out / "thresholds.json", "w") as fh:
        json.dump(
            {name: round(float(p.threshold), 5) for name, p in points.items()}, fh, indent=2
        )
    np.save(out / "thresholds.npy", thr)
    print(f"\nWrote report and thresholds to {out.resolve()}")


if __name__ == "__main__":
    main()
