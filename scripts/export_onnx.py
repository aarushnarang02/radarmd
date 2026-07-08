"""Export a trained checkpoint to ONNX and verify PyTorch/ORT parity.

Usage:
    uv run python scripts/export_onnx.py \
        --checkpoint outputs/checkpoints/best.ckpt \
        --out models/radarmd.onnx --image-size 320
"""

from __future__ import annotations

import argparse
from pathlib import Path

from radarmd.serve.onnx_export import export_onnx, verify_parity
from radarmd.training.module import ChestXrayClassifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/radarmd.onnx")
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    model = ChestXrayClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    backbone = model.model.eval()

    path = export_onnx(backbone, args.out, image_size=args.image_size, opset=args.opset)
    print(f"Exported ONNX model to {Path(path).resolve()}")

    max_diff = verify_parity(backbone, path, image_size=args.image_size, atol=args.atol)
    print(f"Parity OK: max|PyTorch - ORT| = {max_diff:.2e} (tol {args.atol:.0e})")


if __name__ == "__main__":
    main()
