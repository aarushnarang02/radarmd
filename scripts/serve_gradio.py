"""Launch the Gradio UI backed by the ONNX predictor.

Probabilities come from the ONNX model; pass a torch checkpoint to also render a
Grad-CAM overlay for the top finding.

Usage:
    uv run python scripts/serve_gradio.py --onnx models/radarmd.onnx
    uv run python scripts/serve_gradio.py --onnx models/radarmd.onnx \
        --checkpoint outputs/checkpoints/best.ckpt --thresholds outputs/evaluation/thresholds.json
"""

from __future__ import annotations

import argparse

from radarmd.serve.gradio_ui import build_ui
from radarmd.serve.predictor import OnnxPredictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default="models/radarmd.onnx")
    ap.add_argument("--thresholds", default=None, help="thresholds.json from scripts/evaluate.py")
    ap.add_argument("--checkpoint", default=None, help="torch checkpoint for Grad-CAM overlay")
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    predictor = OnnxPredictor(args.onnx, image_size=args.image_size, thresholds=args.thresholds)

    torch_model = None
    if args.checkpoint:
        from radarmd.training.module import ChestXrayClassifier

        torch_model = ChestXrayClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu")

    demo = build_ui(predictor, torch_model=torch_model, image_size=args.image_size)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
