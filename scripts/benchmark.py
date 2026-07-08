"""Benchmark CPU inference latency: PyTorch eager vs ONNX Runtime.

Reports mean/median/p95 ms per image for both backends and the speedup, which
backs the "~4x faster CPU inference (~120 ms/image)" serving claim. Runs on a
checkpoint (exports a temp ONNX) or directly on an existing .onnx file.

Usage:
    uv run python scripts/benchmark.py --checkpoint outputs/checkpoints/best.ckpt
    uv run python scripts/benchmark.py --onnx models/radarmd.onnx --backbone densenet121
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from radarmd.models.factory import create_model
from radarmd.serve.onnx_export import _WithSigmoid, export_onnx, quantize_onnx


def _timeit(fn, x, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return times


def _summarize(name: str, times: list[float]) -> float:
    mean = statistics.mean(times)
    med = statistics.median(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    print(f"{name:16s} mean {mean:7.1f} ms | median {med:7.1f} ms | p95 {p95:7.1f} ms")
    return med


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None, help="existing ONNX model (skip export)")
    ap.add_argument("--backbone", default="densenet121", help="used when no checkpoint")
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1, help="torch/ORT CPU threads")
    ap.add_argument("--no-quantize", action="store_true", help="skip the INT8 model")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    # Build the torch model.
    if args.checkpoint:
        from radarmd.training.module import ChestXrayClassifier

        model = ChestXrayClassifier.load_from_checkpoint(args.checkpoint, map_location="cpu").model
    else:
        model = create_model(args.backbone, pretrained=False)
    torch_model = _WithSigmoid(model).eval()

    # Get an ONNX path (export a temp one if none given).
    onnx_path = args.onnx
    if onnx_path is None:
        onnx_path = "outputs/benchmark_model.onnx"
        export_onnx(model, onnx_path, image_size=args.image_size)
    print(f"ONNX model: {Path(onnx_path).resolve()}\n")

    import onnxruntime as ort

    def _make_session(path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    sess = _make_session(onnx_path)

    x = torch.randn(1, 3, args.image_size, args.image_size)
    x_np = x.numpy()

    def torch_fn(inp):
        with torch.no_grad():
            return torch_model(inp)

    def ort_fn(inp):
        return sess.run(["probs"], {"image": inp})[0]

    print(f"Benchmarking {args.runs} runs, batch=1, {args.threads} thread(s)\n")
    torch_med = _summarize("PyTorch (CPU)", _timeit(torch_fn, x, args.runs, args.warmup))
    ort_med = _summarize("ONNX fp32", _timeit(ort_fn, x_np, args.runs, args.warmup))

    int8_med = None
    if not args.no_quantize:
        int8_path = quantize_onnx(onnx_path)
        int8_sess = _make_session(int8_path)

        def int8_fn(inp):
            return int8_sess.run(["probs"], {"image": inp})[0]

        int8_med = _summarize("ONNX INT8", _timeit(int8_fn, x_np, args.runs, args.warmup))

    print(f"\nPyTorch -> ONNX fp32 speedup (median): {torch_med / ort_med:.2f}x")
    if int8_med:
        print(f"PyTorch -> ONNX INT8 speedup (median): {torch_med / int8_med:.2f}x   |   ~{int8_med:.0f} ms/image")

    # Sanity: fp32 ONNX must still match PyTorch closely.
    with torch.no_grad():
        diff = float(np.max(np.abs(torch_fn(x).numpy() - ort_fn(x_np))))
    print(f"Max output diff (PyTorch vs ONNX fp32): {diff:.2e}")


if __name__ == "__main__":
    main()
