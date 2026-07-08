"""FastAPI service exposing chest X-ray triage over HTTP.

Endpoints:
  - ``GET  /health``   liveness + model metadata
  - ``POST /predict``  multipart image upload -> per-pathology probabilities and
                       which findings are flagged at their operating threshold
  - ``GET  /metrics``  Prometheus metrics (via prometheus-fastapi-instrumentator)

The predictor is ONNX-Runtime backed for fast CPU inference. Configuration comes
from environment variables so the same image runs locally and on Cloud Run:
  RADARMD_ONNX_PATH   (default: models/radarmd.onnx)
  RADARMD_THRESHOLDS  (optional path to thresholds.json)
  RADARMD_IMAGE_SIZE  (default: 320)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..data.constants import NUM_CLASSES, PATHOLOGIES
from .predictor import OnnxPredictor


def create_app(
    onnx_path: str | Path | None = None,
    thresholds: str | Path | dict | None = None,
    image_size: int | None = None,
) -> FastAPI:
    """Build the FastAPI app. Args override the environment defaults (for tests)."""
    onnx_path = onnx_path or os.environ.get("RADARMD_ONNX_PATH", "models/radarmd.onnx")
    thresholds = thresholds or os.environ.get("RADARMD_THRESHOLDS") or None
    image_size = image_size or int(os.environ.get("RADARMD_IMAGE_SIZE", "320"))

    app = FastAPI(title="RadarMD", version="0.1.0", description="Chest X-ray triage")
    app.state.predictor = None
    app.state.config = {"onnx_path": str(onnx_path), "image_size": image_size, "thresholds": thresholds}

    def _get_predictor() -> OnnxPredictor:
        # Lazy-load so the app imports even before a model exists (health checks,
        # tests that don't hit /predict).
        if app.state.predictor is None:
            cfg = app.state.config
            if not Path(cfg["onnx_path"]).exists():
                raise HTTPException(503, f"Model not available at {cfg['onnx_path']}")
            app.state.predictor = OnnxPredictor(
                cfg["onnx_path"], image_size=cfg["image_size"], thresholds=cfg["thresholds"]
            )
        return app.state.predictor

    @app.get("/health")
    def health() -> dict:
        cfg = app.state.config
        return {
            "status": "ok",
            "model_present": Path(cfg["onnx_path"]).exists(),
            "num_classes": NUM_CLASSES,
            "pathologies": list(PATHOLOGIES),
            "image_size": cfg["image_size"],
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(400, "Uploaded file is not a valid image") from exc

        predictor = _get_predictor()
        findings = predictor.predict(image)
        return {
            "findings": [
                {
                    "pathology": f.pathology,
                    "probability": round(f.probability, 5),
                    "flagged": f.flagged,
                    "critical": f.critical,
                }
                for f in findings
            ],
            "flagged": [f.pathology for f in findings if f.flagged],
        }

    _instrument(app)
    return app


def _instrument(app: FastAPI) -> None:
    """Attach Prometheus metrics at /metrics if the instrumentator is present."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    except ImportError:  # pragma: no cover - optional in minimal installs
        pass


# Module-level app for `uvicorn radarmd.serve.app:app`.
app = create_app()
