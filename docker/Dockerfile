# RadarMD serving image: FastAPI + ONNX Runtime, CPU-only, torch-free.
#
# The serving path deliberately avoids torch/MONAI (see serve/preprocess.py),
# which keeps this image ~400MB instead of ~2GB and makes Cloud Run cold starts
# fast. Build from the repo root:
#
#   uv run python scripts/export_onnx.py --checkpoint <ckpt> --out models/radarmd.onnx
#   docker build -f docker/Dockerfile -t radarmd .
#   docker run --rm -p 8080:8080 radarmd

FROM python:3.11-slim

# Runtime deps only — intentionally NOT `pip install .` (that would pull torch).
# pandas is needed because radarmd.data's package init loads label utilities.
RUN pip install --no-cache-dir \
    "onnxruntime>=1.17" \
    "fastapi>=0.110" \
    "uvicorn>=0.29" \
    "pillow>=10.2" \
    "numpy>=1.26" \
    "pandas>=2.1" \
    "python-multipart>=0.0.9" \
    "prometheus-fastapi-instrumentator>=7.0"

WORKDIR /app

# Source (import via PYTHONPATH; no packaging step needed in the container).
COPY src/ /app/src/
# The exported ONNX model (and optional thresholds.json) must exist locally:
# scripts/export_onnx.py writes models/radarmd.onnx.
COPY models/ /app/models/

ENV PYTHONPATH=/app/src \
    RADARMD_ONNX_PATH=/app/models/radarmd.onnx \
    RADARMD_IMAGE_SIZE=320 \
    PYTHONUNBUFFERED=1

# Non-root user (Cloud Run best practice).
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# Cloud Run injects PORT; default to 8080 for local runs.
CMD ["sh", "-c", "uvicorn radarmd.serve.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
