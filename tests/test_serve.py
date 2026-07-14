"""Serving stack: ONNX export/parity, predictor, overlay, and the FastAPI app."""

import io

import numpy as np
import pytest
import torch
from PIL import Image

# The serving stack is optional (`.[serve]`); skip the whole module if its deps
# (onnxruntime, fastapi) aren't installed rather than failing collection.
pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")

from radarmd.data.constants import NUM_CLASSES, PATHOLOGIES  # noqa: E402
from radarmd.models.factory import create_model  # noqa: E402
from radarmd.serve.onnx_export import export_onnx, quantize_onnx, verify_parity  # noqa: E402
from radarmd.serve.overlay import overlay_heatmap  # noqa: E402
from radarmd.serve.predictor import OnnxPredictor  # noqa: E402

IMG_SIZE = 64


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    """Export a small real model to ONNX once for the whole module."""
    torch.manual_seed(0)
    model = create_model("resnet50", pretrained=False)
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(model.eval(), path, image_size=IMG_SIZE)
    return model, str(path)


def _png_bytes(size=IMG_SIZE) -> bytes:
    arr = (np.random.default_rng(1).random((size, size)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, format="PNG")
    return buf.getvalue()


# --- ONNX export / parity ----------------------------------------------------
def test_export_creates_file(onnx_model):
    _, path = onnx_model
    import os

    assert os.path.exists(path)


def test_parity_within_tolerance(onnx_model):
    model, path = onnx_model
    max_diff = verify_parity(model, path, image_size=IMG_SIZE, atol=1e-3)
    assert max_diff <= 1e-3


def test_quantize_produces_usable_int8_model(onnx_model):
    import onnxruntime as ort

    _, path = onnx_model
    q = quantize_onnx(path)
    assert q.exists()
    # The quantized model must still run and emit a 14-vector of probabilities.
    sess = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(4).standard_normal((1, 3, IMG_SIZE, IMG_SIZE)).astype(np.float32)
    out = sess.run(["probs"], {"image": x})[0]
    assert out.shape == (1, NUM_CLASSES)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- predictor ---------------------------------------------------------------
def test_predict_probs_shape_and_range(onnx_model):
    _, path = onnx_model
    pred = OnnxPredictor(path, image_size=IMG_SIZE)
    img = Image.open(io.BytesIO(_png_bytes()))
    probs = pred.predict_probs(img)
    assert probs.shape == (NUM_CLASSES,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_predict_returns_sorted_findings(onnx_model):
    _, path = onnx_model
    pred = OnnxPredictor(path, image_size=IMG_SIZE)
    findings = pred.predict(Image.open(io.BytesIO(_png_bytes())))
    assert len(findings) == NUM_CLASSES
    probs = [f.probability for f in findings]
    assert probs == sorted(probs, reverse=True)
    assert {f.pathology for f in findings} == set(PATHOLOGIES)


def test_thresholds_control_flagging(onnx_model):
    _, path = onnx_model
    # threshold 0 -> everything flagged; threshold 1 -> nothing flagged
    all_on = OnnxPredictor(path, image_size=IMG_SIZE, thresholds={n: 0.0 for n in PATHOLOGIES})
    all_off = OnnxPredictor(path, image_size=IMG_SIZE, thresholds={n: 1.01 for n in PATHOLOGIES})
    img = Image.open(io.BytesIO(_png_bytes()))
    assert all(f.flagged for f in all_on.predict(img))
    assert not any(f.flagged for f in all_off.predict(img))


def test_thresholds_load_from_json(tmp_path, onnx_model):
    import json

    _, path = onnx_model
    tj = tmp_path / "thresholds.json"
    tj.write_text(json.dumps({"Mass": 0.0}))
    pred = OnnxPredictor(path, image_size=IMG_SIZE, thresholds=str(tj))
    findings = {f.pathology: f for f in pred.predict(Image.open(io.BytesIO(_png_bytes())))}
    assert findings["Mass"].flagged  # threshold 0 forces a flag


# --- overlay -----------------------------------------------------------------
def test_overlay_returns_rgb_matching_size():
    base = Image.new("L", (100, 80))
    cam = np.random.default_rng(2).random((16, 16)).astype(np.float32)
    out = overlay_heatmap(base, cam)
    assert out.mode == "RGB"
    assert out.size == (100, 80)


def test_overlay_flat_cam_does_not_crash():
    base = Image.new("L", (32, 32))
    out = overlay_heatmap(base, np.zeros((8, 8), dtype=np.float32))
    assert out.size == (32, 32)


# --- FastAPI app -------------------------------------------------------------
def test_health_endpoint(onnx_model):
    from fastapi.testclient import TestClient

    from radarmd.serve.app import create_app

    _, path = onnx_model
    client = TestClient(create_app(onnx_path=path, image_size=IMG_SIZE))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["num_classes"] == NUM_CLASSES
    assert body["model_present"] is True


def test_predict_endpoint(onnx_model):
    from fastapi.testclient import TestClient

    from radarmd.serve.app import create_app

    _, path = onnx_model
    client = TestClient(create_app(onnx_path=path, image_size=IMG_SIZE))
    r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["findings"]) == NUM_CLASSES
    assert "flagged" in body


def test_predict_rejects_non_image(onnx_model):
    from fastapi.testclient import TestClient

    from radarmd.serve.app import create_app

    _, path = onnx_model
    client = TestClient(create_app(onnx_path=path, image_size=IMG_SIZE))
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_health_when_model_missing():
    from fastapi.testclient import TestClient

    from radarmd.serve.app import create_app

    client = TestClient(create_app(onnx_path="/nonexistent/model.onnx"))
    assert client.get("/health").json()["model_present"] is False
    # /predict should surface a 503 when the model isn't present
    r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 503
