"""Gradio UI: upload an X-ray, see ranked findings and a Grad-CAM overlay.

Probabilities come from the fast ONNX predictor. If a torch model is supplied,
the UI also renders a Grad-CAM heatmap for the top finding to show *where* the
model is looking; without one it still works, just without the overlay.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .overlay import overlay_heatmap
from .predictor import Finding, OnnxPredictor


def _findings_table(findings: list[Finding]) -> list[list]:
    return [
        [f.pathology, round(f.probability, 4), "yes" if f.flagged else "", "critical" if f.critical else ""]
        for f in findings
    ]


def build_ui(predictor: OnnxPredictor, torch_model=None, image_size: int = 320):
    """Create (but don't launch) a Gradio Blocks app.

    ``torch_model`` is an optional :class:`ChestXrayClassifier` (or bare backbone)
    used only for the Grad-CAM overlay.
    """
    import gradio as gr

    explainer = None
    transform = None
    if torch_model is not None:
        from ..data.transforms import build_transforms
        from ..interpret.gradcam import GradCAMExplainer

        backbone = getattr(torch_model, "model", torch_model)
        explainer = GradCAMExplainer(backbone)
        transform = build_transforms(image_size=image_size, train=False)

    def infer(image: Image.Image):
        if image is None:
            return [], None
        findings = predictor.predict(image)
        table = _findings_table(findings)
        overlay = None
        if explainer is not None and findings:
            top = findings[0].pathology
            arr = np.asarray(image.convert("L"), dtype=np.float32)
            tensor = transform(arr)
            cam = explainer.heatmap(tensor, top)
            overlay = overlay_heatmap(image, cam)
        return table, overlay

    with gr.Blocks(title="RadarMD") as demo:
        gr.Markdown("# RadarMD — Chest X-ray Triage\nUpload a chest X-ray to see detected pathologies. **Research use only, not a medical device.**")
        with gr.Row():
            inp = gr.Image(type="pil", label="Chest X-ray")
            overlay_out = gr.Image(label="Grad-CAM (top finding)")
        table_out = gr.Dataframe(
            headers=["Pathology", "Probability", "Flagged", "Severity"],
            label="Findings (ranked)",
        )
        inp.change(infer, inputs=inp, outputs=[table_out, overlay_out])

    return demo
