# RadarMD — Chest X-ray Triage

Deep-learning triage system that detects and localizes **14 thoracic pathologies**
on chest X-rays from the NIH **ChestX-ray14** dataset (112k+ images, 30k patients).
It trains DenseNet-121 and ConvNeXt backbones with PyTorch Lightning, MONAI, and
timm; uses **Grad-CAM** to localize findings (validated against 880 ground-truth
boxes); and ships as an **ONNX** model behind a **FastAPI + Gradio** app on
**Cloud Run**.

> **Status:** under active construction. Stage 1 (data pipeline + tests) is done;
> see the roadmap below.

## The 14 pathologies

Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening,
Hernia. (Eight of these — the ones with ground-truth boxes — are used to validate
localization.)

## Why the design choices matter

- **Patient-level splits.** ~30k patients contribute multiple studies each.
  Splitting by image leaks anatomy across folds and inflates AUROC, so every
  split is by `patient_id`, enforced by a leakage test.
- **Two benchmarks.** We report the CheXNet-style random 70/10/20 patient split
  (headline, comparable to ~0.84 mean AUROC) **and** the harder official NIH
  split, for honesty.
- **Class imbalance.** Findings are rare and skewed; training uses
  `BCEWithLogits` with per-class `pos_weight`, and evaluation prioritizes
  sensitivity on serious findings over raw accuracy.

## Layout

```
src/radarmd/
  data/         label encoding, patient-level splits, bbox parsing, MONAI transforms, DataModule
  models/       backbone factory (stage 2)
  training/     LightningModule (stage 2)
  interpret/    Grad-CAM + localization metrics (stage 4)
  evaluation/   thresholds + reports (stage 5)
  serve/        ONNX runtime, FastAPI, Gradio, Prometheus (stages 6-7)
scripts/        prepare_data.py, inspect_batch.py, train.py, ...
notebooks/      colab_train.ipynb (full-scale training)
tests/          pytest suite (labels, splits, bboxes, transforms)
```

## Quickstart (local dev)

```bash
# Creates .venv and installs the project (editable) with dev + data extras
uv sync --extra dev --extra data

# Download the ~4GB sample dataset (needs Kaggle credentials)
uv run python scripts/prepare_data.py --dest data

# Sanity-check the pipeline: writes a labeled batch grid
uv run python scripts/inspect_batch.py --data-dir data --metadata data/Data_Entry_2017.csv

# Run the test suite
uv run pytest

# Smoke-train DenseNet-121 locally (CPU, 1 epoch) against a data dir
uv run python scripts/train.py --config configs/smoke.yaml \
    data.data_dir=data data.metadata_csv=data/Data_Entry_2017.csv
```

Full training runs on Colab Pro against the complete dataset; local dev uses the
sample so it fits on a laptop.

## Full-scale training (Colab)

The full dataset is 112k 1024px PNGs (~42GB) and Google Drive is slow with that
many small files. `scripts/pack_shards.py` resizes everything to 320px and packs
it into a handful of `.tar` shards (~4-5GB total) that live on Drive; each
training run copies the shards to the VM's local SSD and extracts once, then
reads normal files at full speed. `notebooks/colab_train.ipynb` drives the whole
flow (download → pack → extract → baseline → tuned run → `wandb` sweep). The
sweep space is defined in `configs/sweep.yaml`.

## Roadmap

1. ✅ Data pipeline: labels, patient-level splits, bbox parsing, transforms, DataModule, tests
2. ✅ Model factory (DenseNet-121 / ConvNeXt) + LightningModule (BCEWithLogits + AUROC) + `train.py`; local smoke run; W&B-ready
3. 🚧 Colab training path ready: full-dataset resize+shard packing, warmup+cosine LR, W&B sweep config, Colab notebook. **GPU runs pending** (needs Colab Pro) to hit ≥0.84 mean AUROC across 60+ experiments.
4. ✅ Grad-CAM localization + validation harness vs the 880 GT boxes (pointing game, IoU@{0.1,0.25,0.5}); `scripts/evaluate_localization.py`. Runs on real trained checkpoints.
5. ✅ Evaluation suite: per-class operating thresholds with a sensitivity floor on serious findings (FNR < 8%), plus AUROC/AP/sensitivity/specificity/F1 reports; `scripts/evaluate.py`.
6. ✅ ONNX export (+ INT8 dynamic quantization) with PyTorch/ORT parity checks, a CPU latency benchmark, an ONNX-Runtime `FastAPI` service (`/predict`, `/health`, Prometheus `/metrics`), and a Gradio UI with Grad-CAM overlays.
7. ⬜ Docker + Cloud Run + GitHub Actions CI/CD + Prometheus metrics

## Serving (Stage 6)

```bash
uv sync --extra dev --extra data --extra serve

# Export a trained checkpoint to ONNX (+ parity check)
uv run python scripts/export_onnx.py --checkpoint outputs/checkpoints/best.ckpt \
    --out models/radarmd.onnx --image-size 320

# CPU latency: PyTorch vs ONNX fp32 vs ONNX INT8
uv run python scripts/benchmark.py --onnx models/radarmd.onnx --backbone densenet121

# FastAPI service (POST an image to /predict, metrics at /metrics)
RADARMD_ONNX_PATH=models/radarmd.onnx uv run uvicorn radarmd.serve.app:app --port 8000

# Gradio UI (add --checkpoint for Grad-CAM overlays)
uv run python scripts/serve_gradio.py --onnx models/radarmd.onnx
```

**Note on the ONNX speedup.** The ~4x CPU speedup from ONNX (especially INT8) is
realized on x86 serving hardware (Cloud Run); the `benchmark.py` script measures
it on whatever machine it runs on. On Apple Silicon, PyTorch's ARM kernels are
already fast and ONNX may not beat them, so run the benchmark on the deployment
target to get the representative number. PyTorch/ORT output parity is verified to
< 1e-4 regardless.

> **Local dev note:** `uv run` can occasionally drop the editable install of the
> package. Tests are immune (pytest sets `pythonpath=["src"]`); for scripts, run
> with `PYTHONPATH=src` (or re-run `uv sync`) if you hit `ModuleNotFoundError`.

## Data & license

NIH ChestX-ray14 is released by the NIH Clinical Center for research use. This
project is for research and educational purposes and is **not a medical device**.
Code is MIT-licensed.
