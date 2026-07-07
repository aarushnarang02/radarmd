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
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,data]"

# Download the ~4GB sample dataset (needs Kaggle credentials)
uv run python scripts/prepare_data.py --dest data

# Sanity-check the pipeline: writes a labeled batch grid
uv run python scripts/inspect_batch.py --data-dir data --metadata data/sample_labels.csv

# Run the test suite
uv run pytest
```

Full training runs on Colab Pro against the complete dataset; local dev uses the
sample so it fits on a laptop.

## Roadmap

1. ✅ Data pipeline: labels, patient-level splits, bbox parsing, transforms, DataModule, tests
2. ⬜ Model factory + LightningModule + `train.py`; local smoke run; W&B
3. ⬜ Full Colab training (DenseNet-121 → ConvNeXt), ≥0.84 mean AUROC, 60+ W&B experiments
4. ⬜ Grad-CAM localization validated vs 880 GT boxes (IoU@0.5, pointing game)
5. ⬜ Evaluation suite: operating thresholds with a sensitivity floor on critical findings
6. ⬜ ONNX export + parity/latency benchmark; FastAPI + Gradio app
7. ⬜ Docker + Cloud Run + GitHub Actions CI/CD + Prometheus metrics

## Data & license

NIH ChestX-ray14 is released by the NIH Clinical Center for research use. This
project is for research and educational purposes and is **not a medical device**.
Code is MIT-licensed.
