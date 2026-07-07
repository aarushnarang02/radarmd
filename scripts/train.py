"""Train a RadarMD chest X-ray classifier.

Loads a YAML config (inheriting configs/base.yaml), builds the DataModule and
LightningModule, and fits with PyTorch Lightning. Weights & Biases logging is
optional and controlled by the ``wandb.mode`` config field.

Usage:
    uv run python scripts/train.py --config configs/densenet121.yaml
    uv run python scripts/train.py --config configs/smoke.yaml \
        data.data_dir=/path/to/sample data.metadata_csv=/path/to/Data_Entry_2017.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from radarmd.data.datamodule import ChestXrayDataModule
from radarmd.models.factory import count_parameters
from radarmd.training.config import load_config
from radarmd.training.module import ChestXrayClassifier


def build_datamodule(cfg) -> ChestXrayDataModule:
    d = cfg.data
    return ChestXrayDataModule(
        data_dir=d["data_dir"],
        metadata_csv=d["metadata_csv"],
        image_size=d.get("image_size", 224),
        batch_size=d.get("batch_size", 32),
        num_workers=d.get("num_workers", 4),
        split=d.get("split", "random"),
        official_lists=d.get("official_lists"),
        seed=d.get("seed", 42),
    )


def build_logger(cfg, save_dir: str):
    mode = cfg.wandb.get("mode", "online")
    if mode == "disabled":
        return CSVLogger(save_dir=save_dir, name="radarmd")
    try:
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            project=cfg.wandb.get("project", "radarmd"),
            mode=mode,
            config=cfg.as_dict(),
        )
    except ImportError:
        print("wandb not installed; falling back to CSV logging. Install with '.[track]'.")
        return CSVLogger(save_dir=save_dir, name="radarmd")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a model YAML config")
    ap.add_argument("--base", default=None, help="base config (default: sibling base.yaml)")
    ap.add_argument("--out", default="outputs", help="output/checkpoint directory")
    ap.add_argument("--fast-dev-run", action="store_true", help="Lightning fast_dev_run")
    ap.add_argument("overrides", nargs="*", help="dotted key=value config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, base_path=args.base, overrides=args.overrides)
    L.seed_everything(cfg.data.get("seed", 42), workers=True)

    dm = build_datamodule(cfg)
    dm.prepare_data()
    dm.setup()
    print(
        f"Splits -> train {len(dm.datasets['train'])}, "
        f"val {len(dm.datasets['val'])}, test {len(dm.datasets['test'])}"
    )

    pos_weight = dm.pos_weight() if cfg.optim.get("use_pos_weight", True) else None

    model = ChestXrayClassifier(
        backbone=cfg.model.get("backbone", "densenet121"),
        pretrained=cfg.model.get("pretrained", True),
        lr=cfg.optim.get("lr", 1e-4),
        weight_decay=cfg.optim.get("weight_decay", 1e-5),
        max_epochs=cfg.optim.get("max_epochs", 15),
        warmup_epochs=cfg.optim.get("warmup_epochs", 1),
        scheduler=cfg.optim.get("scheduler", "cosine"),
        drop_rate=cfg.model.get("drop_rate", 0.0),
        pos_weight=pos_weight,
    )
    print(f"Model: {cfg.model.get('backbone')} ({count_parameters(model.model):,} params)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=out / "checkpoints",
        # Reference the real logged key ("val/auroc_mean"); Lightning turns the
        # slash into an underscore in the resulting filename.
        filename="epoch{epoch:02d}-auroc{val/auroc_mean:.4f}",
        auto_insert_metric_name=False,
        monitor="val/auroc_mean",
        mode="max",
        save_top_k=2,
    )

    trainer = L.Trainer(
        max_epochs=cfg.optim.get("max_epochs", 15),
        accelerator=cfg.trainer.get("accelerator", "auto"),
        precision=cfg.trainer.get("precision", "16-mixed"),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 5.0),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 20),
        logger=build_logger(cfg, str(out)),
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, datamodule=dm)
    if not args.fast_dev_run:
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
