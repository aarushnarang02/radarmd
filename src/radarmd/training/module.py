"""LightningModule for multi-label chest X-ray classification.

Loss is ``BCEWithLogitsLoss`` with an optional per-class ``pos_weight`` to
counter the heavy class imbalance. The headline metric is **mean AUROC across
the 14 classes** (macro), computed on the full validation/test set via
torchmetrics; per-class AUROC is logged too so we can see which pathologies lag.
"""

from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics.classification import MultilabelAUROC

from ..data.constants import NUM_CLASSES, PATHOLOGIES
from ..models.factory import count_parameters, create_model


class ChestXrayClassifier(LightningModule):
    """DenseNet-121 / ConvNeXt multi-label classifier with AUROC tracking."""

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 15,
        warmup_epochs: int = 1,
        scheduler: str = "cosine",
        drop_rate: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        # pos_weight is a runtime tensor, not a hyperparameter to serialize.
        self.save_hyperparameters(ignore=["pos_weight"])
        self.model = create_model(
            backbone=backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
        )
        # Always register a pos_weight buffer (ones == unweighted) so the loss's
        # state_dict shape is identical whether or not weighting is used. Without
        # this, a checkpoint trained with pos_weight fails to load into a module
        # built with pos_weight=None (missing/unexpected buffer key).
        if pos_weight is None:
            pos_weight = torch.ones(num_classes)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # One AUROC metric per split; macro-averaged over the 14 classes, plus
        # per-class values via average=None.
        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.val_auroc_per_class = MultilabelAUROC(num_labels=num_classes, average=None)
        self.test_auroc_per_class = MultilabelAUROC(num_labels=num_classes, average=None)

        self.num_classes = num_classes
        self._n_params = count_parameters(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_fit_start(self) -> None:
        # self.log() is disallowed here; write trainable-param count straight to
        # the logger's hyperparameters instead.
        if self.logger is not None:
            self.logger.log_hyperparams({"trainable_params": self._n_params})

    # --- shared step ---------------------------------------------------------
    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"), on_epoch=True, on_step=(stage == "train"))
        if stage != "train":
            probs = torch.sigmoid(logits)
            target = y.int()
            auroc = self.val_auroc if stage == "val" else self.test_auroc
            per_class = self.val_auroc_per_class if stage == "val" else self.test_auroc_per_class
            auroc.update(probs, target)
            per_class.update(probs, target)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    # --- epoch-end AUROC -----------------------------------------------------
    def _log_auroc(self, stage: str) -> None:
        macro = self.val_auroc if stage == "val" else self.test_auroc
        per_class = self.val_auroc_per_class if stage == "val" else self.test_auroc_per_class
        self.log(f"{stage}/auroc_mean", macro.compute(), prog_bar=True)
        values = per_class.compute()
        for name, v in zip(PATHOLOGIES, values, strict=True):
            self.log(f"{stage}/auroc/{name}", v)
        macro.reset()
        per_class.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_auroc("val")

    def on_test_epoch_end(self) -> None:
        self._log_auroc("test")

    # --- optim ---------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.scheduler != "cosine":
            return opt

        max_epochs = self.hparams.max_epochs
        warmup = min(self.hparams.warmup_epochs, max(max_epochs - 1, 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(max_epochs - warmup, 1)
        )
        if warmup <= 0:
            sched = cosine
        else:
            # Linear warmup for `warmup` epochs (start at 10% LR), then cosine
            # decay for the rest. Warmup stabilizes the pretrained backbone
            # before the LR reaches its peak.
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=warmup
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine], milestones=[warmup]
            )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
