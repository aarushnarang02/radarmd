"""LightningModule: loss is finite, AUROC updates, pos_weight is honored."""

import torch

from radarmd.data.constants import NUM_CLASSES
from radarmd.training.module import ChestXrayClassifier


def _batch(n=4, size=64):
    x = torch.randn(n, 3, size, size)
    y = (torch.rand(n, NUM_CLASSES) > 0.5).float()
    return x, y


def test_training_step_returns_finite_loss():
    m = ChestXrayClassifier(backbone="densenet121", pretrained=False)
    loss = m.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_pos_weight_applied():
    pw = torch.ones(NUM_CLASSES) * 3.0
    m = ChestXrayClassifier(backbone="densenet121", pretrained=False, pos_weight=pw)
    assert m.criterion.pos_weight is not None
    assert torch.allclose(m.criterion.pos_weight, pw)


def test_validation_updates_auroc():
    m = ChestXrayClassifier(backbone="densenet121", pretrained=False)
    m.eval()
    # Two batches so both classes are present per label for a defined AUROC.
    m.validation_step(_batch(), 0)
    m.validation_step(_batch(), 1)
    score = m.val_auroc.compute()
    assert 0.0 <= float(score) <= 1.0


def test_configure_optimizers_cosine():
    m = ChestXrayClassifier(backbone="resnet50", pretrained=False, scheduler="cosine", max_epochs=5)
    cfg = m.configure_optimizers()
    assert "optimizer" in cfg and "lr_scheduler" in cfg


def test_warmup_then_cosine_lr_curve():
    # LR should ramp up during warmup, peak, then decay.
    lr, warmup, epochs = 1e-3, 2, 6
    m = ChestXrayClassifier(
        backbone="resnet50", pretrained=False, scheduler="cosine",
        lr=lr, warmup_epochs=warmup, max_epochs=epochs,
    )
    cfg = m.configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[warmup]          # warmup increases LR to the peak
    assert lrs[warmup] > lrs[-1]         # cosine decays afterwards
    assert abs(max(lrs) - lr) < 1e-9     # peak equals the configured LR


def test_zero_warmup_is_plain_cosine():
    m = ChestXrayClassifier(
        backbone="resnet50", pretrained=False, scheduler="cosine",
        warmup_epochs=0, max_epochs=4,
    )
    cfg = m.configure_optimizers()
    assert "lr_scheduler" in cfg
