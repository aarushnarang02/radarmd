"""Config loading, inheritance from base, and dotted CLI overrides."""

from radarmd.training.config import apply_overrides, load_config


def test_load_inherits_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  backbone: densenet121\n  pretrained: true\noptim:\n  lr: 0.0001\n")
    child = tmp_path / "child.yaml"
    child.write_text("model:\n  backbone: convnext_tiny\n")

    cfg = load_config(child, base_path=base)
    # child overrides backbone but inherits pretrained and lr from base
    assert cfg.model["backbone"] == "convnext_tiny"
    assert cfg.model["pretrained"] is True
    assert cfg.optim["lr"] == 0.0001


def test_apply_overrides_coercion():
    cfg = {"optim": {"lr": 0.1}, "model": {}}
    apply_overrides(cfg, ["optim.lr=0.005", "optim.max_epochs=3", "model.pretrained=false"])
    assert cfg["optim"]["lr"] == 0.005
    assert cfg["optim"]["max_epochs"] == 3
    assert isinstance(cfg["optim"]["max_epochs"], int)
    assert cfg["model"]["pretrained"] is False


def test_override_creates_nested_key():
    cfg = {}
    apply_overrides(cfg, ["data.image_size=256"])
    assert cfg["data"]["image_size"] == 256
