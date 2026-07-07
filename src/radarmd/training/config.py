"""Experiment config loading: a base YAML plus dotted CLI overrides.

Configs are plain nested dicts (see ``configs/base.yaml``). ``train.py`` reads a
model-specific YAML that inherits from base, then applies ``key.subkey=value``
overrides from the command line so sweeps and one-off runs don't need new files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (returns ``base``)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(value: str) -> Any:
    """Turn a CLI string into an int/float/bool/None where it obviously is one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` strings to a nested config dict, in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


@dataclass
class ExperimentConfig:
    """Typed view over the config dict, with sensible defaults."""

    data: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    optim: dict = field(default_factory=dict)
    trainer: dict = field(default_factory=dict)
    wandb: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict) -> ExperimentConfig:
        known = {f: cfg.get(f, {}) for f in ("data", "model", "optim", "trainer", "wandb")}
        return cls(**known)

    def as_dict(self) -> dict:
        return {
            "data": self.data,
            "model": self.model,
            "optim": self.optim,
            "trainer": self.trainer,
            "wandb": self.wandb,
        }


def load_config(
    config_path: str | Path,
    base_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load ``config_path``, merging over ``base_path`` and applying overrides.

    If ``base_path`` is None, defaults to ``configs/base.yaml`` next to the
    given config. A model config need only specify what it changes.
    """
    config_path = Path(config_path)
    if base_path is None:
        candidate = config_path.parent / "base.yaml"
        base_path = candidate if candidate.exists() and candidate != config_path else None

    merged: dict = {}
    if base_path is not None:
        with open(base_path) as fh:
            merged = yaml.safe_load(fh) or {}
    with open(config_path) as fh:
        specific = yaml.safe_load(fh) or {}
    _deep_update(merged, specific)

    if overrides:
        apply_overrides(merged, overrides)

    return ExperimentConfig.from_dict(merged)
