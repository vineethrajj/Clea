"""Configuration loading with hardware-tuned defaults baked into config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Config(dict):
    """Dict with attribute-style access, nested."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh))
    # Optional cloud key can come from the environment so it never lands in git.
    env_key = os.environ.get("CLEA_CLOUD_API_KEY")
    if env_key:
        cfg["llm"]["cloud_api_key"] = env_key
    return cfg
