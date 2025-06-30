"""Tiny helper to load & cache the YAML config.
Usage:
    from config_loader import CONFIG
"""
from __future__ import annotations
from pathlib import Path
import functools
import yaml


@functools.lru_cache(maxsize=1)
def _load(path: str | Path = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Configuration file '{cfg_path}' not found. You can copy 'config.yaml' from the repo root."
        )
    with cfg_path.open("r", encoding="utf‑8") as fh:
        return yaml.safe_load(fh)


#: Singleton‑style public accessor
CONFIG: dict = _load()