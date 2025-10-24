"""Lightweight fallback implementation of :mod:`pydantic_settings`.

This module provides a minimal subset of the interface used by FlowDex so
that the application can run in environments where the optional dependency
isn't available.  It loads environment variables (and optionally a ``.env``
file) before initialising the underlying Pydantic model, ensuring values are
cast according to the declared field types.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict


class SettingsConfigDict(dict):
    """Simple mapping used to describe configuration options."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class BaseSettings(BaseModel):
    """Minimal ``BaseSettings`` replacement built on :class:`BaseModel`."""

    model_config = ConfigDict(extra="ignore")

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        config = dict(getattr(type(self), "model_config", {}) or {})
        env_file = config.get("env_file")
        encoding = config.get("env_file_encoding")

        file_values: Dict[str, Any] = {}
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                file_values = {
                    key: value
                    for key, value in dotenv_values(env_path, encoding=encoding).items()
                    if value is not None
                }

        values: Dict[str, Any] = {}
        for field_name in type(self).model_fields:
            env_key = field_name.upper()
            if env_key in os.environ:
                values[field_name] = os.environ[env_key]
            elif env_key in file_values:
                values[field_name] = file_values[env_key]

        values.update(data)
        super().__init__(**values)


__all__ = ["BaseSettings", "SettingsConfigDict"]
