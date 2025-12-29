"""Configuration for ms_utils."""

import json
import logging
from dataclasses import dataclass, field
from typing import Set


@dataclass
class Settings:
    """Global settings for ms_utils."""

    DEFAULT_NAMESPACE: str = "ms"
    REGISTRATION_CONFLICT_RESOLUTION: str = "raise"

    def __setattr__(self, name, value):
        if name == "DEFAULT_NAMESPACE":
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"DEFAULT_NAMESPACE must be a valid python identifier, got {value!r}")
        elif name == "REGISTRATION_CONFLICT_RESOLUTION":
            _allowed_resolutions = {"raise", "override", "ignore"}
            if value not in _allowed_resolutions:
                raise ValueError(
                    f"REGISTRATION_CONFLICT_RESOLUTION must be one of {_allowed_resolutions}, got {value!r}"
                )
        super().__setattr__(name, value)

    def load_from_dict(self, data: dict):
        """Update settings from a dictionary."""
        for key, value in data.items():
            if hasattr(self, key) and not key.startswith("_"):
                setattr(self, key, value)
            else:
                logging.warning(f"Unknown setting ignored: {key}")

    def load_from_json(self, json_str: str):
        """Update settings from a JSON string."""
        data = json.loads(json_str)
        self.load_from_dict(data)


settings = Settings()
