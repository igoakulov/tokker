#!/usr/bin/env python3
from __future__ import annotations

import sys

from tokker.cli import strings
from tokker.cli.config import config, ConfigError
from tokker.models.registry import ModelRegistry


def run_set_default_model(model: str) -> None:
    """Set the default model and persist it to the config."""
    registry = ModelRegistry()
    if not registry.is_model_supported(model):
        print(strings.MSG_INVALID_MODEL.format(model=model), file=sys.stderr)
        sys.exit(1)

    # Get model info for display
    available_models = registry.list_models()
    model_info = next((m for m in available_models if m["name"] == model), None)

    try:
        config.set_default_model(model)
    except (ConfigError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Display confirmation without tick/description; no blank lines
    if model_info:
        provider = model_info["provider"]
        print(strings.MSG_DEFAULT_SET_PROVIDER.format(model=model, provider=provider))
    else:
        print(strings.MSG_DEFAULT_SET.format(model=model))

    print(strings.MSG_CONFIG_SAVED_TO.format(path=config.config_file))
