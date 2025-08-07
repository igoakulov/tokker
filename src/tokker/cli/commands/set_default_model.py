#!/usr/bin/env python3
from __future__ import annotations

from tokker import messages
from tokker.cli.config import config
from tokker.models.registry import ModelRegistry


def run_set_default_model(model: str) -> None:
    """Set the default model and persist it to the config."""
    registry = ModelRegistry()
    available_models = registry.list_models()
    model_info = next((m for m in available_models if m["name"] == model), None)

    # Persist selection (let any exceptions bubble to main)
    config.set_default_model(model)

    # Display confirmation without tick/description; no blank lines
    if model_info:
        provider = model_info["provider"]
        print(messages.MSG_DEFAULT_SET_PROVIDER.format(model=model, provider=provider))
    else:
        print(messages.MSG_DEFAULT_SET.format(model=model))

    print(messages.MSG_CONFIG_SAVED_TO.format(path=config.config_file))
