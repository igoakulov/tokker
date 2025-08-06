#!/usr/bin/env python3
import sys

from tokker.cli.config import config, ConfigError
from tokker.cli.output.base_json import build_base_json
from tokker.cli.output.formats import format_and_print_output
from tokker.cli.google_errors import _handle_google_error
from tokker.models.registry import ModelRegistry


def run_tokenize(text: str, model: str | None, output_format: str) -> None:
    """Tokenize text with selected or default model, format, and print output."""
    selected_model = _select_model(model)
    registry = ModelRegistry()
    _validate_model_or_exit(registry, selected_model)

    try:
        tokenization_result = registry.tokenize(text, selected_model)
    except Exception as err:
        # Route Google-specific auth/setup guidance when appropriate
        _handle_google_error(registry, selected_model, err)
        raise

    try:
        # Track model usage in history and fetch delimiter
        config.add_model_to_history(selected_model)
        delimiter = config.get_delimiter()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build canonical base JSON and print in the desired format
    result = build_base_json(tokenization_result, text, delimiter)
    format_and_print_output(result, output_format, delimiter)


# ---- Command helpers (local to CLI) ----

def _select_model(model: str | None) -> str:
    """Choose the model from CLI arg or default configuration."""
    return model if model else config.get_default_model()


def _validate_model_or_exit(registry: ModelRegistry, model: str) -> None:
    """Validate the model via the registry or exit with error."""
    if not registry.is_model_supported(model):
        print(f"Invalid model: {model}.", file=sys.stderr)
        sys.exit(1)
