#!/usr/bin/env python3
"""
Centralized error handling for the tokker CLI used in __main__.py.
"""

import re
import sys

from tokker import messages
from tokker.utils import (
    get_arg_value,
    is_google_model,
    get_installed_providers,
)


# ---- Public handler (keep main function first) ----


def handle_exception(e: Exception, argv: list[str]) -> int:
    """
    Central mapping from raw exceptions to friendly CLI messages.

    This function prints to stderr only, and returns a non-zero exit code (1).
    It intentionally does not re-raise or exit, leaving control to the caller.
    """
    err_text = str(e) if e is not None else ""
    lower = err_text.lower()

    # 1) Specific known errors
    # 1a) Unknown output format raised by the formatter (via ValueError with a stable prefix)
    if isinstance(e, ValueError) and err_text.startswith("Unknown output format: "):
        sys.stderr.write(
            messages.MSG_UNKNOWN_OUTPUT_FORMAT_FMT.format(
                value=err_text.split(":", 1)[1].strip()
            )
            + "\n"
        )
        return 1

    # Detect if this invocation is setting a default model (-D/--model-default)
    model_default_arg = get_arg_value(argv, "-D", "--model-default")
    if model_default_arg:
        # Standardized invalid model reporting with optional dependency hints
        _print_model_not_found(model_default_arg)
        return 1

    # 2) Likely unknown/invalid model (best-effort)
    model_arg = get_arg_value(argv, "-m", "--model")
    if not model_arg:
        # Fall back to configured default model for clearer messaging when -m/--model is not provided
        try:
            from tokker.cli.config import (
                config,
            )  # local import to avoid import cycles at module import time

            model_arg = config.get_default_model() or None
        except Exception:
            model_arg = None
    if model_arg and re.match(r"^[A-Za-z0-9_.:/\\-]+$", model_arg or ""):
        # Map missing dependency errors for a model to standardized output as well
        if "no module named" in lower:
            _print_model_not_found(model_arg)
            return 1
        # If the error suggests an unknown model, emit standardized message and conditional hints
        if "not found" in lower or "unknown model" in lower or "invalid model" in lower:
            _print_model_not_found(model_arg)
            return 1

    # 3) Google guidance: model prefix or recognizable error markers
    if is_google_model(model_arg) or ("compute_tokens" in lower and "google" in lower):
        _print_google_guidance(model_arg)
        return 1

    # 4) Config/storage hints (permissions, IO, JSON)
    if isinstance(e, (OSError, IOError)) or any(
        k in lower
        for k in [
            "permission denied",
            "read-only file system",
            "ioerror",
            "is a directory",
            "not a directory",
        ]
    ):
        sys.stderr.write(messages.MSG_FILESYSTEM_ERROR_FMT.format(err=err_text) + "\n")
        return 1

    if any(
        k in lower
        for k in [
            "jsondecodeerror",
            "expecting value",
            "invalid json",
            "unterminated string",
        ]
    ):
        sys.stderr.write(messages.MSG_CONFIG_ERROR_FMT.format(err=err_text) + "\n")
        return 1

    # 5) Fallback
    sys.stderr.write(messages.MSG_UNEXPECTED_ERROR_FMT.format(err=err_text) + "\n")
    return 1


# ---- Internal helpers (pure or side-effect-limited) ----


def _print_model_not_found(model: str) -> None:
    """
    Standardized output for invalid/unknown model:
    - Always prints MSG_DEFAULT_MODEL_UNSUPPORTED_FMT
    - If at least one provider is missing, print:
        MSG_DEP_HINT_HEADING
        MSG_DEP_HINT_ALL
        MSG_DEP_HINT_* for each missing provider
    """
    providers = sorted(get_installed_providers())
    providers_str = ", ".join(providers) if providers else "none"
    sys.stderr.write(
        messages.MSG_DEFAULT_MODEL_UNSUPPORTED_FMT.format(
            model=model, providers=providers_str
        )
        + "\n"
    )
    # Only append dependency hints if at least one provider is missing
    missing = {"OpenAI", "Google", "HuggingFace"} - set(providers)
    if missing:
        sys.stderr.write(messages.MSG_DEP_HINT_HEADING + "\n")
        sys.stderr.write(messages.MSG_DEP_HINT_ALL + "\n")
        # Keep a stable output order
        if "OpenAI" in missing:
            sys.stderr.write(messages.MSG_DEP_HINT_TIKTOKEN + "\n")
        if "Google" in missing:
            sys.stderr.write(messages.MSG_DEP_HINT_GOOGLE + "\n")
        if "HuggingFace" in missing:
            sys.stderr.write(messages.MSG_DEP_HINT_TRANSFORMERS + "\n")
    pass


# ---- Printing helpers (side-effects: write to stderr) ----


def _print_lines(lines) -> None:
    for line in lines:
        sys.stderr.write(f"{line}\n")


def _print_google_guidance(model: str | None) -> None:
    """
    Print Google auth guidance using centralized messages.

    If Google provider is not installed, prepend the standardized "model not found"
    block and a blank line before the guidance.
    """
    try:
        installed = get_installed_providers()
        if "Google" not in installed and model:
            _print_model_not_found(model)
            sys.stderr.write("\n")

        header = messages.MSG_GOOGLE_AUTH_HEADER
        guide_line = messages.MSG_GOOGLE_AUTH_GUIDE_URL
        _print_lines([header, guide_line])
    except Exception:
        # As a last resort, avoid crashing while printing guidance
        pass
