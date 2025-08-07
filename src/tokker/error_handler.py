#!/usr/bin/env python3
"""
Centralized error handling for the tokker CLI used in __main__.py.
"""

import re
import sys
from typing import Iterable

from tokker import messages as msg


# ---- Internal helpers (pure, no side-effects) ----


def _arg_value(argv: list[str], *keys: str) -> str | None:
    """
    Best-effort extraction of a value for any of the provided keys from argv.

    Examples:
      _arg_value(["tok", "-m", "gemini-2.5-flash"], "-m", "--model") -> "gemini-2.5-flash"
      _arg_value(["tok", "--model=cl100k_base"], "-m", "--model")   -> "cl100k_base"
    """
    try:
        for i, a in enumerate(argv):
            if a in keys and i + 1 < len(argv):
                return argv[i + 1]
            # Support --key=value style
            for k in keys:
                if a.startswith(k + "="):
                    return a.split("=", 1)[1]
        return None
    except Exception:
        return None


def _looks_like_google_model(model: str | None) -> bool:
    if not model:
        return False
    return model.startswith("gemini-") or model.startswith("models/gemini-")


# ---- Printing helpers (side-effects: write to stderr) ----


def _print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        sys.stderr.write(f"{line}\n")


def _print_google_guidance() -> None:
    """
    Print Google auth guidance using centralized messages.
    """
    try:
        header = msg.MSG_GOOGLE_AUTH_HEADER
        guide_line = msg.MSG_GOOGLE_AUTH_GUIDE_LINE
        steps = list(getattr(msg, "MSG_GOOGLE_AUTH_STEPS", []))
        _print_lines([header, guide_line, *steps])
    except Exception:
        # As a last resort, avoid crashing while printing guidance
        pass


# ---- Public handler ----


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
            msg.MSG_UNKNOWN_OUTPUT_FORMAT_FMT.format(
                value=err_text.split(":", 1)[1].strip()
            )
            + "\n"
        )
        return 1

    # 1b) Missing dependencies via "No module named ..."
    if "no module named" in lower:
        if "tiktoken" in lower:
            sys.stderr.write(msg.MSG_DEP_HINT_FMT.format(package="tiktoken") + "\n")
            return 1
        if "transformers" in lower:
            sys.stderr.write(msg.MSG_DEP_HINT_FMT.format(package="transformers") + "\n")
            return 1
        if "google" in lower or "genai" in lower:
            sys.stderr.write(msg.MSG_DEP_HINT_FMT.format(package="google-genai") + "\n")
            return 1

    # 2) Likely unknown/invalid model (best-effort)
    model_arg = _arg_value(argv, "-m", "--model")
    if model_arg and re.match(r"^[A-Za-z0-9_.:/\-]+$", model_arg or ""):
        # If the error suggests an unknown model, be explicit
        if "not found" in lower or "unknown model" in lower or "invalid model" in lower:
            sys.stderr.write(msg.MSG_MODEL_NOT_FOUND_FMT.format(model=model_arg) + "\n")
            sys.stderr.write(msg.MSG_HINT_LIST_MODELS + "\n")
            return 1
        # Otherwise, offer a helpful hint unless it looks like a Google model
        if not _looks_like_google_model(model_arg):
            sys.stderr.write(msg.MSG_HINT_LIST_MODELS + "\n")

    # 3) Google guidance: model prefix or recognizable error markers
    if _looks_like_google_model(model_arg) or (
        "compute_tokens" in lower and "google" in lower
    ):
        _print_google_guidance()
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
        sys.stderr.write(msg.MSG_FILESYSTEM_ERROR_FMT.format(err=err_text) + "\n")
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
        sys.stderr.write(msg.MSG_CONFIG_ERROR_FMT.format(err=err_text) + "\n")
        return 1

    # 5) Fallback
    sys.stderr.write(msg.MSG_UNEXPECTED_ERROR_FMT.format(err=err_text) + "\n")
    return 1
