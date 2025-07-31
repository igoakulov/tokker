#!/usr/bin/env python3
"""
Tokker - a fast local-first CLI tool for tokenizing text with all the best models in one place

This package provides utilities for tokenizing text using OpenAI and HuggingFace models,
with support for multiple model providers and output formats.
"""

import re
from pathlib import Path

def _get_pyproject_field(field_name, fallback_value):
    """Get a field value from pyproject.toml"""
    try:
        # Get the path to pyproject.toml relative to this file
        current_dir = Path(__file__).parent
        pyproject_path = current_dir.parent / "pyproject.toml"

        if not pyproject_path.exists():
            return fallback_value

        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple regex to extract field from pyproject.toml
        pattern = rf'^{field_name}\s*=\s*["\']([^"\']+)["\']'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1)
        return fallback_value
    except Exception:
        # Fallback value if pyproject.toml can't be read
        return fallback_value

def _get_version():
    """Get version from pyproject.toml"""
    return _get_pyproject_field("version", "0.2.1")

def _get_description():
    """Get description from pyproject.toml"""
    return _get_pyproject_field("description", "Tokker is a fast local-first CLI tool for tokenizing text with all the best models in one place")

__version__ = _get_version()
__author__ = "igoakulov"
__email__ = "igoruphere@gmail.com"
__description__ = _get_description()

# Import main utilities for programmatic use
from .cli.utils import tokenize_text, count_tokens, count_words, count_characters

__all__ = ["tokenize_text", "count_tokens", "count_words", "count_characters"]
