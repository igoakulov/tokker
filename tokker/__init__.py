#!/usr/bin/env python3
"""
Tokker - A CLI tool for token counting and analysis.

This package provides utilities for tokenizing text using OpenAI's tiktoken library,
with support for multiple tokenizers and output formats.
"""

__version__ = "0.1.0"
__author__ = "igoakulov"
__email__ = "your.email@example.com"
__description__ = "A CLI tool for token counting and analysis"

# Import main utilities for programmatic use
from .cli.utils import tokenize_text, count_tokens, count_words, count_characters

__all__ = ["tokenize_text", "count_tokens", "count_words", "count_characters"]
