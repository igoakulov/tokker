"""
Tokenizer plugin system for Tokker CLI.

This package provides a pluggable architecture for supporting multiple
tokenization libraries through a unified interface.
"""

from .base import BaseTokenizer
from .registry import (
    TokenizerRegistry,
    get_registry,
    list_tokenizers,
    tokenize,
    count_tokens,
    validate_tokenizer
)
from .exceptions import (
    TokenizerError,
    TokenizerNotFoundError,
    TokenizerLoadError,
    TokenizerValidationError,
    TokenizationError,
    UnsupportedTokenizerError,
    MissingDependencyError
)

__all__ = [
    # Base classes
    'BaseTokenizer',

    # Registry
    'TokenizerRegistry',
    'get_registry',

    # Convenience functions
    'list_tokenizers',
    'tokenize',
    'count_tokens',
    'validate_tokenizer',

    # Exceptions
    'TokenizerError',
    'TokenizerNotFoundError',
    'TokenizerLoadError',
    'TokenizerValidationError',
    'TokenizationError',
    'UnsupportedTokenizerError',
    'MissingDependencyError',
]
