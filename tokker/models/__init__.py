"""
Model plugin system for Tokker CLI.

This package provides a pluggable architecture for supporting multiple
model providers through a unified interface.
"""

from .base import BaseModel
from .registry import (
    ModelRegistry,
    get_registry,
    list_models,
    tokenize,
    count_tokens,
    validate_model
)
from .exceptions import (
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    ModelValidationError,
    TokenizationError,
    UnsupportedModelError,
    MissingDependencyError
)

__all__ = [
    # Base classes
    'BaseModel',

    # Registry
    'ModelRegistry',
    'get_registry',

    # Convenience functions
    'list_models',
    'tokenize',
    'count_tokens',
    'validate_model',

    # Exceptions
    'ModelError',
    'ModelNotFoundError',
    'ModelLoadError',
    'ModelValidationError',
    'TokenizationError',
    'UnsupportedModelError',
    'MissingDependencyError',
]
