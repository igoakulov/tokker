"""
Model-specific exception classes.

This module defines custom exceptions for model operations
to provide clear error handling and debugging information.
"""


class ModelError(Exception):
    """Base exception for all model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not available."""
    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load or initialize."""
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass


class TokenizationError(ModelError):
    """Raised when tokenization operation fails."""
    pass


class UnsupportedModelError(ModelError):
    """Raised when attempting to use an unsupported model."""
    pass


class MissingDependencyError(ModelError):
    """Raised when required dependencies are missing."""
    pass
