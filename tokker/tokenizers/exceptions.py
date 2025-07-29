"""
Tokenizer-specific exception classes.

This module defines custom exceptions for tokenizer operations
to provide clear error handling and debugging information.
"""


class TokenizerError(Exception):
    """Base exception for all tokenizer-related errors."""
    pass


class TokenizerNotFoundError(TokenizerError):
    """Raised when a requested tokenizer is not available."""
    pass


class TokenizerLoadError(TokenizerError):
    """Raised when a tokenizer fails to load or initialize."""
    pass


class TokenizerValidationError(TokenizerError):
    """Raised when tokenizer validation fails."""
    pass


class TokenizationError(TokenizerError):
    """Raised when tokenization operation fails."""
    pass


class UnsupportedTokenizerError(TokenizerError):
    """Raised when attempting to use an unsupported tokenizer."""
    pass


class MissingDependencyError(TokenizerError):
    """Raised when required dependencies are missing."""
    pass
