"""
Base model interface for the Tokker plugin system.

All model implementations must inherit from BaseModel
to ensure a consistent interface across different model providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    This interface ensures consistent behavior across different
    model providers (OpenAI, HuggingFace, etc.).
    """

    @property
    @abstractmethod
    def library_name(self) -> str:
        """
        Return the provider identifier.

        Examples:
            - 'OpenAI' for tiktoken models
            - 'HuggingFace' for transformers models
            - 'Google' for future Google models
        """
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """
        Return list of model names this implementation supports.

        Examples:
            - ['o200k_base', 'cl100k_base'] for tiktoken
            - ['bert-base-uncased', 'gpt2'] for HuggingFace
        """
        pass

    @abstractmethod
    def tokenize(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Tokenize text and return standardized result.

        Args:
            text: Input text to tokenize
            model_name: Name of model to use

        Returns:
            Dictionary with standardized fields:
            - token_strings: List[str] - decoded token strings
            - token_ids: List[int] - token IDs
            - token_count: int - number of tokens
            - model: str - model name used
            - provider: str - provider identifier
        """
        pass

    @abstractmethod
    def validate_model(self, model_name: str) -> bool:
        """
        Validate if model is supported by this implementation.

        Args:
            model_name: Name to validate

        Returns:
            True if model is supported, False otherwise
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens without full tokenization.

        Args:
            text: Input text
            model_name: Name of model to use

        Returns:
            Number of tokens
        """
        pass
