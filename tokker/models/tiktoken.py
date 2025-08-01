"""
Tiktoken model implementation for the Tokker plugin system.

This module provides tiktoken model support through the unified
BaseModel interface.
"""

from typing import List, Dict, Any, TYPE_CHECKING

from .base import BaseModel
from .exceptions import ModelLoadError, UnsupportedModelError, MissingDependencyError

if TYPE_CHECKING:
    pass


class TiktokenModel(BaseModel):
    """
    Tiktoken model implementation.

    Supports OpenAI's tiktoken encodings including o200k_base and cl100k_base.
    """

    @property
    def library_name(self) -> str:
        """Return the provider identifier."""
        return "OpenAI"

    @property
    def supported_models(self) -> List[str]:
        """Return list of supported tiktoken encodings."""
        return [
            "o200k_base",
            "cl100k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base"
        ]

    def _get_encoding(self, model_name: str):
        """
        Get tiktoken encoding by name.

        Args:
            model_name: Name of the model

        Returns:
            tiktoken.Encoding instance

        Raises:
            ModelLoadError: If model cannot be loaded
            UnsupportedModelError: If model is not supported
        """
        if not self.validate_model(model_name):
            raise UnsupportedModelError(
                f"Model '{model_name}' is not supported. "
                f"Supported models: {', '.join(self.supported_models)}"
            )

        try:
            import tiktoken
            return tiktoken.get_encoding(model_name)
        except ImportError:
            raise MissingDependencyError(
                "tiktoken is required for tiktoken models. "
                "Install with: pip install tiktoken"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load tiktoken encoding '{model_name}': {e}")

    def tokenize(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Tokenize text using tiktoken.

        Args:
            text: Input text to tokenize
            model_name: Name of model to use

        Returns:
            Dictionary with standardized tokenization results
        """
        encoding = self._get_encoding(model_name)

        # Get token IDs
        token_ids = encoding.encode(text)

        # Get token strings by decoding each token individually
        token_strings = []
        for token_id in token_ids:
            try:
                token_str = encoding.decode([token_id])
                token_strings.append(token_str)
            except Exception:
                # Handle potential decoding errors
                token_strings.append(f"<token_{token_id}>")

        return {
            "token_strings": token_strings,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "model": model_name,
            "provider": self.library_name
        }

    def validate_model(self, model_name: str) -> bool:
        """
        Validate if model is supported.

        Args:
            model_name: Name to validate

        Returns:
            True if model is supported, False otherwise
        """
        return model_name in self.supported_models

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens without full tokenization.

        Args:
            text: Input text
            model_name: Name of model to use

        Returns:
            Number of tokens
        """
        encoding = self._get_encoding(model_name)
        return len(encoding.encode(text))
