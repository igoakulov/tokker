"""
HuggingFace model implementation for the Tokker plugin system.

This module provides HuggingFace transformers model support through the unified
BaseModel interface using AutoTokenizer.from_pretrained().
"""

from typing import List, Dict, Any
import logging

from .base import BaseModel
from .exceptions import ModelLoadError, MissingDependencyError

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseModel):
    """
    HuggingFace model implementation.

    Supports any HuggingFace model that can be loaded with AutoTokenizer.from_pretrained().
    Prioritizes fast tokenizers when available.
    """

    def __init__(self):
        """Initialize the HuggingFace model."""
        # Cache for loaded models to avoid repeated downloads
        self._model_cache: Dict[str, Any] = {}

    @property
    def library_name(self) -> str:
        """Return the provider identifier."""
        return "HuggingFace"

    @property
    def supported_models(self) -> List[str]:
        """
        Return empty list since HuggingFace uses BYOM (Bring Your Own Model).

        Any model from HuggingFace Hub can potentially be used.
        """
        return []

    def _get_model(self, model_name: str):
        """
        Load HuggingFace model with caching and fast tokenizer preference.

        Args:
            model_name: HuggingFace model name or path

        Returns:
            Loaded model instance

        Raises:
            MissingDependencyError: If transformers is not available
            ModelLoadError: If model cannot be loaded
        """
        # Check cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            raise MissingDependencyError(
                "transformers is required for HuggingFace models. "
                "Install with: pip install transformers"
            )

        try:
            logger.debug(f"Loading HuggingFace model: {model_name}")

            # Try to load with fast tokenizer first (as per PRD requirement)
            model = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False  # Security consideration
            )

            # Verify it's actually a fast model tokenizer
            if not model.is_fast:
                logger.warning(f"Fast tokenizer not available for {model_name}, using slow tokenizer")
                # Reload without fast requirement as fallback
                model = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=False
                )

            # Cache the model
            self._model_cache[model_name] = model
            logger.debug(f"Successfully loaded model: {model_name} (fast: {model.is_fast})")

            return model

        except Exception as e:
            error_msg = f"Failed to load HuggingFace model '{model_name}': {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)

    def tokenize(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Tokenize text using HuggingFace model.

        Args:
            text: Input text to tokenize
            model_name: HuggingFace model name

        Returns:
            Dictionary with standardized tokenization results
        """
        model = self._get_model(model_name)

        try:
            # Encode the text to get token IDs
            encoding = model(text, return_tensors=None, add_special_tokens=True)
            token_ids = encoding['input_ids']

            # Get token strings by converting each ID back to text
            token_strings = []
            for token_id in token_ids:
                try:
                    # Convert single token ID to string
                    token_str = model.decode([token_id], skip_special_tokens=False)
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

        except Exception as e:
            raise ModelLoadError(f"Failed to tokenize text with '{model_name}': {e}")

    def validate_model(self, model_name: str) -> bool:
        """
        Validate if model can be loaded.

        For HuggingFace BYOM approach:
        - Don't handle tiktoken encodings
        - Try to load any other model name

        Args:
            model_name: Name to validate

        Returns:
            True if model can be loaded by HuggingFace, False otherwise
        """
        # Don't handle tiktoken encodings
        tiktoken_encodings = {"o200k_base", "cl100k_base", "p50k_base", "p50k_edit", "r50k_base"}
        if model_name in tiktoken_encodings:
            return False

        # Check if it's already cached
        if model_name in self._model_cache:
            return True

        # Try to actually load the model to validate it
        try:
            from transformers import AutoTokenizer  # type: ignore

            # Try to load the model (this will validate it exists)
            AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False
            )
            return True

        except ImportError:
            # transformers not available, be permissive for model-like names
            return '/' in model_name
        except Exception:
            # Model doesn't exist or can't be loaded
            return False

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens without full tokenization.

        Args:
            text: Input text
            model_name: HuggingFace model name

        Returns:
            Number of tokens
        """
        model = self._get_model(model_name)

        try:
            # Use the model to encode and count tokens
            encoding = model(text, return_tensors=None, add_special_tokens=True)
            return len(encoding['input_ids'])
        except Exception as e:
            raise ModelLoadError(f"Failed to count tokens with '{model_name}': {e}")

    def supports_model(self, model_name: str) -> bool:
        """
        Check if this implementation can handle the given model.

        For HuggingFace, we accept any string that doesn't look like a tiktoken encoding.

        Args:
            model_name: Model name to check

        Returns:
            True if this implementation should handle the model
        """
        # Don't handle tiktoken encodings
        tiktoken_encodings = {"o200k_base", "cl100k_base", "p50k_base", "p50k_edit", "r50k_base"}
        if model_name in tiktoken_encodings:
            return False

        # Handle everything else (assume it's a HuggingFace model)
        return True
