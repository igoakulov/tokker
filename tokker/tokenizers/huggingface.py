"""
HuggingFace tokenizer implementation for the Tokker plugin system.

This module provides HuggingFace transformers tokenizer support through the unified
BaseTokenizer interface using AutoTokenizer.from_pretrained().
"""

from typing import List, Dict, Any
import logging

from .base import BaseTokenizer
from .exceptions import TokenizerLoadError, MissingDependencyError

logger = logging.getLogger(__name__)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    HuggingFace tokenizer implementation.

    Supports any HuggingFace model that can be loaded with AutoTokenizer.from_pretrained().
    Prioritizes fast tokenizers when available.
    """

    def __init__(self):
        """Initialize the HuggingFace tokenizer."""
        # Cache for loaded tokenizers to avoid repeated downloads
        self._tokenizer_cache: Dict[str, Any] = {}

        # Common HuggingFace tokenizer models for quick access
        self._common_tokenizers = [
            # DeepSeek family
            "deepseek-ai/DeepSeek-Coder-V2-Base",
            "deepseek-ai/DeepSeek-V2",

            # GPT family (HuggingFace portion)
            "gpt2",

            # LLaMA family
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-70B",
            "meta-llama/Meta-Llama-3.1-405B",

            # Qwen family
            "Qwen/Qwen-72B",
            "Qwen/Qwen1.5-110B",
            "Qwen/Qwen2-72B",
            "Qwen/Qwen2.5-72B",

            # Other transformers
            "allenai/longformer-base-4096",
            "bert-base-cased",
            "bert-base-uncased",
            "distilbert-base-cased",
            "distilbert-base-uncased",
            "facebook/bart-base",
            "google/electra-base-discriminator",
            "microsoft/deberta-base",
            "roberta-base",
            "t5-base",
            "xlnet-base-cased"
        ]

    @property
    def library_name(self) -> str:
        """Return the library identifier."""
        return "hf"

    @property
    def supported_tokenizers(self) -> List[str]:
        """
        Return list of common supported HuggingFace tokenizers.

        Note: HuggingFace can potentially load any model from the hub,
        but we list common ones for discovery purposes.
        """
        return self._common_tokenizers

    @property
    def tokenizer_descriptions(self) -> Dict[str, str]:
        """Return descriptions for common HuggingFace tokenizers."""
        return {
            # DeepSeek family - BPE
            "deepseek-ai/DeepSeek-Coder-V2-Base": "BPE, used by DeepSeek-Coder-V2",
            "deepseek-ai/DeepSeek-V2": "BPE, used by DeepSeek-V2",

            # GPT family - BPE
            "gpt2": "BPE, used by GPT-2",

            # LLaMA family - BPE
            "meta-llama/Llama-2-70b-hf": "BPE, used by LLaMA-2",
            "meta-llama/Meta-Llama-3-70B": "BPE, used by LLaMA-3",
            "meta-llama/Meta-Llama-3.1-405B": "BPE, used by LLaMA-3.1",

            # Qwen family - BPE
            "Qwen/Qwen-72B": "BPE, used by Qwen",
            "Qwen/Qwen1.5-110B": "BPE, used by Qwen1.5",
            "Qwen/Qwen2-72B": "BPE, used by Qwen2",
            "Qwen/Qwen2.5-72B": "BPE, used by Qwen2.5",

            # Other transformers
            "allenai/longformer-base-4096": "BPE, used by Longformer",
            "bert-base-cased": "WordPiece, used by BERT",
            "bert-base-uncased": "WordPiece, used by BERT",
            "distilbert-base-cased": "WordPiece, used by DistilBERT",
            "distilbert-base-uncased": "WordPiece, used by DistilBERT",
            "facebook/bart-base": "BPE, used by BART",
            "google/electra-base-discriminator": "WordPiece, used by ELECTRA",
            "microsoft/deberta-base": "SentencePiece, used by DeBERTa",
            "roberta-base": "BPE, used by RoBERTa",
            "t5-base": "SentencePiece, used by T5",
            "xlnet-base-cased": "SentencePiece, used by XLNet"
        }

    def _get_tokenizer(self, tokenizer_name: str):
        """
        Load HuggingFace tokenizer with caching and fast tokenizer preference.

        Args:
            tokenizer_name: HuggingFace model name or path

        Returns:
            Loaded tokenizer instance

        Raises:
            MissingDependencyError: If transformers is not available
            TokenizerLoadError: If tokenizer cannot be loaded
        """
        # Check cache first
        if tokenizer_name in self._tokenizer_cache:
            return self._tokenizer_cache[tokenizer_name]

        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            raise MissingDependencyError(
                "transformers is required for HuggingFace tokenizers. "
                "Install with: pip install transformers"
            )

        try:
            logger.debug(f"Loading HuggingFace tokenizer: {tokenizer_name}")

            # Try to load with fast tokenizer first (as per PRD requirement)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                trust_remote_code=False  # Security consideration
            )

            # Verify it's actually a fast tokenizer
            if not tokenizer.is_fast:
                logger.warning(f"Fast tokenizer not available for {tokenizer_name}, using slow tokenizer")
                # Reload without fast requirement as fallback
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    use_fast=False,
                    trust_remote_code=False
                )

            # Cache the tokenizer
            self._tokenizer_cache[tokenizer_name] = tokenizer
            logger.debug(f"Successfully loaded tokenizer: {tokenizer_name} (fast: {tokenizer.is_fast})")

            return tokenizer

        except Exception as e:
            error_msg = f"Failed to load HuggingFace tokenizer '{tokenizer_name}': {e}"
            logger.error(error_msg)
            raise TokenizerLoadError(error_msg)

    def tokenize(self, text: str, tokenizer_name: str) -> Dict[str, Any]:
        """
        Tokenize text using HuggingFace tokenizer.

        Args:
            text: Input text to tokenize
            tokenizer_name: HuggingFace model name

        Returns:
            Dictionary with standardized tokenization results
        """
        tokenizer = self._get_tokenizer(tokenizer_name)

        try:
            # Encode the text to get token IDs
            encoding = tokenizer(text, return_tensors=None, add_special_tokens=True)
            token_ids = encoding['input_ids']

            # Get token strings by converting each ID back to text
            token_strings = []
            for token_id in token_ids:
                try:
                    # Convert single token ID to string
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    token_strings.append(token_str)
                except Exception:
                    # Handle potential decoding errors
                    token_strings.append(f"<token_{token_id}>")

            return {
                "token_strings": token_strings,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "tokenizer": tokenizer_name,
                "library": self.library_name
            }

        except Exception as e:
            raise TokenizerLoadError(f"Failed to tokenize text with '{tokenizer_name}': {e}")

    def validate_tokenizer(self, tokenizer_name: str) -> bool:
        """
        Validate if tokenizer can be loaded.

        For HuggingFace, we use a two-step approach:
        - Known common tokenizers are immediately valid
        - Other tokenizers are validated by attempting to load them
        - Tiktoken encodings are rejected

        Args:
            tokenizer_name: Name to validate

        Returns:
            True if tokenizer can be loaded by HuggingFace, False otherwise
        """
        # Quick check for common tokenizers
        if tokenizer_name in self._common_tokenizers:
            return True

        # Don't handle tiktoken encodings
        tiktoken_encodings = {"o200k_base", "cl100k_base", "p50k_base", "p50k_edit", "r50k_base"}
        if tokenizer_name in tiktoken_encodings:
            return False

        # Try to actually load the tokenizer to validate it
        try:
            from transformers import AutoTokenizer  # type: ignore

            # Check if it's already cached
            if tokenizer_name in self._tokenizer_cache:
                return True

            # Try to load the tokenizer (this will validate it exists)
            AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                trust_remote_code=False
            )
            return True

        except ImportError:
            # transformers not available, be permissive for model-like names
            return '/' in tokenizer_name or tokenizer_name in self._common_tokenizers
        except Exception:
            # Model doesn't exist or can't be loaded
            return False

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """
        Count tokens without full tokenization.

        Args:
            text: Input text
            tokenizer_name: HuggingFace model name

        Returns:
            Number of tokens
        """
        tokenizer = self._get_tokenizer(tokenizer_name)

        try:
            # Use the tokenizer to encode and count tokens
            encoding = tokenizer(text, return_tensors=None, add_special_tokens=True)
            return len(encoding['input_ids'])
        except Exception as e:
            raise TokenizerLoadError(f"Failed to count tokens with '{tokenizer_name}': {e}")

    def supports_tokenizer(self, tokenizer_name: str) -> bool:
        """
        Check if this implementation can handle the given tokenizer.

        For HuggingFace, we accept any string that doesn't look like a tiktoken encoding.

        Args:
            tokenizer_name: Tokenizer name to check

        Returns:
            True if this implementation should handle the tokenizer
        """
        # Don't handle tiktoken encodings
        tiktoken_encodings = {"o200k_base", "cl100k_base", "p50k_base", "p50k_edit", "r50k_base"}
        if tokenizer_name in tiktoken_encodings:
            return False

        # Handle everything else (assume it's a HuggingFace model)
        return True
