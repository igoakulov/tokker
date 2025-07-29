"""
Tiktoken tokenizer implementation for the Tokker plugin system.

This module provides tiktoken tokenizer support through the unified
BaseTokenizer interface.
"""

from typing import List, Dict, Any, TYPE_CHECKING

from .base import BaseTokenizer
from .exceptions import TokenizerLoadError, UnsupportedTokenizerError, MissingDependencyError

if TYPE_CHECKING:
    pass


class TiktokenTokenizer(BaseTokenizer):
    """
    Tiktoken tokenizer implementation.

    Supports OpenAI's tiktoken encodings including o200k_base and cl100k_base.
    """

    @property
    def library_name(self) -> str:
        """Return the library identifier."""
        return "tt"

    @property
    def supported_tokenizers(self) -> List[str]:
        """Return list of supported tiktoken encodings."""
        return [
            "o200k_base",
            "cl100k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base"
        ]

    @property
    def tokenizer_descriptions(self) -> Dict[str, str]:
        """Return descriptions for each supported tokenizer."""
        return {
            "o200k_base": "BPE, used by GPT-4o, o-family (o1, o3, o4)",
            "cl100k_base": "BPE, used by GPT-3.5, GPT-4",
            "p50k_base": "BPE, used by GPT-3.5",
            "p50k_edit": "BPE, used by GPT-3 edit models for text and code (text-davinci, code-davinci)",
            "r50k_base": "BPE, used by GPT-3 base models (davinci, curie, babbage, ada)"
        }

    def _get_encoding(self, tokenizer_name: str):
        """
        Get tiktoken encoding by name.

        Args:
            tokenizer_name: Name of the tokenizer

        Returns:
            tiktoken.Encoding instance

        Raises:
            TokenizerLoadError: If tokenizer cannot be loaded
            UnsupportedTokenizerError: If tokenizer is not supported
        """
        if not self.validate_tokenizer(tokenizer_name):
            raise UnsupportedTokenizerError(
                f"Tokenizer '{tokenizer_name}' is not supported. "
                f"Supported tokenizers: {', '.join(self.supported_tokenizers)}"
            )

        try:
            import tiktoken
            return tiktoken.get_encoding(tokenizer_name)
        except ImportError:
            raise MissingDependencyError(
                "tiktoken is required for tiktoken tokenizers. "
                "Install with: pip install tiktoken"
            )
        except Exception as e:
            raise TokenizerLoadError(f"Failed to load tiktoken encoding '{tokenizer_name}': {e}")

    def tokenize(self, text: str, tokenizer_name: str) -> Dict[str, Any]:
        """
        Tokenize text using tiktoken.

        Args:
            text: Input text to tokenize
            tokenizer_name: Name of tokenizer to use

        Returns:
            Dictionary with standardized tokenization results
        """
        encoding = self._get_encoding(tokenizer_name)

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
            "tokenizer": tokenizer_name,
            "library": self.library_name
        }

    def validate_tokenizer(self, tokenizer_name: str) -> bool:
        """
        Validate if tokenizer is supported.

        Args:
            tokenizer_name: Name to validate

        Returns:
            True if tokenizer is supported, False otherwise
        """
        return tokenizer_name in self.supported_tokenizers

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """
        Count tokens without full tokenization.

        Args:
            text: Input text
            tokenizer_name: Name of tokenizer to use

        Returns:
            Number of tokens
        """
        encoding = self._get_encoding(tokenizer_name)
        return len(encoding.encode(text))
