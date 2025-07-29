"""
Base tokenizer interface for the Tokker plugin system.

All tokenizer implementations must inherit from BaseTokenizer
to ensure a consistent interface across different tokenization libraries.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizer implementations.

    This interface ensures consistent behavior across different
    tokenization libraries (tiktoken, transformers, etc.).
    """

    @property
    @abstractmethod
    def library_name(self) -> str:
        """
        Return the short library identifier.

        Examples:
            - 'tt' for tiktoken
            - 'hf' for HuggingFace transformers
            - 'sp' for SentencePiece
        """
        pass

    @property
    @abstractmethod
    def supported_tokenizers(self) -> List[str]:
        """
        Return list of tokenizer names this implementation supports.

        Examples:
            - ['o200k_base', 'cl100k_base'] for tiktoken
            - ['bert-base-uncased', 'gpt2'] for HuggingFace
        """
        pass

    @abstractmethod
    def tokenize(self, text: str, tokenizer_name: str) -> Dict[str, Any]:
        """
        Tokenize text and return standardized result.

        Args:
            text: Input text to tokenize
            tokenizer_name: Name of tokenizer to use

        Returns:
            Dictionary with standardized fields:
            - token_strings: List[str] - decoded token strings
            - token_ids: List[int] - token IDs
            - token_count: int - number of tokens
            - tokenizer: str - tokenizer name used
            - library: str - library identifier
        """
        pass

    @abstractmethod
    def validate_tokenizer(self, tokenizer_name: str) -> bool:
        """
        Validate if tokenizer is supported by this implementation.

        Args:
            tokenizer_name: Name to validate

        Returns:
            True if tokenizer is supported, False otherwise
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """
        Count tokens without full tokenization.

        Args:
            text: Input text
            tokenizer_name: Name of tokenizer to use

        Returns:
            Number of tokens
        """
        pass
