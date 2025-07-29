"""
Tokenizer registry for discovery and routing of tokenizer plugins.

The TokenizerRegistry maintains a central registry of all available tokenizer
implementations and provides routing logic to dispatch tokenization requests
to the appropriate implementation.
"""

import importlib
import pkgutil
from typing import Dict, List, Optional, Any
import logging

from .base import BaseTokenizer
from .exceptions import TokenizerNotFoundError, TokenizerLoadError

logger = logging.getLogger(__name__)


class TokenizerRegistry:
    """
    Central registry for tokenizer plugin discovery and routing.

    The registry automatically discovers tokenizer implementations
    and provides a unified interface for tokenization operations.
    """

    def __init__(self):
        """Initialize the registry."""
        self._tokenizers: Dict[str, BaseTokenizer] = {}
        self._tokenizer_mapping: Dict[str, str] = {}  # tokenizer_name -> library_name
        self._library_descriptions: Dict[str, Dict[str, str]] = {}
        self._initialized = False

    def _discover_tokenizers(self) -> None:
        """
        Discover and register all available tokenizer implementations.

        This method scans the tokenizers package for implementations
        and automatically registers them.
        """
        if self._initialized:
            return

        logger.debug("Discovering tokenizer implementations...")

        # Import the tokenizers package to scan for implementations
        import tokker.tokenizers as tokenizers_pkg

        # Scan for tokenizer modules
        for finder, name, ispkg in pkgutil.iter_modules(tokenizers_pkg.__path__):
            if name in ('base', 'registry', 'exceptions', '__init__'):
                continue

            try:
                # Import the module
                module_name = f'tokker.tokenizers.{name}'
                module = importlib.import_module(module_name)

                # Find tokenizer classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a tokenizer class
                    if (isinstance(attr, type) and
                        issubclass(attr, BaseTokenizer) and
                        attr != BaseTokenizer):

                        try:
                            # Instantiate and register the tokenizer
                            tokenizer_instance = attr()
                            self._register_tokenizer(tokenizer_instance)
                            logger.debug(f"Registered tokenizer: {attr_name} from {module_name}")
                        except Exception as e:
                            logger.warning(f"Failed to instantiate tokenizer {attr_name}: {e}")

            except ImportError as e:
                logger.warning(f"Failed to import tokenizer module {name}: {e}")
                continue

        self._initialized = True
        logger.debug(f"Discovery complete. Registered {len(self._tokenizers)} tokenizer libraries.")

    def _register_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        """
        Register a tokenizer implementation.

        Args:
            tokenizer: The tokenizer instance to register
        """
        library_name = tokenizer.library_name

        if library_name in self._tokenizers:
            logger.warning(f"Tokenizer library '{library_name}' already registered. Overwriting.")

        self._tokenizers[library_name] = tokenizer

        # Map individual tokenizer names to library
        for tokenizer_name in tokenizer.supported_tokenizers:
            if tokenizer_name in self._tokenizer_mapping:
                existing_library = self._tokenizer_mapping[tokenizer_name]
                logger.warning(
                    f"Tokenizer '{tokenizer_name}' already mapped to library '{existing_library}'. "
                    f"Overwriting with '{library_name}'."
                )

            self._tokenizer_mapping[tokenizer_name] = library_name

        # Store descriptions if available
        if hasattr(tokenizer, 'tokenizer_descriptions'):
            self._library_descriptions[library_name] = getattr(tokenizer, 'tokenizer_descriptions')

    def get_tokenizer(self, tokenizer_name: str) -> BaseTokenizer:
        """
        Get the appropriate tokenizer implementation for a given tokenizer name.

        Args:
            tokenizer_name: Name of the tokenizer to get

        Returns:
            The tokenizer implementation instance

        Raises:
            TokenizerNotFoundError: If the tokenizer is not found
        """
        self._discover_tokenizers()

        # First check if tokenizer is in the predefined mapping
        if tokenizer_name in self._tokenizer_mapping:
            library_name = self._tokenizer_mapping[tokenizer_name]
            return self._tokenizers[library_name]

        # If not found in mapping, try each tokenizer implementation's validate method
        for library_name, tokenizer_impl in self._tokenizers.items():
            if tokenizer_impl.validate_tokenizer(tokenizer_name):
                return tokenizer_impl

        # If no implementation can handle it, raise error
        available = list(self._tokenizer_mapping.keys())
        raise TokenizerNotFoundError(
            f"Tokenizer '{tokenizer_name}' not found. "
            f"Available common tokenizers: {', '.join(sorted(available))}. "
            f"For HuggingFace models, use format 'org/model-name' (e.g., 'microsoft/codebert-base')"
        )

    def list_tokenizers(self) -> List[Dict[str, str]]:
        """
        List all available tokenizers with their metadata.

        Returns:
            List of dictionaries with tokenizer information:
            - name: tokenizer name
            - library: library identifier
            - description: description of the tokenizer
        """
        self._discover_tokenizers()

        tokenizers = []

        for tokenizer_name, library_name in sorted(self._tokenizer_mapping.items()):
            # Get description from library descriptions or use default
            description = "No description available"

            if library_name in self._library_descriptions:
                lib_descriptions = self._library_descriptions[library_name]
                if tokenizer_name in lib_descriptions:
                    description = lib_descriptions[tokenizer_name]

            tokenizers.append({
                'name': tokenizer_name,
                'library': library_name,
                'description': description
            })

        return tokenizers

    def get_libraries(self) -> List[str]:
        """
        Get list of available tokenizer libraries.

        Returns:
            List of library identifiers
        """
        self._discover_tokenizers()
        return list(self._tokenizers.keys())

    def validate_tokenizer(self, tokenizer_name: str) -> bool:
        """
        Validate if a tokenizer is available.

        Args:
            tokenizer_name: Name of tokenizer to validate

        Returns:
            True if tokenizer is available, False otherwise
        """
        self._discover_tokenizers()

        # First check predefined mapping
        if tokenizer_name in self._tokenizer_mapping:
            return True

        # Then check if any implementation can handle it
        for tokenizer_impl in self._tokenizers.values():
            if tokenizer_impl.validate_tokenizer(tokenizer_name):
                return True

        return False

    def tokenize(self, text: str, tokenizer_name: str) -> Dict[str, Any]:
        """
        Tokenize text using the appropriate tokenizer implementation.

        Args:
            text: Text to tokenize
            tokenizer_name: Name of tokenizer to use

        Returns:
            Tokenization result dictionary

        Raises:
            TokenizerNotFoundError: If tokenizer is not found
            TokenizerLoadError: If tokenization fails
        """
        try:
            tokenizer = self.get_tokenizer(tokenizer_name)
            return tokenizer.tokenize(text, tokenizer_name)
        except Exception as e:
            if isinstance(e, TokenizerNotFoundError):
                raise
            raise TokenizerLoadError(f"Failed to tokenize with '{tokenizer_name}': {e}")

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """
        Count tokens using the appropriate tokenizer implementation.

        Args:
            text: Text to count tokens for
            tokenizer_name: Name of tokenizer to use

        Returns:
            Number of tokens

        Raises:
            TokenizerNotFoundError: If tokenizer is not found
            TokenizerLoadError: If token counting fails
        """
        try:
            tokenizer = self.get_tokenizer(tokenizer_name)
            return tokenizer.count_tokens(text, tokenizer_name)
        except Exception as e:
            if isinstance(e, TokenizerNotFoundError):
                raise
            raise TokenizerLoadError(f"Failed to count tokens with '{tokenizer_name}': {e}")


# Global registry instance
_registry: Optional[TokenizerRegistry] = None


def get_registry() -> TokenizerRegistry:
    """
    Get the global tokenizer registry instance.

    Returns:
        The global TokenizerRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = TokenizerRegistry()
    return _registry


def list_tokenizers() -> List[Dict[str, str]]:
    """
    Convenience function to list all available tokenizers.

    Returns:
        List of tokenizer information dictionaries
    """
    return get_registry().list_tokenizers()


def tokenize(text: str, tokenizer_name: str) -> Dict[str, Any]:
    """
    Convenience function to tokenize text.

    Args:
        text: Text to tokenize
        tokenizer_name: Name of tokenizer to use

    Returns:
        Tokenization result dictionary
    """
    return get_registry().tokenize(text, tokenizer_name)


def count_tokens(text: str, tokenizer_name: str) -> int:
    """
    Convenience function to count tokens.

    Args:
        text: Text to count tokens for
        tokenizer_name: Name of tokenizer to use

    Returns:
        Number of tokens
    """
    return get_registry().count_tokens(text, tokenizer_name)


def validate_tokenizer(tokenizer_name: str) -> bool:
    """
    Convenience function to validate tokenizer availability.

    Args:
        tokenizer_name: Name of tokenizer to validate

    Returns:
        True if tokenizer is available, False otherwise
    """
    return get_registry().validate_tokenizer(tokenizer_name)
