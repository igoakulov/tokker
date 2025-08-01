"""
Model registry for discovery and routing of model plugins.

The ModelRegistry maintains a central registry of all available model
implementations and provides routing logic to dispatch tokenization requests
to the appropriate implementation.
"""

import importlib
import pkgutil
from typing import Dict, List, Optional, Any
import logging

from .base import BaseModel
from .exceptions import ModelNotFoundError, ModelLoadError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for model plugin discovery and routing.

    The registry automatically discovers model implementations
    and provides a unified interface for tokenization operations.
    """

    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, BaseModel] = {}
        self._model_mapping: Dict[str, str] = {}  # model_name -> provider_name
        self._provider_descriptions: Dict[str, Dict[str, str]] = {}
        self._initialized = False

    def _discover_models(self) -> None:
        """
        Discover and register all available model provider implementations.

        This method scans the models package for implementations
        and automatically registers them.
        """
        if self._initialized:
            return

        logger.debug("Discovering model provider implementations...")

        # Import the models package to scan for implementations
        import tokker.models as models_pkg

        # Scan for model modules
        for finder, name, ispkg in pkgutil.iter_modules(models_pkg.__path__):
            if name in ('base', 'registry', 'exceptions', '__init__'):
                continue

            try:
                # Import the module
                module_name = f'tokker.models.{name}'
                try:
                    module = importlib.import_module(module_name)
                except Exception as e:
                    logger.warning(f"Failed to import model provider module {module_name}: {e}")
                    continue

                # Find model classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a model class
                    if (isinstance(attr, type) and
                        issubclass(attr, BaseModel) and
                        attr != BaseModel):

                        try:
                            # Instantiate and register the model
                            try:
                                model_instance = attr()
                            except Exception as e:
                                logger.warning(f"Failed to instantiate model provider {attr_name} from {module_name}: {e}")
                                continue
                            self._register_model(model_instance)
                            logger.debug(f"Registered model provider: {attr_name} from {module_name}")
                        except Exception as e:
                            logger.warning(f"Error while registering model provider {attr_name} from {module_name}: {e}")

            except ImportError as e:
                logger.warning(f"Failed to import model provider module {name}: {e}")
                continue

        self._initialized = True
        logger.debug(f"Discovery complete. Registered {len(self._models)} model providers.")

    def _register_model(self, model: BaseModel) -> None:
        """
        Register a model provider implementation.

        Args:
            model: The model provider instance to register
        """
        provider_name = model.library_name

        if provider_name in self._models:
            logger.warning(f"Model provider '{provider_name}' already registered. Overwriting.")

        self._models[provider_name] = model

        # Map individual model names to provider
        for model_name in model.supported_models:
            if model_name in self._model_mapping:
                existing_provider = self._model_mapping[model_name]
                logger.warning(
                    f"Model '{model_name}' already mapped to provider '{existing_provider}'. "
                    f"Overwriting with '{provider_name}'."
                )

            self._model_mapping[model_name] = provider_name

        # Store descriptions if available
        # NOTE: Description field is deprecated but kept for backward compatibility
        if hasattr(model, 'model_descriptions'):
            self._provider_descriptions[provider_name] = getattr(model, 'model_descriptions')

    def get_model(self, model_name: str) -> BaseModel:
        """
        Get the appropriate model provider implementation for a given model name.

        Args:
            model_name: Name of the model to get

        Returns:
            The model provider implementation instance

        Raises:
            ModelNotFoundError: If the model is not found
        """
        self._discover_models()

        # First check if model is in the predefined mapping
        if model_name in self._model_mapping:
            provider_name = self._model_mapping[model_name]
            return self._models[provider_name]

        # If not found in mapping, try each model implementation's validate method
        for provider_name, model_impl in self._models.items():
            if model_impl.validate_model(model_name):
                return model_impl

        # If no implementation can handle it, raise error
        available = list(self._model_mapping.keys())
        raise ModelNotFoundError(
            f"Model '{model_name}' not found. "
            f"Available common models: {', '.join(sorted(available))}. "
            f"For HuggingFace models, use format 'org/model-name' (e.g., 'microsoft/codebert-base')"
        )

    def list_models(self) -> List[Dict[str, str]]:
        """
        List all available models with their metadata.

        Returns:
            List of dictionaries with model information:
            - name: model name
            - provider: provider identifier
            - description: description of the model (DEPRECATED field)
        """
        self._discover_models()

        models = []

        for model_name, provider_name in sorted(self._model_mapping.items()):
            # Get description from library descriptions or use default
            # NOTE: Description field is deprecated but kept for backward compatibility
            description = "No description available"

            if provider_name in self._provider_descriptions:
                provider_descriptions = self._provider_descriptions[provider_name]
                if model_name in provider_descriptions:
                    description = provider_descriptions[model_name]

            models.append({
                'name': model_name,
                'provider': provider_name,
                'description': description
            })

        return models

    def get_libraries(self) -> List[str]:
        """
        Get list of available model providers.

        Returns:
            List of provider identifiers
        """
        self._discover_models()
        return list(self._models.keys())

    def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model is available.

        Args:
            model_name: Name of model to validate

        Returns:
            True if model is available, False otherwise
        """
        self._discover_models()

        # First check predefined mapping
        if model_name in self._model_mapping:
            return True

        # Then check if any implementation can handle it
        for model_impl in self._models.values():
            if model_impl.validate_model(model_name):
                return True

        return False

    def tokenize(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Tokenize text using the appropriate model provider implementation.

        Args:
            text: Text to tokenize
            model_name: Name of model to use

        Returns:
            Tokenization result dictionary

        Raises:
            ModelNotFoundError: If model is not found
            ModelLoadError: If tokenization fails
        """
        try:
            model = self.get_model(model_name)
            result = model.tokenize(text, model_name)
            # Update field name from library to provider for consistency
            if 'library' in result:
                result['provider'] = result.pop('library')
            return result
        except Exception as e:
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelLoadError(f"Failed to tokenize with model '{model_name}': {e}")

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens using the appropriate model provider implementation.

        Args:
            text: Text to count tokens for
            model_name: Name of model to use

        Returns:
            Number of tokens

        Raises:
            ModelNotFoundError: If model is not found
            ModelLoadError: If token counting fails
        """
        try:
            model = self.get_model(model_name)
            return model.count_tokens(text, model_name)
        except Exception as e:
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelLoadError(f"Failed to count tokens with model '{model_name}': {e}")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Returns:
        The global ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def list_models() -> List[Dict[str, str]]:
    """
    Convenience function to list all available models.

    Returns:
        List of model information dictionaries
    """
    return get_registry().list_models()


def tokenize(text: str, model_name: str) -> Dict[str, Any]:
    """
    Convenience function to tokenize text.

    Args:
        text: Text to tokenize
        model_name: Name of model to use

    Returns:
        Tokenization result dictionary
    """
    return get_registry().tokenize(text, model_name)


def count_tokens(text: str, model_name: str) -> int:
    """
    Convenience function to count tokens.

    Args:
        text: Text to count tokens for
        model_name: Name of model to use

    Returns:
        Number of tokens
    """
    return get_registry().count_tokens(text, model_name)


def validate_model(model_name: str) -> bool:
    """
    Convenience function to validate model availability.

    Args:
        model_name: Name of model to validate

    Returns:
        True if model is available, False otherwise
    """
    return get_registry().validate_model(model_name)
