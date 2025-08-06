from tokker.providers import Provider, PROVIDERS  # uses explicit registration via
# decorators (providers register themselves via @register_provider)
from tokker.exceptions import ModelNotFoundError, ModelLoadError

class ModelRegistry:
    """
    Discover and map models to providers; lazily instantiate providers and
    delegate tokenization. Uses class-level MODELS for static mapping and
    probes HuggingFace for BYOM models.
    """
    def __init__(self):
        # Instantiated providers by provider name
        self._providers: dict[str, Provider] = {}
        # Registered provider classes by provider name (populated from PROVIDERS)
        self._provider_classes: dict[str, type[Provider]] = {}
        # Static model -> provider_name mapping (from class-level MODELS)
        self._model_to_provider: dict[str, str] = {}
        self._discovered: bool = False

    # ---- Local helpers ----

    def _ensure_discovered(self) -> None:
        """Snapshot provider classes and build the static model index (once)."""
        if self._discovered:
            return
        # Snapshot provider classes that registered at import time
        self._provider_classes = dict(PROVIDERS)
        # Build static model index from class-level MODELS
        self._build_model_index()
        self._discovered = True

    def _build_model_index(self) -> None:
        """Rebuild model_name -> provider_name mapping from provider MODELS."""
        self._model_to_provider.clear()
        for provider_name, cls in self._provider_classes.items():
            for model in (getattr(cls, "MODELS", []) or []):
                self._model_to_provider[model] = provider_name

    def _ensure_provider_instance(self, provider_name: str) -> Provider:
        """Return a cached provider instance or construct one via the class map."""
        if provider_name in self._providers:
            return self._providers[provider_name]
        cls = self._provider_classes.get(provider_name)
        if not cls:
            raise ModelNotFoundError(provider_name)
        try:
            instance = cls()
        except Exception as e:
            raise ModelLoadError(provider_name, f"Failed to initialize provider: {e}")
        self._providers[provider_name] = instance
        return instance

    def _find_provider_for_model(self, model_name: str) -> str | None:
        """Resolution policy: static map first, then HF BYOM probe."""
        provider_name = self._model_to_provider.get(model_name)
        if provider_name:
            return provider_name

        if not self._provider_classes.get("HuggingFace"):
            return None

        provider = self._ensure_provider_instance("HuggingFace")
        validate = getattr(provider, "validate_model_with_huggingface", None)
        return getattr(provider, "NAME", "HuggingFace") if (
            callable(validate) and validate(model_name)
        ) else None

    # -------- public API --------

    def get_provider(self, model_name: str) -> Provider:
        """Return a provider instance that supports the given model name."""
        self._ensure_discovered()
        provider_name = self._find_provider_for_model(model_name)
        if not provider_name:
            raise ModelNotFoundError(model_name)
        return self._ensure_provider_instance(provider_name)

    def list_models(self, provider: str | None = None) -> list[dict[str, str]]:
        """List known models, optionally filtered by provider name."""
        self._ensure_discovered()
        items = (
            {"name": m, "provider": p}
            for m, p in self._model_to_provider.items()
            if provider is None or p == provider
        )
        return sorted(items, key=lambda i: (i["name"], i["provider"]))

    def get_providers(self) -> list[str]:
        """Return sorted provider names."""
        self._ensure_discovered()
        return sorted(self._provider_classes.keys())

    def is_model_supported(self, model_name: str) -> bool:
        """Check if a model name is resolvable to a provider."""
        self._ensure_discovered()
        return self._find_provider_for_model(model_name) is not None

    def tokenize(
        self,
        text: str,
        model_name: str,
    ) -> dict[str, str | int | list[str] | list[int]]:
        """Tokenize text via the appropriate provider, normalizing load errors."""
        try:
            provider = self.get_provider(model_name)
            return provider.tokenize(text, model_name)
        except Exception as e:
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelLoadError(model_name, str(e))
