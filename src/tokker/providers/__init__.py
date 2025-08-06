# Import the abstract base class for providers
from .provider import Provider

# Global mapping of provider name -> provider class
PROVIDERS: dict[str, type[Provider]] = {}


def register_provider(cls: type[Provider]) -> type[Provider]:
    name = getattr(cls, "NAME", None)
    if isinstance(name, str) and name and name not in PROVIDERS:
        PROVIDERS[name] = cls
    return cls

# Ensure early import of built-in providers so their @register_provider runs.
# This avoids dynamic filesystem scanning while keeping behavior explicit.
# If a provider has optional dependencies, it should handle ImportError internally.
try:
    from . import tiktoken as _tiktoken_provider  # noqa: F401
except Exception:
    # Optional dependency may not be installed; skip silent import failure
    pass

try:
    from . import huggingface as _huggingface_provider  # noqa: F401
except Exception:
    pass

try:
    from . import google as _google_provider  # noqa: F401
except Exception:
    pass


__all__ = [
    "Provider",
    "register_provider",
    "PROVIDERS",
]
