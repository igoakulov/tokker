"""
Google Gemini model implementation for the Tokker plugin system.

This module integrates with Google's Generative AI SDK (google-genai)
to provide tokenization support via the compute_tokens API for Gemini models.
"""

from typing import List, Dict, Any, Optional
import logging
import os
import shutil
import subprocess

from .base import BaseModel
from .exceptions import (
    ModelLoadError,
    UnsupportedModelError,
    MissingDependencyError,
    TokenizationError,
)

logger = logging.getLogger(__name__)


class GoogleModel(BaseModel):
    """
    Google Gemini model implementation.

    Uses google-genai Client(models.compute_tokens) to obtain token IDs and strings.
    """

    # Supported Gemini models as per PRD
    _SUPPORTED_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]



    def __init__(self, client: Optional[Any] = None):
        """
        Initialize the Google model.

        Args:
            client: Optional pre-initialized google.genai.Client for dependency injection
        """
        self._client = client  # Lazy init by default

    @property
    def library_name(self) -> str:
        """Return the provider identifier."""
        return "Google"

    @property
    def supported_models(self) -> List[str]:
        """Return list of supported Gemini models."""
        return self._SUPPORTED_MODELS

    def _get_client(self):
        """
        Create or return a cached google-genai client instance.

        Returns:
            google.genai.Client instance

        Raises:
            MissingDependencyError: if google-genai is not installed
            ModelLoadError: on unexpected client init failure
        """
        if self._client is not None:
            return self._client

        try:
            # Import within method to avoid hard dependency when provider unused
            from google import genai  # type: ignore
            from google.genai.types import HttpOptions  # type: ignore

            # Detect Vertex project and location to make ADC "just work"
            project = (
                os.environ.get("GOOGLE_CLOUD_PROJECT")
                or os.environ.get("GCLOUD_PROJECT")
                or os.environ.get("CLOUD_PROJECT")
                or None
            )
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

            # Fallback to gcloud config if project is not set and gcloud is available
            if project is None:
                gcloud = shutil.which("gcloud")
                if gcloud:
                    try:
                        proc = subprocess.run(
                            [gcloud, "config", "get-value", "project"],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        candidate = (proc.stdout or "").strip()
                        if candidate and candidate != "(unset)":
                            project = candidate
                    except Exception:
                        # Ignore gcloud probing errors
                        pass

            # Initialize Client in Vertex mode to use ADC
            # If project remains None, let the client attempt its own discovery;
            # we still pass location with a sensible default.
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=HttpOptions(api_version="v1"),
            )
            return self._client
        except ImportError:
            raise MissingDependencyError(
                "google-genai is required for Google Gemini models. "
                "Install with: pip install google-genai"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize Google client: {e}")

    def _ensure_supported(self, model_name: str) -> None:
        """
        Validate the provided model name.

        Raises:
            UnsupportedModelError if the model is not supported
        """
        if model_name not in self._SUPPORTED_MODELS:
            raise UnsupportedModelError(
                f"Model '{model_name}' is not supported by Google provider. "
                f"Supported models: {', '.join(self._SUPPORTED_MODELS)}"
            )

    def tokenize(self, text: str, model_name: str) -> Dict[str, Any]:
        """
        Tokenize text using Google's compute_tokens API.

        Args:
            text: Input text to tokenize
            model_name: Gemini model name

        Returns:
            Dictionary with standardized tokenization results:
            - token_strings: List[str]
            - token_ids: List[int]
            - token_count: int
            - model: str
            - provider: str

        Raises:
            TokenizationError on API or auth errors
        """
        self._ensure_supported(model_name)
        client = self._get_client()

        try:
            # Compute tokens via Gemini API
            # According to PRD sample:
            # response = client.models.compute_tokens(model=model_name, contents="...")
            response = client.models.compute_tokens(model=model_name, contents=text)

            # Expected structure (per PRD):
            # tokens_info=[TokensInfo(role='user', token_ids=[...], tokens=[b'...', ...])]
            token_ids: List[int] = []
            token_strings: List[str] = []

            # Some responses might include multiple roles or segments; concatenate
            tokens_info = getattr(response, "tokens_info", None)
            if not tokens_info:
                # Fallback: some responses could place tokens flat (being defensive)
                # If not present, raise meaningful error
                raise TokenizationError(
                    "Google compute_tokens returned no tokens_info. "
                    "Ensure text is non-empty and credentials are configured."
                )

            for info in tokens_info:
                # Collect IDs
                ids = getattr(info, "token_ids", []) or []
                token_ids.extend(ids)

                # Collect tokens; may be bytes
                toks = getattr(info, "tokens", []) or []
                for t in toks:
                    if isinstance(t, bytes):
                        try:
                            token_strings.append(t.decode("utf-8", errors="replace"))
                        except Exception:
                            # Best-effort fallback
                            token_strings.append(str(t))
                    else:
                        # Already str
                        token_strings.append(str(t))

            return {
                "token_strings": token_strings,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "model": model_name,
                "provider": self.library_name,
            }

        except TokenizationError:
            # Re-raise structured tokenization errors
            raise
        except Exception as e:
            # Common cases include ADC/auth errors and permission issues
            # Surface the raw message so the CLI can show it to the user
            raise TokenizationError(
                f"Google compute_tokens failed for model '{model_name}': {e}"
            )

    def validate_model(self, model_name: str) -> bool:
        """
        Validate if the model is supported by this provider.

        Args:
            model_name: Name to validate

        Returns:
            True if model is supported by Google, False otherwise
        """
        return model_name in self._SUPPORTED_MODELS

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens without full tokenization.

        Uses the same compute_tokens API and returns the token count length.

        Args:
            text: Input text
            model_name: Gemini model name

        Returns:
            Number of tokens

        Raises:
            TokenizationError on API or auth errors
        """
        result = self.tokenize(text, model_name)
        return result.get("token_count", 0)
