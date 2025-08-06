from typing import Any
from tokker.providers import Provider, register_provider
from tokker.exceptions import ModelLoadError, MissingDependencyError, TokenizationError


def _import_auto_tokenizer():  # Local helper
    # Lazy import to avoid triggering Transformers' import-time framework warnings
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        # Structured exception with centralized template
        raise MissingDependencyError("transformers", "tokker[hf]") from e
    return AutoTokenizer

@register_provider
class ProviderHuggingFace(Provider):
    NAME = "HuggingFace"
    MODELS: list[str] = []

    def __init__(self):  # Local helper
        self._model_cache: dict[str, Any] = {}

    def _get_model(self, model_name: str) -> Any:  # Local helper
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        AutoTokenizer = _import_auto_tokenizer()
        try:
            model: Any = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False,
            )
            if not model.is_fast:
                model = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=False,
                )
            self._model_cache[model_name] = model
            return model
        except Exception as e:
            # Use structured exception with model and reason
            raise ModelLoadError(model_name, f"Failed to load HuggingFace model: {e}")

    def tokenize(
        self,
        text: str,
        model_name: str,
    ) -> dict[str, Any]:
        tok = self._get_model(model_name)
        try:
            token_ids = tok.encode(text)
            token_strings: list[str] = []
            for token_id in token_ids:
                try:
                    token_strings.append(tok.decode([token_id]))
                except Exception:
                    token_strings.append(f"<token_{token_id}>")
            return {
                "token_strings": token_strings,
                "token_ids": token_ids,
                "token_count": len(token_ids),
            }
        except Exception as e:
            # Use TokenizationError to reflect runtime failures during encode/decode
            raise TokenizationError(model_name, f"Failed to tokenize text: {e}")

    def validate_model_with_huggingface(self, model_name: str) -> bool:
        if model_name in {
            "o200k_base",
            "cl100k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
        }:
            return False
        if model_name in self._model_cache:
            return True
        try:
            AutoTokenizer = _import_auto_tokenizer()
        except MissingDependencyError:
            # transformers not installed -> cannot validate dynamically; treat as unsupported
            return False
        try:
            AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False,
            )
            return True
        except Exception:
            return False
