try:
    import tiktoken  # type: ignore[import-not-found]
except Exception:
    tiktoken = None  # type: ignore[assignment]

from tokker.providers import Provider, register_provider
from tokker.exceptions import ModelLoadError, MissingDependencyError

@register_provider
class ProviderTiktoken(Provider):
    NAME = "OpenAI"
    MODELS: list[str] = [
        "o200k_base",
        "cl100k_base",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    ]

    def _get_encoding(self, model_name: str):
        if tiktoken is None:
            # Use structured exception with centralized templates
            raise MissingDependencyError("tiktoken", "tokker[tiktoken]")
        try:
            return tiktoken.get_encoding(model_name)
        except Exception as e:
            # Provide structured model/context info
            raise ModelLoadError(model_name, f"Failed to load tiktoken encoding: {e}")

    def tokenize(
        self,
        text: str,
        model_name: str,
    ) -> dict[str, str | int | list[str] | list[int]]:
        encoding = self._get_encoding(model_name)
        token_ids = encoding.encode(text)
        token_strings: list[str] = []
        for token_id in token_ids:
            try:
                token_strings.append(encoding.decode([token_id]))
            except Exception:
                token_strings.append(f"<token_{token_id}>")
        return {
            "token_strings": token_strings,
            "token_ids": token_ids,
            "token_count": len(token_ids),
        }
