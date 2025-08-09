#!/usr/bin/env python3
"""
Unit tests for ModelRegistry aligned with the current API.

Covers:
- Provider discovery snapshotting
- Static model -> provider mapping from class-level MODELS
- HuggingFace BYOM probing path (is_on_huggingface)
- Public APIs: list_models, get_providers, is_model_supported, get_provider_by_model, tokenize
"""

import unittest
from typing import cast
from unittest.mock import Mock, patch

from tokker.models.registry import ModelRegistry


class TestModelRegistryBasics(unittest.TestCase):
    """Basic behavior of the registry lifecycle and lookups."""

    def test_providers_and_models_are_discoverable(self):
        """Ensure registry discovers providers and builds model index."""
        r = ModelRegistry()

        # get_providers should be non-empty and stable
        providers = r.get_providers()
        self.assertIsInstance(providers, list)
        self.assertGreater(len(providers), 0)

        # list_models should return items with name/provider
        models = r.list_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        for item in models:
            self.assertIn("name", item)
            self.assertIn("provider", item)

        # Known OpenAI tiktoken encodings should be present
        model_names = {m["name"] for m in models}
        self.assertIn("cl100k_base", model_names)
        self.assertIn("o200k_base", model_names)

        # Filtering by provider works
        openai_only = r.list_models("OpenAI")
        self.assertTrue(all(m["provider"] == "OpenAI" for m in openai_only))

    def test_is_model_supported(self):
        """Known models should be supported; unknown should not."""
        r = ModelRegistry()
        self.assertTrue(r.is_model_supported("cl100k_base"))
        self.assertFalse(r.is_model_supported("__not_a_real_model__"))

    def test_get_provider_for_known_and_unknown(self):
        """get_provider_by_model returns an instance for known models and errors for unknown."""
        r = ModelRegistry()

        # Known OpenAI model should map to a provider instance
        provider = r.get_provider_by_model("cl100k_base")
        self.assertTrue(hasattr(provider, "NAME"))
        self.assertTrue(hasattr(provider, "tokenize"))

        # Unknown model should raise an exception
        with self.assertRaises(Exception):
            r.get_provider_by_model("__not_a_real_model__")


class TestTokenizeFlow(unittest.TestCase):
    """Tokenization through the registry should delegate to the right provider."""

    @patch("tokker.providers.tiktoken.tiktoken")
    def test_tokenize_known_model(self, mock_tiktoken):
        # Mock tiktoken encoding behavior so we don't require the optional extra
        class FakeEncoding:
            def encode(self, text):
                # simple deterministic split into two "tokens"
                return [1, 2]

            def decode(self, ids):
                # return placeholder strings based on ids
                return "Hello" if ids == [1] else " world"

        mock_tiktoken.get_encoding.return_value = FakeEncoding()

        r = ModelRegistry()
        result = r.tokenize("Hello", "cl100k_base")
        self.assertIn("token_strings", result)
        self.assertIn("token_ids", result)
        self.assertIn("token_count", result)
        # Narrow types for the checker
        token_ids = cast(list[int], result.get("token_ids"))
        token_count = cast(int, result.get("token_count"))
        # token_count should match length of token_ids
        self.assertEqual(token_count, len(token_ids))

    def test_tokenize_unknown_model(self):
        r = ModelRegistry()
        with self.assertRaises(Exception):
            r.tokenize("Hello", "__not_a_real_model__")


class TestHuggingFaceBYOMProbe(unittest.TestCase):
    """Validate dynamic BYOM probing path via HuggingFace provider."""

    @patch("tokker.models.registry.ModelRegistry._ensure_provider_instance")
    def test_hf_probe_is_used_when_static_index_misses(self, mock_ensure):
        """
        Simulate a model not present in the static index but validated by the HF provider.
        """
        r = ModelRegistry()
        # Force discovery to ensure provider class map is populated
        _ = r.get_providers()

        # Fake an HF provider instance with a validate function returning True
        fake_hf_provider = Mock()
        fake_hf_provider.NAME = "HuggingFace"
        fake_hf_provider.is_on_huggingface.return_value = True

        def _ensure_side_effect(name):
            if name == "HuggingFace":
                return fake_hf_provider
            raise AssertionError(f"Unexpected provider request: {name}")

        mock_ensure.side_effect = _ensure_side_effect

        # Pick a likely HF model name (not statically indexed by OpenAI provider)
        model_name = "gpt2"
        self.assertTrue(r.is_model_supported(model_name))

        # Now try to tokenize; we only check resolution path reaches provider
        # We can't actually run HF tokenization here, so stub tokenize too.
        fake_hf_provider.tokenize.return_value = {
            "token_strings": ["g", "pt", "2"],
            "token_ids": [1, 2, 3],
            "token_count": 3,
        }
        result = r.tokenize("gpt2", model_name)
        self.assertEqual(result["token_count"], 3)
        fake_hf_provider.tokenize.assert_called_once()

    @patch("tokker.models.registry.ModelRegistry._ensure_provider_instance")
    def test_hf_probe_negative_when_invalid(self, mock_ensure):
        """
        Simulate HF provider rejecting a model (validate returns False),
        and ensure resolution returns None (i.e., unsupported).
        """
        r = ModelRegistry()
        _ = r.get_providers()

        fake_hf_provider = Mock()
        fake_hf_provider.NAME = "HuggingFace"
        fake_hf_provider.is_on_huggingface.return_value = False
        mock_ensure.return_value = fake_hf_provider

        # Use a bogus model name to avoid static index hits
        bogus_model = "__bogus_hf_model__"
        self.assertFalse(r.is_model_supported(bogus_model))
        with self.assertRaises(Exception):
            r.get_provider_by_model(bogus_model)


if __name__ == "__main__":
    unittest.main()
