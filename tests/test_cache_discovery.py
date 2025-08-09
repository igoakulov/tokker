# tokker/tests/test_cache_discovery.py
#!/usr/bin/env python3
"""
Tests for cache-discovery edge cases and provider-loading resilience.

- Validate that a valid cache can be used without importing providers.
- Validate that a cache miss leads to guarded discovery and cache writing.
- Validate JSON formatter handling of extremely strange/edge-case token strings.
"""

import json
import unittest
from unittest.mock import patch

from tokker.models.registry import ModelRegistry


class TestCacheDiscoveryEdgeCases(unittest.TestCase):
    def test_cache_usage_without_provider_imports(self):
        """
        If a valid cache is returned by the discovery cache, ensure the registry uses it
        without triggering provider imports.
        """
        with patch(
            "tokker.models.registry.load_models_from_cache",
            return_value=({"cl100k_base": "OpenAI"}, ["OpenAI"]),
        ) as _cache:
            with patch("tokker.models.registry.load_providers") as _load_providers:
                _load_providers.side_effect = AssertionError(
                    "Provider loading should be skipped when using cache"
                )

                r = ModelRegistry()
                providers = r.get_providers()
                self.assertIn("OpenAI", providers)

                models = r.list_models()
                self.assertTrue(
                    any(
                        m["name"] == "cl100k_base" and m["provider"] == "OpenAI"
                        for m in models
                    )
                )

    def test_cache_invalidation_guarded_discovery(self):
        """
        When the cache is not usable, ensure we perform guarded discovery and cache writing.
        """
        with patch(
            "tokker.models.registry.load_models_from_cache", return_value=None
        ) as _cache_none:
            load_seen = {"called": False}
            write_seen = {"called": False}

            def _load_providers():
                load_seen["called"] = True

            def _write_cache(cache_path, provider_names, model_index):
                write_seen["called"] = True

            with patch(
                "tokker.models.registry.load_providers", side_effect=_load_providers
            ) as _lp:
                with patch(
                    "tokker.models.registry.write_cache", side_effect=_write_cache
                ) as _wc:
                    r = ModelRegistry()
                    _ = r.get_providers()
                    self.assertTrue(
                        load_seen["called"],
                        "Guarded discovery did not call provider loader",
                    )
                    self.assertTrue(
                        write_seen["called"], "Guarded discovery did not write cache"
                    )


class TestJsonFormatterEdgeCase(unittest.TestCase):
    def test_json_formatter_with_weird_strings(self):
        """
        Ensure the JSON output formatter can handle extremely weird strings
        in token_strings without crashing and produces valid JSON.
        """
        from tokker.cli.output.formats import format_and_print_output

        weird_strings = ["A\nB", '"quote"', "\u0000", "😊", "⎮", "end"]
        base_json = {
            "delimited_text": "|".join(weird_strings),
            "token_strings": weird_strings,
            "token_ids": [1, 2, 3, 4, 5],
            "token_count": 5,
            "word_count": 6,
            "char_count": 10,
            "pivot": {"A\nB": 1, '"quote"': 1, "😊": 1, "⎮": 1, "end": 1},
        }

        # Capture the printer call and validate JSON parses correctly
        from unittest.mock import patch

        with patch("tokker.cli.output.formats.print") as mock_print:
            format_and_print_output(base_json, "json", "|")
            self.assertTrue(mock_print.called)

            printed = mock_print.call_args[0][0]
            data = json.loads(printed)
            self.assertIn("token_count", data)
            self.assertEqual(data["token_count"], 5)
            self.assertIn("A\nB", data["token_strings"])
