import unittest
from unittest.mock import patch, Mock

from tokker import api
from tokker.exceptions import ModelNotFoundError


class TestAPI(unittest.TestCase):
    def test_count_words_basic(self):
        self.assertEqual(api.count_words("Hello world"), 2)
        self.assertEqual(api.count_words("   leading and  multiple   spaces "), 4)
        self.assertEqual(api.count_words(""), 0)
        self.assertEqual(api.count_words("   \n\t  "), 0)

    def test_count_characters(self):
        self.assertEqual(api.count_characters(""), 0)
        self.assertEqual(api.count_characters("abc"), 3)
        self.assertEqual(api.count_characters("Hello world"), 11)

    @patch("tokker.api.ModelRegistry")
    def test_list_models_all_and_filtered(self, mock_registry_cls):
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry

        all_models = [
            {"name": "cl100k_base", "provider": "OpenAI"},
            {"name": "gpt2", "provider": "HuggingFace"},
        ]
        mock_registry.list_models.side_effect = [
            all_models,  # first call: no filter
            [all_models[0]],  # second call: filter OpenAI
        ]

        # No filter
        result_all = api.list_models()
        self.assertEqual(result_all, all_models)
        mock_registry.list_models.assert_any_call(provider=None)

        # With provider filter
        result_openai = api.list_models(provider="OpenAI")
        self.assertEqual(result_openai, [all_models[0]])
        mock_registry.list_models.assert_any_call(provider="OpenAI")

    @patch("tokker.api.ModelRegistry")
    def test_get_providers(self, mock_registry_cls):
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_providers.return_value = ["Google", "HuggingFace", "OpenAI"]

        providers = api.get_providers()
        self.assertEqual(providers, ["Google", "HuggingFace", "OpenAI"])
        mock_registry.get_providers.assert_called_once()

    @patch("tokker.api.ModelRegistry")
    def test_tokenize_success(self, mock_registry_cls):
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = True
        mock_registry.tokenize.return_value = {
            "token_strings": ["Hello", " world"],
            "token_ids": [1, 2],
            "token_count": 2,
        }

        result = api.tokenize("Hello world", "cl100k_base")
        self.assertEqual(result["token_count"], 2)
        mock_registry.is_model_supported.assert_called_once_with("cl100k_base")
        mock_registry.tokenize.assert_called_once_with("Hello world", "cl100k_base")

    @patch("tokker.api.ModelRegistry")
    def test_tokenize_unknown_model_raises(self, mock_registry_cls):
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = False

        with self.assertRaises(ModelNotFoundError):
            api.tokenize("Hello", "__bogus__")
        mock_registry.is_model_supported.assert_called_once_with("__bogus__")
        mock_registry.tokenize.assert_not_called()

    @patch("tokker.api.tokenize")
    def test_count_tokens_prefers_token_count(self, mock_tokenize):
        mock_tokenize.return_value = {
            "token_strings": ["a", "b", "c"],
            "token_ids": [10, 11, 12],
            "token_count": 3,
        }
        self.assertEqual(api.count_tokens("abc", "cl100k_base"), 3)

    @patch("tokker.api.tokenize")
    def test_count_tokens_fallback_to_len_token_ids_when_missing_count(self, mock_tokenize):
        mock_tokenize.return_value = {
            "token_strings": ["a", "b", "c", "d"],
            "token_ids": [1, 2, 3, 4],
            # token_count intentionally missing
        }
        self.assertEqual(api.count_tokens("abcd", "cl100k_base"), 4)

    @patch("tokker.api.tokenize")
    def test_count_tokens_coerces_non_int_count(self, mock_tokenize):
        mock_tokenize.return_value = {
            "token_strings": ["x", "y"],
            "token_ids": [5, 6],
            "token_count": 2.0,  # float or any numeric should be coerced to int
        }
        self.assertEqual(api.count_tokens("xy", "cl100k_base"), 2)


if __name__ == "__main__":
    unittest.main()
