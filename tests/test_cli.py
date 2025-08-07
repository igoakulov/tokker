"""
Unit tests for CLI functionality aligned with the new structure.

Tests the CLI argument parsing, command handling, and output formatting.
"""

import unittest
import json
from unittest.mock import Mock, patch
from io import StringIO

from tokker.cli.arguments import build_argument_parser
from tokker.cli.tokenize import main as cli_main
from tokker.cli.commands.list_models import run_list_models
from tokker.cli.commands.set_default_model import run_set_default_model
from tokker.cli.commands.tokenize_text import run_tokenize
from tokker import messages as msg


class TestCLIParser(unittest.TestCase):
    """Test cases for CLI argument parsing."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = build_argument_parser()

    def test_basic_tokenize_args(self):
        """Test basic tokenization arguments."""
        # Test with text argument
        args = self.parser.parse_args(["Hello world"])
        self.assertEqual(args.text, "Hello world")
        self.assertIsNone(args.model)
        self.assertEqual(args.output, "json")

        # Test with model
        args = self.parser.parse_args(["Hello world", "--model", "gpt2"])
        self.assertEqual(args.text, "Hello world")
        self.assertEqual(args.model, "gpt2")

        # Test with output format
        args = self.parser.parse_args(["Hello world", "--output", "plain"])
        self.assertEqual(args.text, "Hello world")
        self.assertEqual(args.output, "plain")

    def test_models_args(self):
        """Test models list arguments."""
        args = self.parser.parse_args(["--models"])
        self.assertTrue(args.models)

    def test_history_and_clear_args(self):
        """Test history-related arguments."""
        args = self.parser.parse_args(["--history"])
        self.assertTrue(args.history)
        args = self.parser.parse_args(["--history-clear"])
        self.assertTrue(args.history_clear)

    def test_pivot_output_arg(self):
        """Test that pivot is an accepted output format."""
        args = self.parser.parse_args(["Hello", "--output", "pivot"])
        self.assertEqual(args.text, "Hello")
        self.assertEqual(args.output, "pivot")

    def test_model_default_args(self):
        """Test model default arguments."""
        args = self.parser.parse_args(["--model-default", "cl100k_base"])
        self.assertEqual(args.model_default, "cl100k_base")

    def test_no_args(self):
        """Test parsing with no arguments (stdin mode)."""
        args = self.parser.parse_args([])
        self.assertIsNone(args.text)
        self.assertFalse(args.models)
        self.assertIsNone(args.model_default)


class TestListModelsCommand(unittest.TestCase):
    """Test cases for models command handling."""

    @patch("tokker.cli.commands.list_models.print")
    @patch("tokker.cli.commands.list_models.ModelRegistry")
    def test_models_output(self, mock_registry_cls, mock_print):
        """Test models output."""
        # Mock registry and models
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.list_models.side_effect = (
            lambda provider=None: [
                {"name": "cl100k_base", "provider": "OpenAI"},
                {"name": "gemini-2.5-pro", "provider": "Google"},
                {"name": "gpt2", "provider": "HuggingFace"},
            ]
            if provider is None
            else (
                [{"name": "cl100k_base", "provider": "OpenAI"}]
                if provider == "OpenAI"
                else [{"name": "gemini-2.5-pro", "provider": "Google"}]
                if provider == "Google"
                else [{"name": "gpt2", "provider": "HuggingFace"}]
            )
        )

        # Call function
        run_list_models()

        # Verify print calls
        self.assertTrue(mock_print.called)
        # Ensure Google section header is printed and guidance line present
        printed = "\n".join(
            call.args[0] for call in mock_print.call_args_list if call.args
        )
        self.assertIn("Google:", printed)
        self.assertIn(msg.MSG_AUTH_REQUIRED.strip(), printed)


class TestSetDefaultModelCommand(unittest.TestCase):
    """Test cases for model default command handling."""

    @patch("tokker.cli.commands.set_default_model.config")
    @patch("tokker.cli.commands.set_default_model.ModelRegistry")
    @patch("builtins.print")
    def test_valid_model_default(self, mock_print, mock_registry_cls, mock_config):
        """Test setting valid default model."""
        # Mock registry
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = True
        mock_registry.list_models.return_value = [
            {"name": "cl100k_base", "provider": "OpenAI"}
        ]

        # Mock config
        mock_config.config_file = "/path/to/config.json"

        # Call function
        run_set_default_model("cl100k_base")

        # Verify model was set
        mock_config.set_default_model.assert_called_once_with("cl100k_base")

        # Verify success message (no checkmark or description in current behavior)
        calls = mock_print.call_args_list
        printouts = [c.args[0] for c in calls if c.args]
        combined = "\n".join(printouts)
        self.assertIn("Default model set to: cl100k_base (OpenAI)", combined)
        self.assertIn("Configuration saved to:", combined)


class TestTokenizeCommand(unittest.TestCase):
    """Test cases for tokenize command handling."""

    @patch("tokker.cli.commands.tokenize_text.ModelRegistry")
    @patch("tokker.cli.commands.tokenize_text.config")
    @patch("tokker.cli.output.formats.print")
    def test_tokenize_with_specific_model(
        self, mock_print, mock_config, mock_registry_cls
    ):
        """Test tokenization with specific model."""
        # Mock config
        mock_config.get_delimiter.return_value = "⎮"
        mock_config.add_model_to_history.return_value = None

        # Mock registry
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = True
        mock_result = {
            "token_strings": ["Hello", " world"],
            "token_ids": [123, 456],
            "token_count": 2,
        }
        mock_registry.tokenize.return_value = mock_result

        # Call function
        run_tokenize("Hello world", "gpt2", "json")

        # Verify tokenization and printing
        mock_registry.tokenize.assert_called_once_with("Hello world", "gpt2")
        self.assertTrue(mock_print.called)

    @patch("tokker.cli.commands.tokenize_text.ModelRegistry")
    @patch("tokker.cli.commands.tokenize_text.config")
    @patch("tokker.cli.output.formats.print")
    def test_tokenize_pivot_output(self, mock_print, mock_config, mock_registry_cls):
        """Test pivot output prints a token frequency map in JSON."""
        # Mock config
        mock_config.get_delimiter.return_value = "⎮"
        mock_config.add_model_to_history.return_value = None

        # Mock registry and tokenization result with duplicate tokens for pivot
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = True
        mock_result = {
            "token_strings": ["foo", " ", "foo"],
            "token_ids": [1, 2, 1],
            "token_count": 3,
        }
        mock_registry.tokenize.return_value = mock_result

        # Execute
        run_tokenize("foo foo", "cl100k_base", "pivot")

        # Ensure pivot JSON printed
        self.assertTrue(mock_print.called)
        printed = "".join(
            call.args[0] for call in mock_print.call_args_list if call.args
        )
        data = json.loads(printed)
        # Expect pivot counts for tokens present in token_strings (spaces may be present)
        self.assertIn("foo", data)
        self.assertEqual(data["foo"], 2)

    @patch("tokker.cli.commands.tokenize_text.ModelRegistry")
    @patch("tokker.cli.commands.tokenize_text.config")
    @patch("tokker.cli.output.formats.print")
    def test_tokenize_with_default_model(
        self, mock_print, mock_config, mock_registry_cls
    ):
        """Test tokenization with default model."""
        # Mock config
        mock_config.get_default_model.return_value = "cl100k_base"
        mock_config.get_delimiter.return_value = "⎮"
        mock_config.add_model_to_history.return_value = None

        # Mock registry
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = True
        mock_result = {
            "token_strings": ["Hello", " world"],
            "token_ids": [123, 456],
            "token_count": 2,
        }
        mock_registry.tokenize.return_value = mock_result

        # Call function
        run_tokenize("Hello world", None, "json")

        # Verify default model was used
        mock_registry.tokenize.assert_called_once_with("Hello world", "cl100k_base")
        # Verify output was printed
        self.assertTrue(mock_print.called)

    @patch("tokker.cli.output.formats.print")
    def test_output_count_format_only_counts(self, mock_print):
        """Ensure 'count' output prints only counts JSON."""
        result = {
            "delimited_text": "Hello⎮ world",
            "token_strings": ["Hello", " world"],
            "token_ids": [1, 2],
            "token_count": 2,
            "word_count": 2,
            "char_count": 11,
            "pivot": {"Hello": 1, " world": 1},
        }
        # Reuse the existing format function via import to keep test fast
        from tokker.cli.output.formats import format_and_print_output

        format_and_print_output(result, "count", "⎮")
        out = "".join(c.args[0] for c in mock_print.call_args_list if c.args)
        data = json.loads(out)
        self.assertEqual(set(data.keys()), {"token_count", "word_count", "char_count"})


class TestMainFunction(unittest.TestCase):
    """Test cases for main CLI entry point."""

    @patch("tokker.cli.tokenize.run_list_models")
    @patch("sys.argv", ["tok", "--models"])
    def test_main_models(self, mock_handle_models):
        """Test main function with models command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_models.assert_called_once()

    @patch("tokker.cli.tokenize.run_set_default_model")
    @patch("sys.argv", ["tok", "--model-default", "gpt2"])
    def test_main_model_default(self, mock_handle_default):
        """Test main function with model default command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_default.assert_called_once_with("gpt2")

    @patch("tokker.cli.tokenize.run_show_history")
    @patch("sys.argv", ["tok", "--history"])
    def test_main_history(self, mock_handle_history):
        """Test main function with history command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_history.assert_called_once()

    @patch("tokker.cli.tokenize.run_clear_history")
    @patch("sys.argv", ["tok", "--history-clear"])
    def test_main_history_clear(self, mock_handle_clear):
        """Test main function with history clear command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_clear.assert_called_once()

    @patch("tokker.cli.tokenize.run_tokenize")
    @patch("sys.argv", ["tok", "Hello world"])
    def test_main_tokenize(self, mock_handle_tokenize):
        """Test main function with tokenize command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_tokenize.assert_called_once_with("Hello world", None, "json")

    @patch("tokker.cli.tokenize.run_tokenize")
    @patch("sys.stdin", StringIO("Hello from stdin"))
    @patch("sys.argv", ["tok"])
    def test_main_stdin(self, mock_handle_tokenize):
        """Test main function with stdin input."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_tokenize.assert_called_once_with("Hello from stdin", None, "json")


class TestGoogleAuthMapping(unittest.TestCase):
    """Tests for Google auth guidance driven by tokker.__main__ mapping."""

    def _import_main_fresh(self):
        import importlib
        import sys

        if "tokker.__main__" in sys.modules:
            del sys.modules["tokker.__main__"]
        return importlib.import_module("tokker.__main__")

    @patch("sys.argv", ["tok", "Hello", "--model", "gemini-2.5-flash"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("auth error: compute_tokens failed"),
    )
    def test_google_no_adc_no_gcloud_guidance(self, _cli_main):
        """No ADC and no gcloud: expect guidance printed and non-zero exit."""
        main_module = self._import_main_fresh()
        from io import StringIO

        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected_lines = [
                msg.MSG_GOOGLE_AUTH_HEADER,
                msg.MSG_GOOGLE_AUTH_GUIDE_LINE,
            ] + list(msg.MSG_GOOGLE_AUTH_STEPS)
            expected = "\n".join(expected_lines) + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--model", "gemini-2.5-flash"])
    @patch("tokker.cli.tokenize.main", side_effect=RuntimeError("auth error"))
    def test_google_adc_missing_file_or_generic_error_still_guidance(self, _cli_main):
        """When ADC is unspecified and a generic auth error occurs, show guidance."""
        main_module = self._import_main_fresh()
        from io import StringIO

        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected_lines = [
                msg.MSG_GOOGLE_AUTH_HEADER,
                msg.MSG_GOOGLE_AUTH_GUIDE_LINE,
            ] + list(msg.MSG_GOOGLE_AUTH_STEPS)
            expected = "\n".join(expected_lines) + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--model", "gemini-2.5-flash"])
    @patch(
        "tokker.cli.tokenize.main", side_effect=RuntimeError("original google error")
    )
    def test_google_error_with_model_hint_still_guidance(self, _cli_main):
        """Even if error text is opaque, model hint should trigger guidance."""
        main_module = self._import_main_fresh()
        from io import StringIO

        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected_lines = [
                msg.MSG_GOOGLE_AUTH_HEADER,
                msg.MSG_GOOGLE_AUTH_GUIDE_LINE,
            ] + list(msg.MSG_GOOGLE_AUTH_STEPS)
            expected = "\n".join(expected_lines) + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
