"""
Unit tests for CLI functionality aligned with the new structure.

Tests the CLI argument parsing, command handling, and output formatting.
"""

import unittest
import json
import os
from unittest.mock import Mock, patch
from io import StringIO

from tokker.cli.arguments import build_argument_parser
from tokker.cli.tokenize import main as cli_main
from tokker.cli.commands.list_models import run_list_models
from tokker.cli.commands.set_default_model import run_set_default_model
from tokker.cli.commands.tokenize_text import run_tokenize


class TestCLIParser(unittest.TestCase):
    """Test cases for CLI argument parsing."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = build_argument_parser()

    def test_basic_tokenize_args(self):
        """Test basic tokenization arguments."""
        # Test with text argument
        args = self.parser.parse_args(['Hello world'])
        self.assertEqual(args.text, 'Hello world')
        self.assertIsNone(args.model)
        self.assertEqual(args.output, 'json')

        # Test with model
        args = self.parser.parse_args(['Hello world', '--model', 'gpt2'])
        self.assertEqual(args.text, 'Hello world')
        self.assertEqual(args.model, 'gpt2')

        # Test with output format
        args = self.parser.parse_args(['Hello world', '--output', 'plain'])
        self.assertEqual(args.text, 'Hello world')
        self.assertEqual(args.output, 'plain')

    def test_models_args(self):
        """Test models list arguments."""
        args = self.parser.parse_args(['--models'])
        self.assertTrue(args.models)

    def test_pivot_output_arg(self):
        """Test that pivot is an accepted output format."""
        args = self.parser.parse_args(['Hello', '--output', 'pivot'])
        self.assertEqual(args.text, 'Hello')
        self.assertEqual(args.output, 'pivot')

    def test_model_default_args(self):
        """Test model default arguments."""
        args = self.parser.parse_args(['--model-default', 'cl100k_base'])
        self.assertEqual(args.model_default, 'cl100k_base')

    def test_no_args(self):
        """Test parsing with no arguments (stdin mode)."""
        args = self.parser.parse_args([])
        self.assertIsNone(args.text)
        self.assertFalse(args.models)
        self.assertIsNone(args.model_default)


class TestListModelsCommand(unittest.TestCase):
    """Test cases for models command handling."""

    @patch('tokker.cli.commands.list_models.print')
    @patch('tokker.cli.commands.list_models.ModelRegistry')
    def test_models_output(self, mock_registry_cls, mock_print):
        """Test models output."""
        # Mock registry and models
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.list_models.side_effect = lambda provider=None: [
            {'name': 'cl100k_base', 'provider': 'OpenAI'},
            {'name': 'gemini-2.5-pro', 'provider': 'Google'},
            {'name': 'gpt2', 'provider': 'HuggingFace'},
        ] if provider is None else (
            [{'name': 'cl100k_base', 'provider': 'OpenAI'}] if provider == "OpenAI" else
            [{'name': 'gemini-2.5-pro', 'provider': 'Google'}] if provider == "Google" else
            [{'name': 'gpt2', 'provider': 'HuggingFace'}]
        )

        # Call function
        run_list_models()

        # Verify print calls
        self.assertTrue(mock_print.called)
        # Ensure Google section header is printed
        printed = "\n".join(call.args[0] for call in mock_print.call_args_list if call.args)
        self.assertIn("Google:", printed)


class TestSetDefaultModelCommand(unittest.TestCase):
    """Test cases for model default command handling."""

    @patch('tokker.cli.commands.set_default_model.config')
    @patch('tokker.cli.commands.set_default_model.ModelRegistry')
    @patch('builtins.print')
    def test_valid_model_default(self, mock_print, mock_registry_cls, mock_config):
        """Test setting valid default model."""
        # Mock registry
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = True
        mock_registry.list_models.return_value = [
            {'name': 'cl100k_base', 'provider': 'OpenAI'}
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

    @patch('tokker.cli.commands.set_default_model.sys.exit')
    @patch('tokker.cli.commands.set_default_model.ModelRegistry')
    def test_invalid_model_default(self, mock_registry_cls, mock_exit):
        """Test setting invalid default model."""
        # Mock registry
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.is_model_supported.return_value = False
        mock_registry.list_models.return_value = [
            {'name': 'cl100k_base', 'provider': 'OpenAI'}
        ]

        # Call function
        run_set_default_model("invalid_model")

        # Verify exit was called
        mock_exit.assert_called_once_with(1)


class TestTokenizeCommand(unittest.TestCase):
    """Test cases for tokenize command handling."""

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('tokker.cli.commands.tokenize_text.config')
    @patch('tokker.cli.output.formats.print')
    def test_tokenize_with_specific_model(self, mock_print, mock_config, mock_registry_cls):
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

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('tokker.cli.commands.tokenize_text.config')
    @patch('tokker.cli.output.formats.print')
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
        printed = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
        data = json.loads(printed)
        # Expect pivot counts for tokens present in token_strings (spaces may be present)
        self.assertIn("foo", data)
        self.assertEqual(data["foo"], 2)

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('tokker.cli.commands.tokenize_text.config')
    @patch('tokker.cli.output.formats.print')
    def test_tokenize_with_default_model(self, mock_print, mock_config, mock_registry_cls):
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

    @patch('tokker.cli.commands.tokenize_text._validate_model_or_exit', side_effect=SystemExit(1))
    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    def test_tokenize_invalid_model(self, mock_registry_cls, _):
        """Tokenization with invalid model short-circuits before tokenization."""
        # Mock registry and ensure tokenize would fail if reached
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.tokenize.side_effect = AssertionError("tokenize should not be called for invalid model")
        mock_registry.is_model_supported.return_value = False

        # Call function and expect SystemExit due to validation helper
        with self.assertRaises(SystemExit):
            run_tokenize("Hello world", "invalid_model", "json")

        # Ensure we short-circuited before tokenization
        mock_registry.tokenize.assert_not_called()


class TestMainFunction(unittest.TestCase):
    """Test cases for main CLI entry point."""

    @patch('tokker.cli.tokenize.run_list_models')
    @patch('sys.argv', ['tok', '--models'])
    def test_main_models(self, mock_handle_models):
        """Test main function with models command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_models.assert_called_once()

    @patch('tokker.cli.tokenize.run_set_default_model')
    @patch('sys.argv', ['tok', '--model-default', 'gpt2'])
    def test_main_model_default(self, mock_handle_default):
        """Test main function with model default command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_default.assert_called_once_with("gpt2")

    @patch('tokker.cli.tokenize.run_tokenize')
    @patch('sys.argv', ['tok', 'Hello world'])
    def test_main_tokenize(self, mock_handle_tokenize):
        """Test main function with tokenize command."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_tokenize.assert_called_once_with("Hello world", None, "json")

    @patch('tokker.cli.tokenize.run_tokenize')
    @patch('sys.stdin', StringIO('Hello from stdin'))
    @patch('sys.argv', ['tok'])
    def test_main_stdin(self, mock_handle_tokenize):
        """Test main function with stdin input."""
        result = cli_main()
        self.assertEqual(result, 0)
        mock_handle_tokenize.assert_called_once_with("Hello from stdin", None, "json")


class TestGoogleAuthFlow(unittest.TestCase):
    """Unit tests for Google provider auth checks and fallbacks."""

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('shutil.which', return_value=None)
    @patch.dict(os.environ, {}, clear=True)
    def test_google_no_adc_no_gcloud_guidance(self, _which, mock_registry_cls):
        """When no ADC and no gcloud, show guidance and exit."""
        # Arrange registry to identify Google provider and raise on tokenize
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_provider.return_value.NAME = "Google"
        mock_registry.is_model_supported.return_value = True
        mock_registry.tokenize.side_effect = RuntimeError("auth error")

        # Act + Assert
        with self.assertRaises(SystemExit):
            run_tokenize("hi", "gemini-2.5-pro", "json")

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('subprocess.run')
    @patch('shutil.which', return_value="/usr/bin/gcloud")
    @patch.dict(os.environ, {}, clear=True)
    def test_google_no_adc_with_gcloud_attempts_login(self, _which, mock_run, mock_registry_cls):
        """When gcloud exists and no ADC, attempt browser sign-in and exit."""
        # Arrange registry to identify Google provider and raise on tokenize
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_provider.return_value.NAME = "Google"
        mock_registry.is_model_supported.return_value = True
        mock_registry.tokenize.side_effect = RuntimeError("auth error")

        # Mock gcloud run result
        mock_proc = Mock()
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        # Act + Assert
        with self.assertRaises(SystemExit):
            run_tokenize("hi", "gemini-2.5-pro", "json")

        # Ensure gcloud login attempted
        mock_run.assert_called_once()

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('os.path.isfile', return_value=False)
    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/no/such/file.json"}, clear=True)
    def test_google_adc_missing_file(self, _isfile, mock_registry_cls):
        """When ADC path is set but missing, show guidance and exit."""
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_provider.return_value.NAME = "Google"
        mock_registry.is_model_supported.return_value = True
        mock_registry.tokenize.side_effect = RuntimeError("auth error")

        with self.assertRaises(SystemExit):
            run_tokenize("hi", "gemini-2.5-pro", "json")

    @patch('tokker.cli.commands.tokenize_text.ModelRegistry')
    @patch('os.path.isfile', return_value=True)
    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/key.json"}, clear=True)
    def test_google_adc_file_exists_surface_error(self, _isfile, mock_registry_cls):
        """When ADC file exists but tokenization fails, surface original error and exit."""
        mock_registry = Mock()
        mock_registry_cls.return_value = mock_registry
        mock_registry.get_provider.return_value.NAME = "Google"
        mock_registry.is_model_supported.return_value = True
        mock_registry.tokenize.side_effect = RuntimeError("original google error")

        with self.assertRaises(SystemExit):
            run_tokenize("hi", "gemini-2.5-pro", "json")


if __name__ == '__main__':
    unittest.main()
