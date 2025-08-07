#!/usr/bin/env python3
"""
Smoke test script for tokker CLI tool.

This script performs basic functionality tests to ensure
that both tiktoken and HuggingFace models work correctly
after installation.
"""

import subprocess
import json
import tempfile
import os
import sys
import unittest
from typing import List


def run_command(cmd: List[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        # Use local module instead of global tok command
        # Route through tokker.__main__ for centralized error handling
        if cmd[0] == "tok":
            cmd = [sys.executable, "-m", "tokker"] + cmd[1:]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def _has_tiktoken() -> bool:
    try:
        import tiktoken as _tiktoken  # type: ignore  # noqa

        return True
    except Exception:
        return False


def _has_transformers() -> bool:
    try:
        import transformers as _transformers  # type: ignore  # noqa

        return True
    except Exception:
        return False


class SmokeTests(unittest.TestCase):
    def test_models_list(self):
        """Test the --models command."""
        exit_code, stdout, stderr = run_command(["tok", "--models"])
        self.assertEqual(exit_code, 0, f"Models list failed: {stderr}")
        # Headings are standardized; just ensure sections are present
        self.assertIn("OpenAI", stdout, "Models output missing OpenAI section")
        self.assertIn("Google", stdout, "Models output missing Google section")
        self.assertTrue(
            ("Auth setup required" in stdout and "google-auth-guide" in stdout)
            or ("Auth required" in stdout and "google-auth-guide" in stdout),
            "Google auth guide link missing",
        )
        self.assertIn(
            "HuggingFace", stdout, "Models output missing HuggingFace section"
        )
        # Maintain provider order in output
        openai_idx = stdout.index("OpenAI")
        google_idx = stdout.index("Google")
        hf_idx = stdout.index("HuggingFace")
        self.assertTrue(
            openai_idx < google_idx < hf_idx,
            "Providers not in expected order (OpenAI, Google, HuggingFace)",
        )
        self.assertIn("cl100k_base", stdout, "Models output missing expected model")

    @unittest.skipUnless(_has_tiktoken(), "Skipping: tiktoken extra not installed")
    def test_tiktoken_model(self):
        """Test tiktoken model functionality."""
        exit_code, stdout, stderr = run_command(
            ["tok", "Hello world", "--model", "cl100k_base"]
        )
        self.assertEqual(exit_code, 0, f"Tiktoken tokenization failed: {stderr}")

        result = json.loads(stdout)
        expected_fields = [
            "delimited_text",
            "token_strings",
            "token_ids",
            "token_count",
            "word_count",
            "char_count",
        ]
        for field in expected_fields:
            self.assertIn(field, result, f"Missing field '{field}' in tiktoken result")
        self.assertEqual(
            result["token_count"], len(result["token_strings"]), "Token count mismatch"
        )

    @unittest.skipUnless(
        _has_transformers(), "Skipping: transformers (hf) extra not installed"
    )
    def test_huggingface_model(self):
        """Test HuggingFace model functionality."""
        exit_code, stdout, stderr = run_command(
            ["tok", "Hello world", "--model", "gpt2"]
        )
        self.assertEqual(exit_code, 0, f"HuggingFace tokenization failed: {stderr}")

        result = json.loads(stdout)
        expected_fields = [
            "delimited_text",
            "token_strings",
            "token_ids",
            "token_count",
            "word_count",
            "char_count",
        ]
        for field in expected_fields:
            self.assertIn(
                field, result, f"Missing field '{field}' in HuggingFace result"
            )
        self.assertEqual(
            result["token_count"], len(result["token_strings"]), "Token count mismatch"
        )

    @unittest.skipUnless(_has_tiktoken(), "Skipping: tiktoken extra not installed")
    def test_output_formats(self):
        """Test different output formats."""
        exit_code, stdout, stderr = run_command(
            ["tok", "Hello world", "--model", "cl100k_base", "--output", "plain"]
        )
        self.assertEqual(exit_code, 0, f"Plain format failed: {stderr}")
        self.assertIn("⎮", stdout, "Plain format missing delimiter")

        exit_code, stdout, stderr = run_command(
            ["tok", "Hello world", "--model", "cl100k_base", "--output", "count"]
        )
        self.assertEqual(exit_code, 0, f"Count format failed: {stderr}")
        result = json.loads(stdout)
        self.assertIn("token_count", result, "Count format missing token_count")

        exit_code, stdout, stderr = run_command(
            ["tok", "foo foo bar", "--model", "cl100k_base", "--output", "pivot"]
        )
        self.assertEqual(exit_code, 0, f"Pivot format failed: {stderr}")
        pivot = json.loads(stdout)
        self.assertTrue(
            isinstance(pivot, dict) and bool(pivot),
            "Pivot should be a non-empty object",
        )
        self.assertTrue(
            ("foo" in pivot) or (" foo" in pivot), "Pivot missing expected token 'foo'"
        )
        total = sum(pivot.values())
        self.assertGreaterEqual(total, 3, "Pivot counts do not sum to expected total")

    @unittest.skipUnless(_has_tiktoken(), "Skipping: tiktoken extra not installed")
    def test_stdin_input(self):
        """Test stdin input functionality."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Hello from stdin")
            temp_file = f.name
        try:
            exit_code, stdout, stderr = run_command(
                [
                    "sh",
                    "-c",
                    f"cat {temp_file} | python -m tokker.cli.tokenize --model cl100k_base",
                ]
            )
            self.assertEqual(exit_code, 0, f"Stdin input failed: {stderr}")
            result = json.loads(stdout)
            self.assertIn("token_count", result, "Stdin result missing token_count")
        finally:
            os.unlink(temp_file)

    @unittest.skipUnless(_has_tiktoken(), "Skipping: tiktoken extra not installed")
    def test_model_default(self):
        """Test model default functionality."""
        exit_code, stdout, stderr = run_command(
            ["tok", "--model-default", "cl100k_base"]
        )
        self.assertEqual(exit_code, 0, f"Setting default model failed: {stderr}")
        self.assertIn(
            "Default model set to: cl100k_base",
            stdout,
            "Default model confirmation missing",
        )

        exit_code, stdout, stderr = run_command(["tok", "Hello world"])
        self.assertEqual(exit_code, 0, f"Using default model failed: {stderr}")
        result = json.loads(stdout)
        self.assertIn("token_count", result, "Default model run missing token_count")

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        exit_code, stdout, stderr = run_command(
            ["tok", "Hello world", "--model", "nonexistent-model"]
        )
        self.assertNotEqual(exit_code, 0, "Invalid model should have failed")
        # Accept current not-found message produced by ModelNotFoundError
        combined = (stderr or "") + (stdout or "")
        self.assertTrue(
            ("not found" in combined.lower())
            or ("run 'tok -m'" in combined.lower())
            or ("invalid model" in combined.lower()),
            "Invalid model message not found",
        )

        exit_code, stdout, stderr = run_command(
            ["tok", "--model-default", "nonexistent-model"]
        )
        # Now standardized: invalid default model triggers standardized error and non-zero exit
        self.assertNotEqual(
            exit_code,
            0,
            "Setting unknown default model should fail with standardized error",
        )
        self.assertIn(
            "Model 'nonexistent-model' not found with installed providers:", stderr
        )

    @unittest.skipUnless(_has_tiktoken(), "Skipping: tiktoken extra not installed")
    def test_history_functionality(self):
        """Test history and history-clear functionality."""
        run_command(["tok", "--history-clear"])

        exit_code, stdout, stderr = run_command(
            ["tok", "Hello world", "--model", "cl100k_base"]
        )
        self.assertEqual(exit_code, 0, f"Model usage for history failed: {stderr}")

        exit_code, stdout, stderr = run_command(["tok", "--history"])
        self.assertEqual(exit_code, 0, f"History command failed: {stderr}")
        self.assertIn("cl100k_base", stdout, "History missing expected model")
        self.assertIn("History:", stdout, "History output missing header")

        exit_code, stdout, stderr = run_command(["tok", "--history-clear"])
        self.assertEqual(exit_code, 0, f"History clear failed: {stderr}")
        self.assertTrue(
            ("History is already empty." in stdout) or ("cleared" in stdout.lower()),
            "History clear confirmation missing",
        )

        exit_code, stdout, stderr = run_command(["tok", "--history"])
        self.assertEqual(exit_code, 0, f"History check after clear failed: {stderr}")
        self.assertTrue(
            ("History empty." in stdout)
            or ("Your history is empty" in stdout)
            or ("Your history is empty" in stderr),
            "History not cleared properly",
        )


if __name__ == "__main__":
    unittest.main()
