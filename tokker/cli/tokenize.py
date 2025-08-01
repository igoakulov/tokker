#!/usr/bin/env python3
"""
Main CLI module for tokenization commands.

Provides the main CLI interface for tokenizing text and managing model settings.
"""

import argparse
import sys
import subprocess
from typing import Optional, Dict, Any
from .config import config
from datetime import datetime
from .utils import format_plain_output, format_json_output, count_words
from tokker.models.registry import get_registry
from tokker import __description__


def _build_cli_output_structure(tokenization_result: Dict[str, Any], text: str, delimiter: str) -> Dict[str, Any]:
    """
    Build the CLI output structure from the provider-agnostic base result.

    Args:
        tokenization_result: Result returned by the model registry (provider-agnostic)
        text: Original input text
        delimiter: Delimiter to join token strings

    Returns:
        Dictionary with CLI output fields consumed by formatters
    """
    token_strings = tokenization_result['token_strings']

    # Create pivot dictionary for token frequency analysis
    pivot = {}
    for token_text in token_strings:
        if token_text:
            pivot[token_text] = pivot.get(token_text, 0) + 1

    return {
        "converted": delimiter.join(token_strings),
        "token_strings": token_strings,
        "token_ids": tokenization_result['token_ids'],
        "token_count": tokenization_result['token_count'],
        "word_count": count_words(text),
        "char_count": len(text),
        "pivot": pivot
    }


def _format_and_print_output(result: Dict[str, Any], output_format: str, delimiter: str) -> None:
    """
    Format and print the CLI output based on the selected output format.

    Args:
        result: CLI output structure dictionary
        output_format: One of 'json', 'plain', 'count', 'table'
        delimiter: Delimiter for plain output
    """
    if output_format == "json":
        # Exclude pivot/model/provider from JSON output
        json_result = {
            "converted": result["converted"],
            "token_strings": result["token_strings"],
            "token_ids": result["token_ids"],
            "token_count": result["token_count"],
            "word_count": result["word_count"],
            "char_count": result["char_count"]
        }
        print(format_json_output(json_result))
    elif output_format == "plain":
        plain_output = format_plain_output(result, delimiter)
        print(plain_output)
    elif output_format == "count":
        # Exclude model/provider from count summary
        count_summary = {
            "token_count": result["token_count"],
            "word_count": result["word_count"],
            "char_count": result["char_count"]
        }
        print(format_json_output(count_summary))
    elif output_format == "pivot":
        # Print pivot only, sorted by highest count first, then token A–Z
        pivot = result.get("pivot", {})
        items = sorted(pivot.items(), key=lambda kv: (-kv[1], kv[0]))
        table_obj = {k: v for k, v in items}
        print(format_json_output(table_obj))


def _handle_google_auth_failure(original_error: Exception) -> None:
    """Handle Google provider auth failures by running ADC flow and guiding user to retry."""
    print(f"Error from Google: {original_error}", file=sys.stderr)
    print("Attempting to set up Google Application Default Credentials...", file=sys.stderr)
    print("Running: gcloud auth application-default login", file=sys.stderr)

    try:
        proc = subprocess.run(
            ["gcloud", "auth", "application-default", "login"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
    except FileNotFoundError:
        print("gcloud CLI not found. Install the Google Cloud SDK or use a service account JSON:", file=sys.stderr)
        print("https://cloud.google.com/sdk/docs/install", file=sys.stderr)
        print("Alternatively, set GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/key.json", file=sys.stderr)
    except Exception as ge:
        print(f"Failed to run gcloud auth flow: {ge}", file=sys.stderr)

    print("When authentication completes, please re-run your tok command.", file=sys.stderr)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Tokker - {__description__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
------------
Examples:
  echo 'Hello world' | tok
  tok 'Hello world'
  tok 'Hello world' -m deepseek-ai/DeepSeek-R1
  tok 'Hello world' -m gemini-2.5-pro -o count
  tok 'Hello world' -o pivot
  tok -D cl100k_base
------------
Google auth setup   →   https://github.com/igoakulov/tokker/blob/main/tokker/google-auth-guide.md
        """
    )

    # Text input for tokenization (positional argument)
    parser.add_argument(
        "text",
        nargs="?",
        help="text to tokenize (or read from stdin if not provided)"
    )

    # Model selection
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="model to use (overrides default)"
    )

    # Output format
    parser.add_argument(
        "-o", "--output",
        type=str,
        choices=["json", "plain", "count", "pivot"],
        default="json",
        help="output format (default: json)"
    )

    # Set default model
    parser.add_argument(
        "-D", "--model-default",
        type=str,
        help="set default model"
    )

    # List available models
    parser.add_argument(
        "-M", "--models",
        action="store_true",
        help="list all available models"
    )

    # Model history commands
    parser.add_argument(
        "-H", "--history",
        action="store_true",
        help="show history of used models"
    )

    parser.add_argument(
        "-X", "--history-clear",
        action="store_true",
        help="clear history"
    )

    return parser


def handle_model_default(model: str) -> None:
    """Handle setting the default model."""
    try:
        # Use registry to validate model and get description
        registry = get_registry()
        if not registry.validate_model(model):
            print(f"Invalid model: {model}.", file=sys.stderr)
            sys.exit(1)

        # Get model info for display
        available_models = registry.list_models()
        model_info = next((m for m in available_models if m['name'] == model), None)

        # Set the default model
        config.set_default_model(model)

        # Display confirmation without tick and without description; no surrounding blank lines
        if model_info:
            provider = model_info['provider']
            print(f"Default model set to: {model} ({provider})")
        else:
            print(f"Default model set to: {model}")
        print(f"Configuration saved to: {config.config_file}")
    except (OSError, RuntimeError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_models() -> None:
    """Handle listing available models."""
    try:
        registry = get_registry()
        available_models = registry.list_models()

        # New template format output (always use template format)
        print("============")

        # Separate OpenAI, Google, and HuggingFace models
        openai_models = []
        google_models = []
        huggingface_models = []

        for model in available_models:
            if model['provider'] == 'OpenAI':
                openai_models.append(model)
            elif model['provider'] == 'Google':
                google_models.append(model)
            elif model['provider'] == 'HuggingFace':
                huggingface_models.append(model)

        # Sort OpenAI models alphabetically
        openai_models.sort(key=lambda x: x['name'])

        # OpenAI section (match template headings and spacing)
        print("OpenAI:\n")
        # Static descriptions per updated template
        openai_descriptions = {
            "cl100k_base": "used in GPT-3.5 (late), GPT-4",
            "o200k_base": "used in GPT-4o, o-family (o1, o3, o4)",
            "p50k_base": "used in GPT-3.5 (early)",
            "p50k_edit": "used in GPT-3 edit models (text-davinci, code-davinci)",
            "r50k_base": "used in GPT-3 base models (davinci, curie, babbage, ada)"
        }

        # model name padded to 20 chars, 2-space indent, description follows
        for model in openai_models:
            name = model['name']
            description = openai_descriptions.get(name, None)
            if description:
                print(f"  {name:<22}{description}")

        print("------------")
        print("Google:\n")
        # Show the five supported Gemini models without descriptions in preferred order
        google_preferred_order = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        google_available = {m['name'] for m in google_models}
        for gm in google_preferred_order:
            if gm in google_available:
                print(f"  {gm}")
        print("\nAuth setup required   ->   https://github.com/igoakulov/tokker/blob/main/tokker/google-auth-guide.md")
        print("------------")
        print("HuggingFace (BYOM - Bring You Own Model):\n")
        print("  1. Go to   ->   https://huggingface.co/models?library=transformers")
        print("  2. Search any model with TRANSFORMERS library support")
        print("  3. Copy its `USER/MODEL` into your command like:\n")

        # Example models per template
        example_models = [
            "deepseek-ai/DeepSeek-R1",
            "google-bert/bert-base-uncased",
            "google/gemma-3n-E4B-it",
            "meta-llama/Meta-Llama-3.1-405B",
            "mistralai/Devstral-Small-2507",
            "moonshotai/Kimi-K2-Instruct",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        ]
        for example in example_models:
            print(f"  {example}")
        # footer double line for end of page
        print("============")

    except (OSError, RuntimeError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)



def handle_history() -> None:
    """Handle displaying model usage history."""
    try:
        history = config.load_model_history()

        # page header
        print("============")
        print("History:\n")

        if not history:
            print("History empty.\n")
            print("============")
            return

        # list entries (most recent first)
        for entry in history:
            model_name = entry.get('model', 'unknown')
            timestamp = entry.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    print(f"  {model_name:<32}{formatted_time}")
                except (ValueError, TypeError):
                    print(f"  {model_name}")
            else:
                print(f"  {model_name}")

        print("============")

    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_history_clear() -> None:
    """Handle clearing model usage history with confirmation."""
    try:
        history = config.load_model_history()

        if not history:
            print("History is already empty.")
            return

        # Ask user for confirmation
        response = input(f"Clear model history ({len(history)} entries)? [y/N]: ").strip().lower()

        if response in ('y', 'yes'):
            config.clear_model_history()
            print("History cleared.")
        else:
            print("History not cleared.")

    except KeyboardInterrupt:
        print("Operation cancelled.")
        sys.exit(1)
    except (EOFError, OSError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _resolve_selected_model(model: Optional[str]) -> str:
    """Resolve the selected model using CLI argument or default config."""
    return model if model else config.get_default_model()


def _validate_model_or_exit(registry, model: str) -> None:
    """Validate a model via the registry or exit with an error."""
    if not registry.validate_model(model):
        print(f"Invalid model: {model}.", file=sys.stderr)
        sys.exit(1)


def _google_auth_guidance() -> None:
    """Print standard Google auth guidance block to stderr."""
    print("Google auth setup required for Gemini (takes 3 mins):", file=sys.stderr)
    print("  https://github.com/igoakulov/tokker/blob/main/tokker/google-auth-guide.md", file=sys.stderr)
    print("-----------", file=sys.stderr)
    print("Alternatively, sign in via browser:", file=sys.stderr)
    print("  1. Install this: https://cloud.google.com/sdk/docs/install", file=sys.stderr)
    print("  2. Run this command:", file=sys.stderr)
    print("     gcloud auth application-default login", file=sys.stderr)


def _handle_google_error(registry, model: str, original_error: Exception) -> None:
    """Handle provider-specific Google errors with ADC/gcloud fallback."""
    try:
        provider_name = registry.get_model(model).library_name
    except Exception:
        provider_name = None

    if provider_name != "Google":
        raise original_error

    import os
    import shutil

    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_path:
        if not os.path.isfile(adc_path):
            print(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {adc_path}", file=sys.stderr)
            _google_auth_guidance()
            sys.exit(1)
        # ADC file exists; surface original Google error
        print(str(original_error), file=sys.stderr)
        sys.exit(1)

    gcloud_path = shutil.which("gcloud")
    if gcloud_path:
        print("Attempting browser sign-in via gcloud (Application Default Credentials)...", file=sys.stderr)
        try:
            import subprocess
            proc = subprocess.run(
                [gcloud_path, "auth", "application-default", "login"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                print(proc.stdout, file=sys.stderr, end="")
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
        except Exception as ge:
            print(f"gcloud sign-in attempt failed: {ge}", file=sys.stderr)
        sys.exit(1)

    _google_auth_guidance()
    sys.exit(1)


def _perform_tokenization(registry, text: str, model: str) -> Dict[str, Any]:
    """Perform tokenization with the registry and handle Google-specific errors."""
    try:
        return registry.tokenize(text, model)
    except Exception as err:
        _handle_google_error(registry, model, err)
        # _handle_google_error always exits for Google errors; re-raise otherwise
        raise


def handle_tokenize(text: str, model: Optional[str], output_format: str) -> None:
    """Handle tokenizing text with the specified or default model."""
    try:
        selected_model = _resolve_selected_model(model)
        registry = get_registry()
        _validate_model_or_exit(registry, selected_model)

        tokenization_result = _perform_tokenization(registry, text, selected_model)

        # Track model usage in history
        config.add_model_to_history(selected_model)

        # Build CLI output structure from base result
        delimiter = config.get_delimiter()
        result = _build_cli_output_structure(tokenization_result, text, delimiter)

        # Format and display output (keep script-friendly: no leading separator)
        _format_and_print_output(result, output_format, delimiter)

    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle models command
    if args.models:
        handle_models()
        return 0

    # Handle history command
    if args.history:
        handle_history()
        return 0

    # Handle history clear command
    if args.history_clear:
        handle_history_clear()
        return 0

    # Handle set default model command
    if args.model_default:
        handle_model_default(args.model_default)
        return 0

    # Determine text source: command line argument or stdin
    text = None
    if args.text is not None:
        text = args.text
    elif not sys.stdin.isatty():
        # Read from stdin (piped input)
        text = sys.stdin.read().strip()

    # Handle tokenization
    if text is not None and text:
        handle_tokenize(text, args.model, args.output)
        return 0

    # No text provided from either source
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
