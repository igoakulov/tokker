#!/usr/bin/env python3
"""
Main CLI module for tokenization commands.

Provides the main CLI interface for tokenizing text and managing model settings.
"""

import argparse
import sys
from typing import Optional
from .config import config
from datetime import datetime
from .utils import format_plain_output, format_summary_output, format_json_output, count_words
from tokker.models.registry import get_registry
from tokker import __description__


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Tokker - {__description__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tok 'Hello world'
  echo 'Hello world' | tok
  tok 'Hello world' --model cl100k_base
  tok 'Hello world' --model gpt2
  tok 'Hello world' --output count
  tok --models
  tok --model-default cl100k_base
  tok --history
  tok --history-clear
        """
    )

    # Text input for tokenization (positional argument)
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to tokenize (or read from stdin if not provided)"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (overrides default). Use --models to see available options"
    )

    # Output format
    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "plain", "count", "table"],
        default="json",
        help="Output format (default: json)"
    )

    # Set default model
    parser.add_argument(
        "--model-default",
        type=str,
        help="Set the default model in configuration. Use --models to see available options"
    )

    # List available models
    parser.add_argument(
        "--models",
        action="store_true",
        help="List all available models with descriptions"
    )

    # Model history commands
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show history of used models, with most recent on top"
    )

    parser.add_argument(
        "--history-clear",
        action="store_true",
        help="Clear model usage history"
    )

    return parser


def handle_model_default(model: str) -> None:
    """Handle setting the default model."""
    try:
        # Use registry to validate model and get description
        registry = get_registry()
        if not registry.validate_model(model):
            print(f"Error: Model '{model}' not found.", file=sys.stderr)
            print("Use `tok --models` to see all available models.", file=sys.stderr)
            sys.exit(1)

        # Get model info for display
        available_models = registry.list_models()
        model_info = next((m for m in available_models if m['name'] == model), None)

        # Set the default model
        config.set_default_model(model)

        # Display confirmation with provider and description
        if model_info:
            provider = model_info['provider']
            description = model_info['description']
            print(f"✓ Default model set to: {model} ({provider}) - {description}")
        else:
            print(f"✓ Default model set to: {model}")

        print(f"Configuration saved to: {config.config_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_models() -> None:
    """Handle listing available models."""
    try:
        registry = get_registry()
        available_models = registry.list_models()

        # New template format output (always use template format)

        # Separate OpenAI and HuggingFace models
        openai_models = []
        huggingface_models = []

        for model in available_models:
            if model['provider'] == 'OpenAI':
                openai_models.append(model)
            elif model['provider'] == 'HuggingFace':
                huggingface_models.append(model)

        # Sort OpenAI models alphabetically
        openai_models.sort(key=lambda x: x['name'])

        # OpenAI section
        print("--- OpenAI ---")
        # Static text as per PRD template - not using deprecated description field
        openai_descriptions = {
            "cl100k_base": "used in GPT-3.5 (late), GPT-4",
            "o200k_base": "used in GPT-4o, o-family (o1, o3, o4)",
            "p50k_base": "used in GPT-3.5 (early)",
            "p50k_edit": "used in GPT-3 edit models for text and code (text-davinci, code-davinci)",
            "r50k_base": "used in GPT-3 base models (davinci, curie, babbage, ada)"
        }

        for model in openai_models:
            name = model['name']
            description = openai_descriptions.get(name, "OpenAI model")
            print(f"{name:<15} ->   {description}")

        print()

        # HuggingFace section
        print("--- HuggingFace ---")
        print("BYOM - Bring You Own Model:")
        print("1. Go to   ->   https://huggingface.co/models?library=transformers")
        print("2. Search any model with TRANSFORMERS library support")
        print("3. Copy its `USER/MODEL-NAME` into your command:")
        print()

        # Show example HuggingFace models
        example_models = [
            "deepseek-ai/DeepSeek-R1",
            "google-bert/bert-base-uncased",
            "google/gemma-3n-E4B-it",
            "meta-llama/Meta-Llama-3.1-405B",
            "mistralai/Devstral-Small-2507",
            "moonshotai/Kimi-K2-Instruct",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "Etc."
        ]

        for example in example_models:
            print(example)

        print()

        # Related Commands footer
        print("--- Related Commands ---")
        print("`--model-default 'model-name'` to set default model")
        print("`--history` to view all models you have used")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)



def handle_history() -> None:
    """Handle displaying model usage history."""
    try:
        history = config.load_model_history()

        if not history:
            print("Your history is empty.")
            print()
            print("--- Related Commands ---")
            print("`--model-default 'model-name'` to set default model")
            print("`--history-clear` to clear your history")
            return

        print("Model History (most recent first):")
        print()
        for entry in history:
            model_name = entry.get('model', 'unknown')
            timestamp = entry.get('timestamp', '')
            # Format timestamp for better readability
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    print(f"  {model_name:<30} (used: {formatted_time})")
                except (ValueError, TypeError):
                    print(f"  {model_name}")
            else:
                print(f"  {model_name}")

        print()
        print("--- Related Commands ---")
        print("`--model-default 'model-name'` to set default model")
        print("`--history-clear` to clear your history")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_history_clear() -> None:
    """Handle clearing model usage history with confirmation."""
    try:
        history = config.load_model_history()

        if not history:
            print("Model history is already empty.")
            return

        # Ask user for confirmation
        response = input(f"Clear model history ({len(history)} entries)? [y/N]: ").strip().lower()

        if response in ('y', 'yes'):
            config.clear_model_history()
            print("✓ Model history cleared.")
        else:
            print("Model history not cleared.")

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_tokenize(text: str, model: Optional[str], output_format: str) -> None:
    """Handle tokenizing text with the specified or default model."""
    try:
        # Determine which model to use
        if model:
            selected_model = model
        else:
            selected_model = config.get_default_model()

        # Use registry system for tokenization
        registry = get_registry()

        # Validate model using registry
        if not registry.validate_model(selected_model):
            raise ValueError(
                f"Invalid model: {selected_model}. "
                "Use `tok --models` to see all available models."
            )

        # Perform tokenization using registry
        tokenization_result = registry.tokenize(text, selected_model)

        # Track model usage in history
        config.add_model_to_history(selected_model)

        # Convert to legacy format for backward compatibility
        delimiter = config.get_delimiter()
        token_strings = tokenization_result['token_strings']

        # Create pivot dictionary for token frequency analysis
        pivot = {}
        for token_text in token_strings:
            if token_text:
                pivot[token_text] = pivot.get(token_text, 0) + 1

        # Build result in legacy format
        result = {
            "converted": delimiter.join(token_strings),
            "token_strings": token_strings,
            "token_ids": tokenization_result['token_ids'],
            "token_count": tokenization_result['token_count'],
            "word_count": count_words(text),
            "char_count": len(text),
            "pivot": pivot,
            "model": tokenization_result['model'],
            "provider": tokenization_result['provider']  # Provider field from registry
        }

        # Format and display output
        if output_format == "json":
            print(format_json_output(result))
        elif output_format == "plain":
            delimiter = config.get_delimiter()
            plain_output = format_plain_output(result, delimiter)
            print(plain_output)
        elif output_format == "count":
            count_summary = format_summary_output(result)
            print(format_json_output(count_summary))

    except Exception as e:
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
