#!/usr/bin/env python3
"""
Main CLI module for tokenization commands.

Provides the main CLI interface for tokenizing text and managing tokenizer settings.
"""

import argparse
import sys
from typing import Optional
from .config import config
from .utils import format_plain_output, format_summary_output, format_json_output, count_words
from tokker.tokenizers.registry import get_registry


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Tok CLI - Tokenize text using tiktoken and HuggingFace tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tok 'Hello world'
  echo 'Hello world' | tok
  tok 'Hello world' --tokenizer cl100k_base
  tok 'Hello world' --tokenizer gpt2
  tok 'Hello world' --format plain
  tok --tokenizer-list
  tok --tokenizer-list --format json
  tok --tokenizer-default cl100k_base
        """
    )

    # Text input for tokenization (positional argument)
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to tokenize (or read from stdin if not provided)"
    )

    # Tokenizer selection
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer to use (overrides default). Use --tokenizer-list to see available options"
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "plain", "summary", "table"],
        default="json",
        help="Output format (default: json)"
    )

    # Set default tokenizer
    parser.add_argument(
        "--tokenizer-default",
        type=str,
        help="Set the default tokenizer in configuration. Use --tokenizer-list to see available options"
    )

    # List available tokenizers
    parser.add_argument(
        "--tokenizer-list",
        action="store_true",
        help="List all available tokenizers with descriptions"
    )

    return parser


def handle_tokenizer_default(tokenizer: str) -> None:
    """Handle setting the default tokenizer."""
    try:
        # Use registry to validate tokenizer and get description
        registry = get_registry()
        if not registry.validate_tokenizer(tokenizer):
            available = [t['name'] for t in registry.list_tokenizers()]
            print(f"Error: Tokenizer '{tokenizer}' not found.", file=sys.stderr)
            print(f"Available tokenizers: {', '.join(sorted(available))}", file=sys.stderr)
            sys.exit(1)

        # Get tokenizer info for display
        tokenizers = registry.list_tokenizers()
        tokenizer_info = next((t for t in tokenizers if t['name'] == tokenizer), None)

        # Set the default tokenizer
        config.set_default_tokenizer(tokenizer)

        # Display confirmation with library and description
        if tokenizer_info:
            library = tokenizer_info['library']
            description = tokenizer_info['description']
            print(f"✓ Default tokenizer set to: {tokenizer} ({library}) - {description}")
        else:
            print(f"✓ Default tokenizer set to: {tokenizer}")

        print(f"Configuration saved to: {config.config_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_tokenizer_list(output_format: str) -> None:
    """Handle listing available tokenizers."""
    try:
        registry = get_registry()
        tokenizers = registry.list_tokenizers()

        if output_format == "json":
            # JSON format output
            import json
            print(json.dumps(tokenizers, indent=2, ensure_ascii=False))
        else:
            # Table format output (default for --tokenizer-list)
            # Group tokenizers by family, merge tiktoken and HuggingFace
            families = {}

            for tokenizer in tokenizers:
                name = tokenizer['name']
                library = tokenizer['library']
                description = tokenizer['description']

                # Determine family based on tokenizer name and description
                family = _get_tokenizer_family(name, description)

                if family not in families:
                    families[family] = []

                families[family].append({
                    'name': name,
                    'library': library,
                    'description': description
                })

            # Define the desired family order
            family_order = ["DeepSeek", "GPT", "LLaMA", "Qwen", "Other"]

            # Sort families by the defined order, and within each family sort tokenizers A-Z
            for family_name in family_order:
                if family_name not in families:
                    continue

                family_tokenizers = sorted(families[family_name], key=lambda x: x['name'])

                # Format family header - "Other" doesn't get " Family" suffix
                if family_name == "Other":
                    header = f"\n{family_name}:"
                    separator = "=" * (len(family_name) + 1)
                else:
                    header = f"\n{family_name} Family:"
                    separator = "=" * (len(family_name) + 8)

                print(header)
                print(separator)

                for tokenizer in family_tokenizers:
                    name = tokenizer['name']
                    library = tokenizer['library']
                    description = tokenizer['description']

                    # Format the line with proper spacing
                    name_col = f"  {name:<30}"
                    lib_col = f"({library})"
                    print(f"{name_col} {lib_col:<10} — {description}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _get_tokenizer_family(name: str, description: str) -> str:
    """Determine the family name for a tokenizer based on its name and description."""
    # GPT family (both tiktoken and HuggingFace)
    if any(x in name.lower() for x in ['gpt', 'o200k', 'cl100k', 'p50k', 'r50k']) or 'GPT' in description:
        return "GPT"

    # LLaMA family
    if 'llama' in name.lower() or 'LLaMA' in description:
        return "LLaMA"

    # Qwen family
    if 'qwen' in name.lower() or 'Qwen' in description:
        return "Qwen"

    # DeepSeek family
    if 'deepseek' in name.lower() or 'DeepSeek' in description:
        return "DeepSeek"

    # Everything else goes to "Other"
    return "Other"


def handle_tokenize(text: str, tokenizer: Optional[str], output_format: str) -> None:
    """Handle tokenizing text."""
    try:
        # Determine which tokenizer to use
        if tokenizer:
            selected_tokenizer = tokenizer
        else:
            selected_tokenizer = config.get_default_tokenizer()

        # Use registry system for tokenization
        registry = get_registry()

        # Validate tokenizer using registry
        if not registry.validate_tokenizer(selected_tokenizer):
            available = [t['name'] for t in registry.list_tokenizers()]
            raise ValueError(
                f"Invalid tokenizer: {selected_tokenizer}. "
                f"Available options: {', '.join(sorted(available))}"
            )

        # Tokenize using registry
        registry_result = registry.tokenize(text, selected_tokenizer)

        # Convert to legacy format for backward compatibility
        delimiter = config.get_delimiter()
        token_strings = registry_result['token_strings']

        # Create pivot dictionary for word frequency
        pivot = {}
        for token_text in token_strings:
            if token_text:
                pivot[token_text] = pivot.get(token_text, 0) + 1

        # Build result in legacy format
        result = {
            "converted": delimiter.join(token_strings),
            "token_strings": token_strings,
            "token_ids": registry_result['token_ids'],
            "token_count": registry_result['token_count'],
            "word_count": count_words(text),
            "char_count": len(text),
            "pivot": pivot,
            "tokenizer": registry_result['tokenizer'],
            "library": registry_result['library']  # New field from registry
        }

        # Format output
        if output_format == "json":
            print(format_json_output(result))
        elif output_format == "plain":
            delimiter = config.get_delimiter()
            plain_output = format_plain_output(result, delimiter)
            print(plain_output)
        elif output_format == "summary":
            summary = format_summary_output(result)
            print(format_json_output(summary))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle tokenizer list command
    if args.tokenizer_list:
        # For --tokenizer-list, use table format by default, JSON only if explicitly requested
        list_format = "json" if "--format" in sys.argv and args.format == "json" else "table"
        handle_tokenizer_list(list_format)
        return 0

    # Handle set default tokenizer command
    if args.tokenizer_default:
        handle_tokenizer_default(args.tokenizer_default)
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
        handle_tokenize(text, args.tokenizer, args.format)
        return 0

    # No text provided from either source
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
