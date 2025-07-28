#!/usr/bin/env python3
"""
Main CLI module for tokenization commands.

Provides the main CLI interface for tokenizing text and managing tokenizer settings.
"""

import argparse
import sys
from typing import Optional
from .config import config
from .utils import tokenize_text, format_plain_output, format_summary_output, validate_tokenizer, format_json_output


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Tok CLI - Tokenize text using OpenAI tiktoken",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tok 'Hello world'
  echo 'Hello world' | tok
  tok 'Hello world' --tokenizer cl100k_base
  tok 'Hello world' --format plain
  tok --set-default-tokenizer cl100k_base
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
        choices=["o200k_base", "cl100k_base"],
        help="Tokenizer to use (overrides default)"
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "plain", "summary"],
        default="json",
        help="Output format (default: json)"
    )

    # Set default tokenizer
    parser.add_argument(
        "--set-default-tokenizer",
        type=str,
        choices=["o200k_base", "cl100k_base"],
        help="Set the default tokenizer in configuration"
    )

    return parser


def handle_set_default_tokenizer(tokenizer: str) -> None:
    """Handle setting the default tokenizer."""
    try:
        config.set_default_tokenizer(tokenizer)
        print(f"✓ Default tokenizer set to: {tokenizer}")
        print(f"Configuration saved to: {config.config_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_tokenize(text: str, tokenizer: Optional[str], output_format: str) -> None:
    """Handle tokenizing text."""
    try:
        # Determine which tokenizer to use
        if tokenizer:
            selected_tokenizer = tokenizer
        else:
            selected_tokenizer = config.get_default_tokenizer()

        # Validate tokenizer
        validate_tokenizer(selected_tokenizer, config.get_valid_tokenizers())

        # Tokenize the text
        delimiter = config.get_delimiter()
        result = tokenize_text(text, selected_tokenizer, delimiter)

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

    # Handle set default tokenizer command
    if args.set_default_tokenizer:
        handle_set_default_tokenizer(args.set_default_tokenizer)
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
