#!/usr/bin/env python3
import argparse
from tokker.cli import strings

def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for tokker."""
    parser = argparse.ArgumentParser(
        description="Tokker: a fast local-first CLI tokenizer with all the best models in one place",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{strings.SEP_MAIN}
Examples:
  echo 'Hello world' | tok
  tok 'Hello world'
  tok 'Hello world' -m deepseek-ai/DeepSeek-R1
  tok 'Hello world' -m gemini-2.5-pro -o count
  tok 'Hello world' -o pivot
  tok -D cl100k_base
{strings.SEP_MAIN}
Google auth setup   →   {strings.GOOGLE_AUTH_GUIDE}
        """
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="text to tokenize (or read from stdin if not provided)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="model to use (overrides default)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        choices=strings.OutputFormat.values(),
        default=strings.OutputFormat.JSON.value,
        help="output format (default: json)"
    )

    parser.add_argument(
        "-D", "--model-default",
        type=str,
        help="set default model"
    )

    parser.add_argument(
        "-M", "--models",
        action="store_true",
        help="list all models"
    )

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
