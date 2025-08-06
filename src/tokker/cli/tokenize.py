#!/usr/bin/env python3
import sys
import tokker.runtime as _tokker_runtime  # noqa: F401  # Ensure Transformers logging is configured early
from tokker.cli.arguments import build_argument_parser
from tokker.cli.commands.list_models import run_list_models
from tokker.cli.commands.set_default_model import run_set_default_model
from tokker.cli.commands.show_history import run_show_history
from tokker.cli.commands.clear_history import run_clear_history
from tokker.cli.commands.tokenize_text import run_tokenize
from tokker.exceptions import ModelError, ModelNotFoundError, MissingDependencyError


def main() -> int:
    """Main CLI entry point: parse args and dispatch to commands."""
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        if getattr(args, "models", False):
            run_list_models()
            return 0

        if getattr(args, "history", False):
            run_show_history()
            return 0

        if getattr(args, "history_clear", False):
            run_clear_history()
            return 0

        if getattr(args, "model_default", None):
            run_set_default_model(args.model_default)
            return 0

        # Determine text source: CLI arg or stdin
        text = None
        if getattr(args, "text", None) is not None:
            text = args.text
        elif not sys.stdin.isatty():
            text = sys.stdin.read().strip()

        if text:
            run_tokenize(
                text,
                getattr(args, "model", None),
                getattr(args, "output", "json"),
            )
            return 0

        parser.print_help()
        return 1

    except ModelNotFoundError as e:
        # Unknown model; suggest listing available models
        print(str(e), file=sys.stderr)
        print("Run 'tok -M' to list available models.", file=sys.stderr)
        return 1
    except MissingDependencyError as e:
        # Missing optional provider dependency; provide extras guidance
        print(str(e), file=sys.stderr)
        return 1
    except ModelError as e:
        # Other model/provider-related failures (load/tokenization/etc.)
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        # Safety net for non-model unexpected errors (no traceback)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
