#!/usr/bin/env python3
import sys

from tokker.cli import strings
from tokker.cli.config import config, ConfigError


def run_clear_history() -> None:
    """Clear saved model usage history."""
    try:
        history = config.load_history()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not history:
        print(strings.MSG_HISTORY_ALREADY_EMPTY)
        return

    try:
        config.clear_history()
    except ConfigError as e:
        print(strings.MSG_OPERATION_CANCELLED)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(strings.MSG_HISTORY_CLEARED)
