#!/usr/bin/env python3
from tokker import messages
from tokker.cli.config import config


def run_clear_history() -> None:
    """Clear saved model usage history."""
    history = config.load_history()

    if not history:
        print(messages.MSG_HISTORY_ALREADY_EMPTY)
        return

    config.clear_history()

    print(messages.MSG_HISTORY_CLEARED)
