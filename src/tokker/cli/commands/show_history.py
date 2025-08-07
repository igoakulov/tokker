#!/usr/bin/env python3
from tokker import messages
from tokker.cli.config import config


def run_show_history() -> None:
    """Display model usage history."""
    history = config.load_history()

    print(messages.SEP_MAIN)
    print(messages.HDR_HISTORY)

    if not history:
        print(messages.MSG_HISTORY_EMPTY)
        print(messages.SEP_MAIN)
        return

    # List entries (most recent first)
    for entry in history:
        model_name = entry.get("model", "unknown")
        timestamp = entry.get("timestamp", "")
        ts = str(timestamp).replace("T", " ")[:16]
        print(f"  {model_name:<32}{ts}")

    print(messages.SEP_MAIN)
