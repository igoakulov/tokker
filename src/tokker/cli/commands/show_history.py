#!/usr/bin/env python3
import sys
from datetime import datetime

from tokker.cli import strings
from tokker.cli.config import config, ConfigError


def run_show_history() -> None:
    """Display model usage history."""
    try:
        history = config.load_history()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(strings.SEP_MAIN)
    print(strings.HDR_HISTORY)

    if not history:
        print(strings.MSG_HISTORY_EMPTY)
        print(strings.SEP_MAIN)
        return

    # List entries (most recent first)
    for entry in history:
        model_name = entry.get("model", "unknown")
        timestamp = entry.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(str(timestamp))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                print(f"  {model_name:<32}{formatted_time}")
            except (ValueError, TypeError):
                print(f"  {model_name}")
        else:
            print(f"  {model_name}")

    print(strings.SEP_MAIN)
