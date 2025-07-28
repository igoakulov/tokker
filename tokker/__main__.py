#!/usr/bin/env python3
"""
Main entry point for the Tokker CLI tool.

This module serves as the entry point when running `python -m tokker`.
It handles the initial setup and delegates to the CLI modules.
"""

import sys
from pathlib import Path

def main():
    """Main entry point for the CLI."""
    # Ensure the tokker package is importable regardless of CWD
    package_dir = Path(__file__).parent.absolute()
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

    # Import and run the CLI
    try:
        from cli.tokenize import main as cli_main
        return cli_main()
    except ImportError as e:
        print(f"Error importing CLI modules: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
