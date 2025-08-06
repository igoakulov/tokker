#!/usr/bin/env python3
import sys
# Apply centralized runtime/environment setup early
import tokker.runtime as _tokker_runtime  # noqa: F401  # Ensure Transformers logging is configured before CLI loads

from tokker.cli.tokenize import main as cli_main

def main():
    out = 1
    try:
        out = cli_main()
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
    return out

if __name__ == "__main__":
    sys.exit(main())
