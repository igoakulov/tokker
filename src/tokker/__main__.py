#!/usr/bin/env python3
import sys

# Apply centralized runtime/environment setup early
import tokker.runtime as _tokker_runtime  # noqa

from tokker.cli.tokenize import main as cli_main
from tokker.error_handler import handle_exception


def main():
    try:
        return cli_main()
    except Exception as e:
        return handle_exception(e, sys.argv or [])


if __name__ == "__main__":
    sys.exit(main())
