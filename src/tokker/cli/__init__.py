#!/usr/bin/env python3
from .tokenize import main

def main_entry():
    """Main entry point for the CLI."""
    return main()

# For compatibility with different entry point styles
__all__ = ["main", "main_entry"]
