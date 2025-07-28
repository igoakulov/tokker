#!/usr/bin/env python3
"""
CLI module for Tokker.

Provides the command-line interface for tokenization operations.
"""

from .tokenize import main

def main_entry():
    """Main entry point for the CLI."""
    return main()

# For compatibility with different entry point styles
__all__ = ["main", "main_entry"]
