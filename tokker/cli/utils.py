#!/usr/bin/env python3
"""
Utility functions for tokenization and text processing.

Provides core tokenization logic using tiktoken and various counting utilities.
"""

import tiktoken
from typing import Dict, Any
import re
import json


def get_tokenizer(tokenizer_name: str) -> tiktoken.Encoding:
    """
    Get tiktoken tokenizer by name.

    Args:
        tokenizer_name: Name of the tokenizer ('o200k_base' or 'cl100k_base')

    Returns:
        tiktoken.Encoding instance

    Raises:
        ValueError: If tokenizer_name is invalid
    """
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        raise ValueError(f"Invalid tokenizer '{tokenizer_name}': {e}")


def tokenize_text(text: str, tokenizer_name: str = "o200k_base", delimiter: str = "⏐") -> Dict[str, Any]:
    """
    Tokenize text using specified tokenizer.

    Args:
        text: Input text to tokenize
        tokenizer_name: Name of tokenizer to use
        delimiter: Delimiter to use for joining token strings

    Returns:
        Dictionary containing tokenization results
    """
    tokenizer = get_tokenizer(tokenizer_name)

    # Get token IDs
    token_ids = tokenizer.encode(text)

    # Get token strings
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

    # Create pivot dictionary for word frequency
    pivot = {}
    for token_text in tokens:
        if token_text:
            pivot[token_text] = pivot.get(token_text, 0) + 1

    return {
        "converted": delimiter.join(tokens),
        "token_strings": tokens,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "word_count": count_words(text),
        "char_count": len(text),
        "pivot": pivot,
        "tokenizer": tokenizer_name
    }


def count_words(text: str) -> int:
    """
    Count words in text using simple regex-based approach.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    if not text.strip():
        return 0

    # Split on whitespace and count non-empty segments
    words = re.findall(r'\S+', text)
    return len(words)


def count_characters(text: str) -> int:
    """
    Count characters in text.

    Args:
        text: Input text

    Returns:
        Number of characters
    """
    return len(text)


def count_tokens(text: str, tokenizer_name: str = "o200k_base") -> int:
    """
    Count tokens in text using specified tokenizer.

    Args:
        text: Input text
        tokenizer_name: Name of tokenizer to use

    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text))


def format_plain_output(tokenization_result: Dict[str, Any], delimiter: str = "⏐") -> str:
    """
    Format tokenization result as plain delimited text.

    Args:
        tokenization_result: Dictionary from tokenize_text()
        delimiter: Delimiter character to use

    Returns:
        Plain text string with tokens separated by delimiter
    """
    tokens = tokenization_result["token_strings"]
    return delimiter.join(tokens)


def format_summary_output(tokenization_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create summary output from tokenization result.

    Args:
        tokenization_result: Dictionary from tokenize_text()

    Returns:
        Compact summary dictionary
    """
    return {
        "token_count": tokenization_result["token_count"],
        "word_count": tokenization_result["word_count"],
        "char_count": tokenization_result["char_count"],
        "tokenizer": tokenization_result["tokenizer"]
    }


def validate_tokenizer(tokenizer_name: str, valid_tokenizers: set) -> None:
    """
    Validate tokenizer name against valid options.

    Args:
        tokenizer_name: Name to validate
        valid_tokenizers: Set of valid tokenizer names

    Raises:
        ValueError: If tokenizer_name is invalid
    """
    if tokenizer_name not in valid_tokenizers:
        raise ValueError(
            f"Invalid tokenizer: {tokenizer_name}. "
            f"Valid options: {', '.join(sorted(valid_tokenizers))}"
        )


def format_json_output(data: Dict[str, Any]) -> str:
    """
    Format JSON with properties on separate lines but arrays on single lines.

    Args:
        data: Dictionary to format as JSON

    Returns:
        JSON string with compact arrays and indented properties
    """
    def compact_list_formatter(obj, current_indent=0, indent_amount=2):
        """Custom formatter that keeps arrays compact."""
        if isinstance(obj, dict):
            if not obj:
                return "{}"

            indent_str = " " * current_indent
            next_indent_str = " " * (current_indent + indent_amount)

            items = []
            for key, value in obj.items():
                formatted_value = compact_list_formatter(value, current_indent + indent_amount, indent_amount)
                items.append(f'{next_indent_str}"{key}": {formatted_value}')

            return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            # Format list items compactly on one line
            formatted_items = []
            for item in obj:
                if isinstance(item, str):
                    formatted_items.append(json.dumps(item, ensure_ascii=False))
                else:
                    formatted_items.append(json.dumps(item))

            return "[" + ", ".join(formatted_items) + "]"

        else:
            return json.dumps(obj, ensure_ascii=False)

    return compact_list_formatter(data)
