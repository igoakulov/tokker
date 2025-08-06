#!/usr/bin/env python3
import json
import sys
from typing import Any

from tokker.cli.strings import OutputFormat


def format_and_print_output(result: dict[str, Any], output_format: str, delimiter: str) -> None:
    """
    Print the result in a specific format: json, plain, count, pivot.
    """
    member = OutputFormat.__members__.get(str(output_format).upper())
    if member is None:
        print(f"Unknown output format: {output_format}", file=sys.stderr)
        return

    if member is OutputFormat.JSON:
        json_result = {
            "delimited_text": result.get("delimited_text", ""),
            "token_strings": result.get("token_strings", []),
            "token_ids": result.get("token_ids", []),
            "token_count": result.get("token_count", 0),
            "word_count": result.get("word_count", 0),
            "char_count": result.get("char_count", 0),
        }
        print(_format_json_output(json_result))
        return

    if member is OutputFormat.PLAIN:
        print(_format_plain_output(result, delimiter))
        return

    if member is OutputFormat.COUNT:
        count_summary = {
            "token_count": result.get("token_count", 0),
            "word_count": result.get("word_count", 0),
            "char_count": result.get("char_count", 0),
        }
        print(_format_json_output(count_summary))
        return

    if member is OutputFormat.PIVOT:
        pivot = result.get("pivot", {}) or {}
        # Sorted by frequency desc, then lexicographically
        items = sorted(pivot.items(), key=lambda kv: (-kv[1], kv[0]))
        table_obj = {k: v for k, v in items}
        print(_format_json_output(table_obj))
        return


def _format_plain_output(tokenization_result: dict[str, Any], delimiter: str = "⏐") -> str:
    """Join token strings with the provided delimiter."""
    tokens = tokenization_result.get("token_strings", []) or []
    return delimiter.join(tokens)


def _format_json_output(data: dict) -> str:
    """Pretty-compact JSON printer that preserves non-ASCII characters."""
    def compact(obj: Any, current_indent: int = 0, indent_amount: int = 2) -> str:
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            indent_str = " " * current_indent
            next_indent_str = " " * (current_indent + indent_amount)
            items = []
            for k, v in obj.items():
                pretty_v = compact(v, current_indent + indent_amount, indent_amount)
                items.append(f'{next_indent_str}"{k}": {pretty_v}')
            return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"
        if isinstance(obj, list):
            if not obj:
                return "[]"
            parts = [
                json.dumps(x, ensure_ascii=False) if isinstance(x, str)
                else json.dumps(x)
                for x in obj
            ]
            return "[" + ", ".join(parts) + "]"
        return json.dumps(obj, ensure_ascii=False)

    return compact(data)
