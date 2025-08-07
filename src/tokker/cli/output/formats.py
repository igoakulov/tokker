#!/usr/bin/env python3
import json
from typing import Any
from tokker import messages


def format_and_print_output(
    result: dict[str, Any], output_format: str, delimiter: str
) -> None:
    """
    Print the result in a specific format: json, plain, count, pivot.
    """
    member = messages.OutputFormat.__members__.get(str(output_format).upper())
    if member is None:
        raise ValueError(f"Unknown output format: {output_format}")

    handlers = {
        messages.OutputFormat.JSON: lambda: _handle_json(result),
        messages.OutputFormat.PLAIN: lambda: _handle_plain(result, delimiter),
        messages.OutputFormat.COUNT: lambda: _handle_count(result),
        messages.OutputFormat.PIVOT: lambda: _handle_pivot(result),
    }

    # Map is exhaustive for our enum; direct dispatch without secondary guard
    handlers[member]()


# ---- Handlers (ordered by common usage and enum order) ----


def _handle_json(result: dict[str, Any]) -> None:
    json_result = {
        "delimited_text": result.get("delimited_text", ""),
        "token_strings": result.get("token_strings", []),
        "token_ids": result.get("token_ids", []),
        "token_count": result.get("token_count", 0),
        "word_count": result.get("word_count", 0),
        "char_count": result.get("char_count", 0),
    }
    print(_format_json_output(json_result))
    pass


def _handle_plain(result: dict[str, Any], delimiter: str) -> None:
    print(_format_plain_output(result, delimiter))
    pass


def _handle_count(result: dict[str, Any]) -> None:
    count_summary = {
        "token_count": result.get("token_count", 0),
        "word_count": result.get("word_count", 0),
        "char_count": result.get("char_count", 0),
    }
    print(_format_json_output(count_summary))
    pass


def _handle_pivot(result: dict[str, Any]) -> None:
    pivot = result.get("pivot", {}) or {}
    # Sorted by frequency desc, then lexicographically
    items = sorted(pivot.items(), key=lambda kv: (-kv[1], kv[0]))
    table_obj = {k: v for k, v in items}
    print(_format_json_output(table_obj))
    pass


# ---- Low-level formatters ----


def _format_plain_output(
    tokenization_result: dict[str, Any], delimiter: str = "⏐"
) -> str:
    """Join token strings with the provided delimiter."""
    tokens = tokenization_result.get("token_strings", []) or []
    return delimiter.join(tokens)


def _format_json_output(data: dict) -> str:
    """Pretty-compact JSON printer: dicts pretty, lists one line, unicode preserved."""

    def compact(obj: Any, indent: int = 0, step: int = 2) -> str:
        # Dicts: multi-line with indentation
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            pad, pad_next = " " * indent, " " * (indent + step)
            items = (
                f'{pad_next}"{k}": {compact(v, indent + step, step)}'
                for k, v in obj.items()
            )
            return "{\n" + ",\n".join(items) + f"\n{pad}" + "}"
        # Lists: always one line
        if isinstance(obj, list):
            return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in obj) + "]"
        # Scalars
        return json.dumps(obj, ensure_ascii=False)

    return compact(data)
