#!/usr/bin/env python3
"""
Shared, side-effect free helpers used by CLI and error handling.

Functions:
- get_arg_value: extract a flag's value from argv (supports split and --flag=value syntax)
- is_google_model: predicate for Google Gemini model name prefixes
- get_installed_providers: detect which providers are actually usable in this environment
- get_install_commands: map providers to install commands (strings from tokker.messages)
"""

from collections.abc import Iterable

from tokker import messages


def get_arg_value(argv: Iterable[str], *flags: str) -> str | None:
    """
    Return the value associated with a CLI flag from argv.

    Useful for tokker flags like:
      -m / --model             → model name to tokenize with
      -D / --model-default     → model name to set as default

    Supported syntaxes:
      1) Split form:        ["tok", "-m", "cl100k_base"]      → "cl100k_base"
      2) --flag=value form: ["tok", "--model=cl100k_base"]    → "cl100k_base"

    Args:
        argv: An iterable of argument tokens (e.g., sys.argv)
        *flags: One or more flag strings to match (e.g., "-m", "--model")

    Returns:
        The flag value (str) if found; otherwise None.
    """
    try:
        xs = list(argv)
        flagset = set(flags)
        for i, a in enumerate(xs):
            # Split form: -f VALUE or --flag VALUE
            if a in flagset:
                if i + 1 < len(xs):
                    return xs[i + 1]
                continue
            # --flag=value form
            for f in flagset:
                if a.startswith(f + "="):
                    return a.split("=", 1)[1]
        return None
    except Exception:
        # Best-effort: never raise from argv parsing
        return None


def is_google_model(model: str | None) -> bool:
    """
    Return True if the provided model name looks like a Google Gemini model.

    Heuristic (conservative):
      - Name starts with "gemini-"
      - Or name starts with "models/gemini-"

    Args:
        model: The model name or None

    Returns:
        True if the name resembles a Google Gemini model; otherwise False.
    """
    if not model:
        return False
    return model.startswith("gemini-") or model.startswith("models/gemini-")


def get_installed_providers() -> set[str]:
    """
    Detect which provider integrations are actually usable in this environment
    by checking whether their optional dependencies can be imported.

    Providers:
      - "OpenAI"     requires `tiktoken`
      - "HuggingFace" requires `transformers`
      - "Google"      requires `google-genai` (HttpOptions import)

    Returns:
        A set like {"OpenAI", "Google"} of installed providers.
    """
    installed: set[str] = set()

    # OpenAI (tiktoken)
    try:
        import tiktoken  # type: ignore  # noqa: F401

        installed.add("OpenAI")
    except Exception:
        pass

    # HuggingFace (transformers)
    try:
        import transformers  # type: ignore  # noqa: F401

        installed.add("HuggingFace")
    except Exception:
        pass

    # Google (google-genai)
    try:
        from google.genai.types import HttpOptions  # type: ignore  # noqa: F401

        installed.add("Google")
    except Exception:
        pass

    return installed


def get_install_commands(providers: Iterable[str]) -> list[str]:
    """
    Map provider names to their install commands (strings). Order is stable.

    Input providers should be a subset of:
      "OpenAI", "Google", "HuggingFace"

    Returns:
      A list of raw command strings. For example, given ["Google", "HuggingFace"],
      returns:
        [
          "pip install 'tokker[google-genai]'",
          "pip install 'tokker[transformers]'",
        ]
    """
    out: list[str] = []
    name_to_cmd = {
        "OpenAI": messages.CMD_INSTALL_TIKTOKEN,
        "Google": messages.CMD_INSTALL_GOOGLE,
        "HuggingFace": messages.CMD_INSTALL_TRANSFORMERS,
    }
    for name in ("OpenAI", "Google", "HuggingFace"):
        if name in providers:
            out.append(name_to_cmd[name])
    return out
