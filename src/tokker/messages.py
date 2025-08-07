#!/usr/bin/env python3
from enum import Enum

# ---- Separators and headings ----
SEP_MAIN = "============"
SEP_SUB = "------------"

HDR_HISTORY = "History:\n"

# ---- Links and guidance ----
GOOGLE_AUTH_GUIDE = "https://github.com/igoakulov/tokker/blob/main/google-auth-guide.md"
MSG_AUTH_REQUIRED = f"\nAuth setup required   ->   {GOOGLE_AUTH_GUIDE}"

# Centralized Google auth guidance (used by __main__.py)
MSG_GOOGLE_AUTH_HEADER = "Google auth setup required for Gemini (takes ~3 mins):"
# This line keeps the URL centralized and formatted
MSG_GOOGLE_AUTH_GUIDE_URL = f"  {GOOGLE_AUTH_GUIDE}"

# ---- Messages ----
MSG_DEFAULT_SET = "Default model set to: {model}"
MSG_DEFAULT_SET_PROVIDER = "Default model set to: {model} ({provider})"
MSG_CONFIG_SAVED_TO = "Configuration saved to: {path}"
# Provider-aware validation message for -D/--model-default when unresolved
MSG_DEFAULT_MODEL_UNSUPPORTED_FMT = (
    "Model '{model}' not found with installed providers: {providers}."
)

MSG_HISTORY_EMPTY = "History empty.\n"
MSG_HISTORY_CLEARED = "History cleared."
MSG_HISTORY_ALREADY_EMPTY = "History is already empty."
MSG_OPERATION_CANCELLED = "Operation cancelled."
# Output/format errors
MSG_UNKNOWN_OUTPUT_FORMAT_FMT = (
    "Unknown output format: {value}. Allowed: json, plain, count, pivot"
)

# CLI/global error and hint messages (for __main__.py mapping)
# Unknown/invalid model hints
MSG_DEP_HINT_HEADING = "Install model providers:"

# Install commands
CMD_INSTALL_ALL = "pip install 'tokker[all]'"
CMD_INSTALL_TIKTOKEN = "pip install 'tokker[tiktoken]'"
CMD_INSTALL_GOOGLE = "pip install 'tokker[google-genai]'"
CMD_INSTALL_TRANSFORMERS = "pip install 'tokker[transformers]'"

# Human-readable hint lines, built from commands
MSG_DEP_HINT_ALL = f"  {CMD_INSTALL_ALL}                 - all at once"
MSG_DEP_HINT_TIKTOKEN = f"  {CMD_INSTALL_TIKTOKEN}            - OpenAI"
MSG_DEP_HINT_GOOGLE = f"  {CMD_INSTALL_GOOGLE}        - Google"
MSG_DEP_HINT_TRANSFORMERS = f"  {CMD_INSTALL_TRANSFORMERS}        - HuggingFace"

# Config/FS and generic error formats
MSG_FILESYSTEM_ERROR_FMT = "Filesystem error: {err}"
MSG_CONFIG_ERROR_FMT = "Configuration error: {err}"
MSG_UNEXPECTED_ERROR_FMT = "Unexpected error: {err}"

# ---- BYOM (HuggingFace) instructions ----
BYOM_INSTRUCTIONS = [
    "  1. Go to   ->   https://huggingface.co/models?library=transformers",
    "  2. Search any model with TRANSFORMERS library support",
    "  3. Copy its `USER/MODEL` into your command like:\n",
]

# ---- OpenAI tokenizer descriptions ----
OPENAI_DESCRIPTIONS = {
    "o200k_base": "- for GPT-4o, o-family (o1, o3, o4)",
    "cl100k_base": "- for GPT-3.5 (late), GPT-4",
    "p50k_base": "- for GPT-3.5 (early)",
    "p50k_edit": "- for GPT-3 edit models (text-davinci, code-davinci)",
    "r50k_base": "- for GPT-3 base models (davinci, curie, babbage, ada)",
}

# ---- Example HuggingFace models for BYOM ----
BYOM_EXAMPLE_MODELS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "moonshotai/Kimi-K2-Instruct",
    "deepseek-ai/DeepSeek-R1",
    "google/gemma-3n-E4B-it",
    "google-bert/bert-base-uncased",
    "meta-llama/Meta-Llama-3.1-405B",
    "mistralai/Devstral-Small-2507",
]


# ---- Output formats (CLI-level enum) ----
class OutputFormat(Enum):
    JSON = "json"
    PLAIN = "plain"
    COUNT = "count"
    PIVOT = "pivot"

    @classmethod
    def values(cls) -> list[str]:
        """Return list of string values for argparse choices."""
        return [m.value for m in cls]


# ---- Helper to standardize missing-dependency exceptions ----
def missing_dep_error(package: str) -> RuntimeError:
    """
    Create a RuntimeError that mimics ImportError's canonical message:
      "No module named 'PACKAGE'"
    This exact text is recognized by the centralized error handler, which maps
    it to MSG_DEP_HINT_FMT for user-friendly install guidance.
    """
    return RuntimeError(f"No module named '{package}'")
