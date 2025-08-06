#!/usr/bin/env python3
from enum import Enum

# ---- Separators and headings ----
SEP_MAIN = "============"
SEP_SUB = "------------"

HDR_OPENAI = "OpenAI:\n"
HDR_GOOGLE = "Google:\n"
HDR_HF_BYOM = "HuggingFace (BYOM - Bring You Own Model):\n"
HDR_HISTORY = "History:\n"

# ---- Links and guidance ----
GOOGLE_AUTH_GUIDE = "https://github.com/igoakulov/tokker/blob/main/google-auth-guide.md"
MSG_AUTH_REQUIRED = f"\nAuth setup required   ->   {GOOGLE_AUTH_GUIDE}"

# ---- Messages ----
MSG_INVALID_MODEL = "Invalid model: {model}."
MSG_DEFAULT_SET = "Default model set to: {model}"
MSG_DEFAULT_SET_PROVIDER = "Default model set to: {model} ({provider})"
MSG_CONFIG_SAVED_TO = "Configuration saved to: {path}"

MSG_HISTORY_EMPTY = "History empty.\n"
MSG_HISTORY_CLEARED = "History cleared."
MSG_HISTORY_ALREADY_EMPTY = "History is already empty."
MSG_OPERATION_CANCELLED = "Operation cancelled."

# ---- BYOM (HuggingFace) instructions ----
BYOM_INSTRUCTIONS = [
    "  1. Go to   ->   https://huggingface.co/models?library=transformers",
    "  2. Search any model with TRANSFORMERS library support",
    "  3. Copy its `USER/MODEL` into your command like:\n",
]

# ---- OpenAI tokenizer descriptions ----
OPENAI_DESCRIPTIONS = {
    "cl100k_base": "used in GPT-3.5 (late), GPT-4",
    "o200k_base": "used in GPT-4o, o-family (o1, o3, o4)",
    "p50k_base": "used in GPT-3.5 (early)",
    "p50k_edit": "used in GPT-3 edit models (text-davinci, code-davinci)",
    "r50k_base": "used in GPT-3 base models (davinci, curie, babbage, ada)",
}

# ---- Example HuggingFace models for BYOM ----
BYOM_EXAMPLE_MODELS = [
    "deepseek-ai/DeepSeek-R1",
    "google-bert/bert-base-uncased",
    "google/gemma-3n-E4B-it",
    "meta-llama/Meta-Llama-3.1-405B",
    "mistralai/Devstral-Small-2507",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "openai/gpt-oss-120b"
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
