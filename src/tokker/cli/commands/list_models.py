#!/usr/bin/env python3


from tokker.cli import strings
from tokker.models.registry import ModelRegistry


def run_list_models() -> None:
    """List available models, grouped by provider with curated messaging."""
    registry = ModelRegistry()

    # New template format output (always use template format)
    print(strings.SEP_MAIN)

    # OpenAI (curated descriptions)
    print(strings.HDR_OPENAI)
    for model in registry.list_models("OpenAI"):
        name = model["name"]
        description = strings.OPENAI_DESCRIPTIONS.get(name)
        if description:
            print(f"  {name}{description}")

    print(strings.SEP_SUB)
    print(strings.HDR_GOOGLE)
    for model in registry.list_models("Google"):
        print(f"  {model['name']}")
    print(strings.MSG_AUTH_REQUIRED)

    print(strings.SEP_SUB)
    print(strings.HDR_HF_BYOM)
    for line in strings.BYOM_INSTRUCTIONS:
        print(line)

    for example in strings.BYOM_EXAMPLE_MODELS:
        print(f"  {example}")
    print(strings.SEP_MAIN)
