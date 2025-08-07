#!/usr/bin/env python3


from tokker import messages
from tokker.models.registry import ModelRegistry


def run_list_models() -> None:
    """List available models, grouped by provider with curated messaging."""
    registry = ModelRegistry()

    # New template format output (always use template format)
    print(messages.SEP_MAIN)

    # OpenAI (curated descriptions)
    print(messages.HDR_OPENAI)
    for model in registry.list_models("OpenAI"):
        name = model["name"]
        description = messages.OPENAI_DESCRIPTIONS.get(name)
        if description:
            print(f"  {name}{description}")

    print(messages.SEP_SUB)
    print(messages.HDR_GOOGLE)
    for model in registry.list_models("Google"):
        print(f"  {model['name']}")
    print(messages.MSG_AUTH_REQUIRED)

    print(messages.SEP_SUB)
    print(messages.HDR_HF_BYOM)
    for line in messages.BYOM_INSTRUCTIONS:
        print(line)

    for example in messages.BYOM_EXAMPLE_MODELS:
        print(f"  {example}")
    print(messages.SEP_MAIN)
