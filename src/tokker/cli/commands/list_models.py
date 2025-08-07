#!/usr/bin/env python3


from tokker import messages
from tokker.models.registry import ModelRegistry
from tokker.utils import get_installed_providers, get_install_commands


def run_list_models() -> None:
    """List available models, grouped by provider with curated messaging."""
    registry = ModelRegistry()

    # New template format output (always use template format)
    print(messages.SEP_MAIN)
    installed = get_installed_providers()

    # Provider sections in canonical order with dynamic headings
    provider_order = [
        ("OpenAI", None),
        ("Google", None),
        ("HuggingFace", None),
    ]

    for idx, (provider_name, install_cmd) in enumerate(provider_order):
        if provider_name in installed:
            heading = f"{provider_name} (installed)"
        else:
            # Resolve install command via mapping helper for consistency with messages
            cmds = get_install_commands([provider_name])
            install_cmd = cmds[0] if cmds else ""
            heading = f"{provider_name} (install with: {install_cmd})"
        print(heading + "\n")

        if provider_name == "OpenAI":
            # Emit strictly in the insertion order defined by OPENAI_DESCRIPTIONS
            openai_models = registry.list_models("OpenAI")
            openai_names = {m["name"] for m in openai_models}
            for name in messages.OPENAI_DESCRIPTIONS.keys():
                if name in openai_names:
                    description = messages.OPENAI_DESCRIPTIONS.get(name, "")
                    if description:
                        print(f"  {name:<22}{description}")
                    else:
                        print(f"  {name}")

        elif provider_name == "Google":
            # List Google models in reverse alphabetical order and show auth guidance
            google_models = registry.list_models("Google")
            for model in sorted(google_models, key=lambda m: m["name"], reverse=True):
                print(f"  {model['name']}")
            print(messages.MSG_AUTH_REQUIRED)
        else:
            # HuggingFace BYOM guidance and examples (order defined in messages.BYOM_EXAMPLE_MODELS)
            for line in messages.BYOM_INSTRUCTIONS:
                print(line)
            for example in messages.BYOM_EXAMPLE_MODELS:
                print(f"  {example}")

        # Separator between sections (but not after the last)
        if idx < len(provider_order) - 1:
            print(messages.SEP_SUB)

    print(messages.SEP_MAIN)
    pass
