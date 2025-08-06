class ModelError(Exception):
    """Base error for all model/provider-related failures."""
    pass


class ModelNotFoundError(ModelError):
    """Requested model is unknown or not registered."""
    TEMPLATE = "Model '{model}' not found."

    def __init__(self, model: str):
        super().__init__(self.TEMPLATE.format(model=model))
        self.model = model


class ModelLoadError(ModelError):
    """Provider or model failed to initialize or load resources."""
    TEMPLATE = "Failed to initialize model '{model}': {reason}"

    def __init__(self, model: str, reason: str):
        super().__init__(self.TEMPLATE.format(model=model, reason=reason))
        self.model = model
        self.reason = reason


class TokenizationError(ModelError):
    """Tokenization call failed or response was invalid."""
    TEMPLATE = "Failed to tokenize with model '{model}': {reason}"

    def __init__(self, model: str, reason: str):
        super().__init__(self.TEMPLATE.format(model=model, reason=reason))
        self.model = model
        self.reason = reason


class MissingDependencyError(ModelError):
    """Optional dependency is missing or failed to import."""
    TEMPLATE = (
        "Package '{package}' is not installed. Install model provider packages:\n"
        "  pip install 'tokker[all]'                 - all at once\n"
        "  pip install 'tokker[tiktoken]'            - OpenAI\n"
        "  pip install 'tokker[google-genai]'        - Google\n"
        "  pip install 'tokker[transformers]'        - HuggingFace"
    )

    def __init__(self, package: str, install_hint: str | None = None):
        hint = install_hint or package
        super().__init__(self.TEMPLATE.format(package=package, install_hint=hint))
        self.package = package
        self.install_hint = hint
