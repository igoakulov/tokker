#!/usr/bin/env python3
"""
Configuration management for Tokker CLI.

Handles loading and saving tokenizer configuration from ~/.config/tokker/tokenizer_config.json
"""

import json
from pathlib import Path
from typing import Dict, Any

# Default configuration values
DEFAULT_CONFIG = {
    "default_tokenizer": "o200k_base",
    "delimiter": "⎮"
}

# Valid tokenizer names
VALID_TOKENIZERS = {"o200k_base", "cl100k_base"}

class ConfigError(Exception):
    """Raised when configuration operations fail."""
    pass

class Config:
    """Manages tokenizer configuration."""

    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path.home() / ".config" / "tokker"
        self.config_file = self.config_dir / "tokenizer_config.json"
        self._config = None

    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise ConfigError(
                f"Cannot create configuration directory {self.config_dir}. "
                f"Please check permissions or create the directory manually."
            ) from e

    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self._config is not None:
            return self._config

        self._ensure_config_dir()

        if not self.config_file.exists():
            # Create default config file
            self.save(DEFAULT_CONFIG)
            self._config = DEFAULT_CONFIG.copy()
        else:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                raise ConfigError(f"Error loading configuration: {e}")

        # Ensure required keys exist
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in self._config:
                self._config[key] = default_value

        return self._config

    def save(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        self._ensure_config_dir()

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self._config = config
        except IOError as e:
            raise ConfigError(f"Error saving configuration: {e}")

    def get_default_tokenizer(self) -> str:
        """Get the default tokenizer from configuration."""
        config = self.load()
        return config.get("default_tokenizer", DEFAULT_CONFIG["default_tokenizer"])

    def set_default_tokenizer(self, tokenizer: str) -> None:
        """Set the default tokenizer in configuration."""
        if tokenizer not in VALID_TOKENIZERS:
            raise ConfigError(
                f"Invalid tokenizer: {tokenizer}. "
                f"Valid options: {', '.join(sorted(VALID_TOKENIZERS))}"
            )

        config = self.load()
        config["default_tokenizer"] = tokenizer
        self.save(config)

    def get_delimiter(self) -> str:
        """Get the delimiter from configuration."""
        config = self.load()
        return config.get("delimiter", DEFAULT_CONFIG["delimiter"])

    def get_valid_tokenizers(self) -> set:
        """Get set of valid tokenizer names."""
        return VALID_TOKENIZERS.copy()

# Global configuration instance
config = Config()
