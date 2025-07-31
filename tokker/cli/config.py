#!/usr/bin/env python3
"""
Configuration management for Tokker CLI.

Handles loading and saving model configuration from ~/.config/tokker/model_config.json
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Default configuration values
DEFAULT_CONFIG = {
    "default_model": "o200k_base",
    "delimiter": "⎮"
}

class ConfigError(Exception):
    """Raised when configuration operations fail."""
    pass

class Config:
    """Manages model configuration."""

    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path.home() / ".config" / "tokker"
        self.config_file = self.config_dir / "config.json"
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

    def get_default_model(self) -> str:
        """Get the default model from configuration."""
        config = self.load()
        return config.get("default_model", DEFAULT_CONFIG["default_model"])

    def set_default_model(self, model: str) -> None:
        """Set the default model in configuration."""
        # Validation is now handled by the CLI layer using the registry
        # This keeps the config system focused on configuration management
        config = self.load()
        config["default_model"] = model
        self.save(config)

    def get_delimiter(self) -> str:
        """Get the delimiter from configuration."""
        config = self.load()
        return config.get("delimiter", DEFAULT_CONFIG["delimiter"])



    def get_history_file(self) -> Path:
        """Get the path to the model history file."""
        return self.config_dir / "history.json"

    def load_model_history(self) -> List[Dict[str, Any]]:
        """Load model usage history from file."""
        history_file = self.get_history_file()

        if not history_file.exists():
            return []

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If history file is corrupted, start fresh
            return []

    def save_model_history(self, history: List[Dict[str, Any]]) -> None:
        """Save model usage history to file."""
        self._ensure_config_dir()
        history_file = self.get_history_file()

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise ConfigError(f"Error saving model history: {e}")

    def add_model_to_history(self, model_name: str) -> None:
        """Add a model to usage history, moving it to the top if it already exists."""
        history = self.load_model_history()

        # Remove existing entry if present
        history = [entry for entry in history if entry.get('model') != model_name]

        # Add new entry at the top
        new_entry = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'count': 1
        }

        # Update count if we're re-adding
        for entry in history:
            if entry.get('model') == model_name:
                new_entry['count'] = entry.get('count', 0) + 1
                break

        history.insert(0, new_entry)

        # Keep only the last 50 entries to prevent unbounded growth
        history = history[:50]

        self.save_model_history(history)

    def clear_model_history(self) -> None:
        """Clear all model usage history."""
        history_file = self.get_history_file()
        if history_file.exists():
            history_file.unlink()

# Global configuration instance
config = Config()
