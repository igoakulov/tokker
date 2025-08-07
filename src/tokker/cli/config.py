#!/usr/bin/env python3
from datetime import datetime
import json
from pathlib import Path

# Default configuration values
DEFAULT_CONFIG: dict[str, str] = {"default_model": "o200k_base", "delimiter": "⎮"}


class Config:
    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path.home() / ".config" / "tokker"
        self.config_file = self.config_dir / "config.json"
        self._config: dict[str, str] | None = None

    def _ensure_config_dir(self) -> None:
        # Let PermissionError bubble if cannot create
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, str]:
        if self._config is not None:
            return self._config

        self._ensure_config_dir()

        if not self.config_file.exists():
            self.save(DEFAULT_CONFIG)
            self._config = DEFAULT_CONFIG.copy()
        else:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("Invalid configuration format")
                self._config = {str(k): str(v) for k, v in loaded.items()}

        for k, v in DEFAULT_CONFIG.items():
            self._config.setdefault(k, v)

        return self._config

    def save(self, config: dict[str, str]) -> None:
        self._ensure_config_dir()
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self._config = config

    def get_default_model(self) -> str:
        config = self.load()
        return config.get("default_model", DEFAULT_CONFIG["default_model"])

    def set_default_model(self, model: str) -> None:
        config = self.load()
        config["default_model"] = model
        self.save(config)

    def get_delimiter(self) -> str:
        config = self.load()
        return config.get("delimiter", DEFAULT_CONFIG["delimiter"])

    def get_history_file(self) -> Path:
        return self.config_dir / "history.json"

    def load_history(self) -> list[dict[str, str | int]]:
        history_file = self.get_history_file()
        if not history_file.exists():
            return []

        with open(history_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if not isinstance(loaded, list):
                return []
            # Ensure expected shapes (no timestamp validation; trust write path)
            coerced: list[dict[str, str | int]] = []
            for item in loaded:
                if not isinstance(item, dict):
                    continue
                if "model" not in item or "timestamp" not in item:
                    continue

                model = str(item.get("model"))
                timestamp = str(item.get("timestamp"))
                count_val = item.get("count", 1)

                is_count_numeric = isinstance(count_val, (int, float, str))
                count = (
                    int(count_val)
                    if is_count_numeric and str(count_val).isdigit()
                    else 1
                )

                coerced.append({"model": model, "timestamp": timestamp, "count": count})
            return coerced

    def save_history(self, history: list[dict[str, str | int]]) -> None:
        self._ensure_config_dir()
        history_file = self.get_history_file()
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def add_model_to_history(self, model_name: str) -> None:
        history = self.load_history()
        history = [entry for entry in history if entry.get("model") != model_name]

        new_entry: dict[str, str | int] = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "count": 1,
        }

        for entry in history:
            if entry.get("model") == model_name:
                prev = entry.get("count", 0)
                new_entry["count"] = int(prev) + 1 if isinstance(prev, int) else 1
                break

        history.insert(0, new_entry)
        history = history[:50]
        self.save_history(history)

    def clear_history(self) -> None:
        history_file = self.get_history_file()
        if history_file.exists():
            history_file.unlink()


# Global configuration instance
config = Config()
