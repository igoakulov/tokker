# Tokker

A fast, simple CLI tool for tokenizing text using OpenAI's `tiktoken` library. Get accurate token counts for GPT models with a single command.

---

## Features

- **Simple Usage**: Just `tok 'your text'` - that's it!
- **Multiple Tokenizers**: Support for `o200k_base` (GPT-4o) and `cl100k_base` (GPT-4) tokenizers
- **Flexible Output**: JSON, plain text, and summary output formats
- **Configuration**: Persistent configuration for default tokenizer settings
- **Text Analysis**: Token count, word count, character count, and token frequency analysis
- **Cross-platform**: Works on Windows, macOS, and Linux

---

## Installation

Install from PyPI with pip:

```bash
pip install tokker
```

That's it! The `tok` command is now available in your terminal.

---

## Main commands

Quick Tips:
- Use single quotes to avoid shell interpretation: `tok 'Hello world!'`
- Pipe text from other commands: `echo "Hello world" | tok`
- Process files: `cat file.txt | tok --format summary`
- Chain with other tools: `curl -s https://example.com | tok`
- Set your preferred tokenizer once: `tok --set-default-tokenizer o200k_base`

### Full output

```bash
$ tok 'Hello world'
{
  "converted": "Hello⎮ world",
  "token_strings": ["Hello", " world"],
  "token_ids": [24912, 2375],
  "token_count": 2,
  "word_count": 2,
  "char_count": 11,
  "pivot": {
    "Hello": 1,
    " world": 1
  },
  "tokenizer": "o200k_base"
}
```

### Plain Text Output

```bash
$ tok 'Hello world' --format plain
Hello⎮ world
```

### Summary Statistics

```bash
$ tok 'Hello world' --format summary
{
  "token_count": 2,
  "word_count": 2,
  "char_count": 11,
  "tokenizer": "o200k_base"
}
```

---

## Other Commands

### Using Different Tokenizers

```bash
$ tok 'Hello world' --tokenizer cl100k_base
```

### Set Default Tokenizer:

```bash
$ tok --set-default-tokenizer o200k_base
✓ Default tokenizer set to: o200k_base
Configuration saved to: ~/.config/tokker/tokenizer_config.json
```

### Other

```
usage: tok [-h] [--tokenizer {o200k_base,cl100k_base}]
           [--format {json,plain,summary}]
           [--set-default-tokenizer {o200k_base,cl100k_base}]
           [text]

positional arguments:
  text                  Text to tokenize (or read from stdin if not provided)

options:
  --tokenizer           Tokenizer to use (o200k_base, cl100k_base)
  --format              Output format (json, plain, summary)
  --set-default-tokenizer  Set default tokenizer
  -h, --help           Show help message
```

---

## Tokenizers

- o200k_base (Default): used by GPT-4o, GPT-4o-mini; 200K vocab size
- cl100k_base: used by GPT-4, GPT-3.5; 100K vocab size

---

## Configuration

Tokker stores your preferences in `~/.config/tokker/tokenizer_config.json`:

```json
{
  "default_tokenizer": "o200k_base",
  "delimiter": "⎮"
}
```

---

## Programmatic Usage

You can also use tokker in your Python code:

```python
import tokker

# Count tokens
count = tokker.count_tokens("Hello world")
print(f"Token count: {count}")

# Full tokenization
result = tokker.tokenize_text("Hello world", "o200k_base")
print(result["token_count"])
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Issues and pull requests are welcome! Visit the [GitHub repository](https://github.com/igoakulov/tokker).

---

## Acknowledgments

- OpenAI for the tiktoken library
