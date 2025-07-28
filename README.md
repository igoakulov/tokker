# Tokker CLI

An open-source, locally-run tool for CLI (later also for Alfred / Aircast on macOS) that performs tokenization of text using OpenAI’s `tiktoken` (later also HuggingFace `transformers` library).

---

## Features

- **Token Counting**: Accurate token count using OpenAI's tiktoken library
- **Multiple Tokenizers**: Support for `cl100k_base` (GPT-4) and `o200k_base` (GPT-4o) tokenizers
- **Flexible Output**: JSON, plain text, and summary output formats
- **Configuration**: Persistent configuration for default tokenizer and delimiter settings
- **Text Analysis**: Word count, character count, and token frequency analysis
- **Cross-platform**: Works on Windows, macOS, and Linux

---

## Setup

TBD

---

## Usage

### Get full output

```bash
$ tokker --text 'hello world'
{
  'converted': 'hello⎮ world',
  'token_strings': ['hello', ' world'],
  'token_ids': [15339, 1917],
  'token_count': 2,
  'word_count': 2,
  'char_count': 11,
  'pivot': {
    'hello': 1,
    ' world': 1
  },
  'tokenizer': 'cl100k_base'
}
```

---

### Get plain (delimited) text

```bash
$ tokker --text 'hello world' --format plain
Hello⎮ world
```

---

### Get summary

```bash
$ tokker --text 'hello world' --format summary
{
  'token_count': 2,
  'word_count': 2,
  'char_count': 11,
  'tokenizer': 'o200k_base'
}
```

---

### Run specific (non-default) tokenizer

``` bash
$ tokker --text 'hello world' --tokenizer o200k_base
```

---

### Set defaul tokenizer

```bash
# Set default tokenizer
$ tokker --set-default-tokenizer o200k_base
✓ Default tokenizer set to: o200k_base
Configuration saved to: /home/user/.config/tokker/tokenizer_config.json
```

---

## Configuration

Tokker stores configuration in `~/.config/tokker/tokenizer_config.json`:

```json
{
  'default_tokenizer': 'cl100k_base',
  'delimiter': '⎮'
}
```

- `default_tokenizer`: Default tokenizer to use (`cl100k_base` or `o200k_base`)
- `delimiter`: Character used to separate tokens in plain text output

---

## Tokenizers

### cl100k_base
- **Used by**: GPT-4, GPT-3.5-turbo
- **Description**: OpenAI's standard tokenizer for GPT-4 models
- **Vocabulary size**: ~100,000 tokens

---

### o200k_base
- **Used by**: GPT-4o, GPT-4o-mini
- **Description**: Newer tokenizer with improved efficiency
- **Vocabulary size**: ~200,000 tokens

---

## Project Structure
```
tokker/
├── tokker/
│   ├── __main__.py          # Entry point for python -m tokker
│   └── cli/
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       ├── tokenize.py      # Main CLI interface
│       └── utils.py         # Core tokenization utilities
├── tests/                   # Test suite
├── README.md               # This file
├── LICENSE                 # MIT License
├── pyproject.toml          # Project configuration
└── requirements.txt        # Dependencies
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Changelog

### v0.1.0 (Initial Release)

- Basic tokenization functionality
- Support for cl100k_base and o200k_base tokenizers
- JSON, plain text, and summary output formats
- Configuration management
- Command-line interface

---

## Acknowledgments

- OpenAI for the tiktoken library
