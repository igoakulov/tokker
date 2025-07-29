# Tokker

A fast, simple CLI tool for tokenizing text using OpenAI's `tiktoken` library and HuggingFace transformers. Get accurate token counts for GPT models, LLaMA, BERT and more with a single command.

---

## Features

- **Simple Usage**: Just `tok 'your text'` - that's it!
- **26 Tokenizers**: Best from OpenAI's tiktoken (tt) and HuggingFace transformers (hf) libraries - GPT, Deepseek, Llama, Qwen, Bert and other tokenizers - all in one place.
- **Flexible Output**: JSON, plain text, and summary output formats
- **Configuration**: Persistent configuration for default tokenizer and delimiter
- **Text Analysis**: Token count, word count, character count, and token frequency
- **Cross-platform**: Works on Windows, macOS, and Linux
- **100% local**: Works fully locally on device after installation

---

## Installation

Install from PyPI with pip:

```bash
pip install tokker
```

That's it! The `tok` command is now available in your terminal.

---

## Command Reference

```
usage: tok [-h] [--tokenizer TOKENIZER] [--format {json,plain,summary}]
           [--tokenizer-default TOKENIZER] [--tokenizer-list]
           [text]

positional arguments:
  text                  Text to tokenize (or read from stdin if not provided)

options:
  -h, --help           Show this help message and exit
  --tokenizer TOKENIZER
                       Tokenizer to use (overrides default). Use --tokenizer-list to see available options
  --format {json,plain,summary}
                       Output format (default: json)
  --tokenizer-default TOKENIZER
                       Set the default tokenizer in configuration. Use --tokenizer-list to see available options
  --tokenizer-list     List all available tokenizers with descriptions
```

## Usage

Tip: When using `bash` or `zsh`, wrap input text in single quotes ('like this'). Double quotes cause issues with special characters such as `!` (used for history expansion).

### Tokenize

```bash
# Tokenize with default tokenizer
tok 'Hello world'

# Get a specific output format
tok 'Hello world' --format plain

# Use a specific tokenizer
tok 'Hello world' --tokenizer gpt2

# Pipe text from other commands
echo "Hello world" | tok
cat file.txt | tok --format summary
```

### Tokenize (Pipeline)

```bash
# Process files
cat document.txt | tok --tokenizer gpt2 --format summary

# Chain with other tools
curl -s https://example.com | tok --tokenizer bert-base-uncased

# Compare tokenizers
echo "Machine learning is awesome" | tok --tokenizer gpt2
echo "Machine learning is awesome" | tok --tokenizer bert-base-uncased
```

### List Available Tokenizers

```bash
# See all available tokenizers
tok --tokenizer-list
```

Output:
```
DeepSeek Family:
================
  deepseek-ai/DeepSeek-Coder-V2-Base    (hf) — BPE, used by DeepSeek-Coder-V2
  deepseek-ai/DeepSeek-V2               (hf) — BPE, used by DeepSeek-V2

GPT Family:
===========
  cl100k_base                           (tt) — BPE, used by GPT-3.5, GPT-4
  gpt2                                  (hf) — BPE, used by GPT-2
  o200k_base                            (tt) — BPE, used by GPT-4o, o-family (o1, o3, o4)
  p50k_base                             (tt) — BPE, used by GPT-3.5
  p50k_edit                             (tt) — BPE, used by GPT-3 edit models for text and code
  r50k_base                             (tt) — BPE, used by GPT-3 base models

LLaMA Family:
=============
  meta-llama/Llama-2-70b-hf             (hf) — BPE, used by LLaMA-2
  meta-llama/Meta-Llama-3-70B           (hf) — BPE, used by LLaMA-3
  meta-llama/Meta-Llama-3.1-405B        (hf) — BPE, used by LLaMA-3.1

Qwen Family:
============
  Qwen/Qwen-72B                         (hf) — BPE, used by Qwen
  Qwen/Qwen1.5-110B                     (hf) — BPE, used by Qwen1.5
  Qwen/Qwen2-72B                        (hf) — BPE, used by Qwen2
  Qwen/Qwen2.5-72B                      (hf) — BPE, used by Qwen2.5

Other:
======
  allenai/longformer-base-4096          (hf) — BPE, used by Longformer
  bert-base-cased                       (hf) — WordPiece, used by BERT
  bert-base-uncased                     (hf) — WordPiece, used by BERT
  distilbert-base-cased                 (hf) — WordPiece, used by DistilBERT
  distilbert-base-uncased               (hf) — WordPiece, used by DistilBERT
  facebook/bart-base                    (hf) — BPE, used by BART
  google/electra-base-discriminator     (hf) — WordPiece, used by ELECTRA
  microsoft/deberta-base                (hf) — SentencePiece, used by DeBERTa
  roberta-base                          (hf) — BPE, used by RoBERTa
  t5-base                               (hf) — SentencePiece, used by T5
  xlnet-base-cased                      (hf) — SentencePiece, used by XLNet
```

### Set Default Tokenizer

```bash
# Set your preferred tokenizer
tok --tokenizer-default o200k_base
```

Output:
```
✓ Default tokenizer set to: o200k_base (tt) — BPE, used by GPT-4o, o-family (o1, o3, o4)
Configuration saved to: ~/.config/tokker/tokenizer_config.json
```

---

## Output Formats

### Full JSON Output (Default)

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
  "tokenizer": "o200k_base",
  "library": "tt"
}
```

### Plain Text Output

```bash
$ tok 'Hello world' --format plain
Hello⎮ world
```

### Summary Output

```bash
$ tok 'Hello world' --format summary
{
  "token_count": 2,
  "word_count": 2,
  "char_count": 11,
  "tokenizer": "o200k_base",
  "library": "tt"
}
```

### Tokenizer List JSON

```bash
# Get tokenizer list as JSON
tok --tokenizer-list --format json

# Process and extract token count
tok 'Hello world' --format summary | jq '.token_count'
```

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
count = tokker.count_tokens("Hello world", "o200k_base")
print(f"Token count: {count}")

# Full tokenization
result = tokker.tokenize("Hello world", "gpt2")
print(result["token_count"])

# List available tokenizers
tokenizers = tokker.list_tokenizers()
for tokenizer in tokenizers:
    print(f"{tokenizer['name']} ({tokenizer['library']}) — {tokenizer['description']}")
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
- HuggingFace for the transformers library
