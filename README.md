# Tokker

Tokker is a fast local-first CLI tool for tokenizing text with all the best models in one place.

---

## Features

- **Simple Usage**: Just `tok 'your text'` - that's it!
- **Models**:
  - OpenAI: GPT-3, GPT-3.5, GPT-4, GPT-4o, o-family (o1, o3, o4)
  - HuggingFace: select literally [any model](https://huggingface.co/models?library=transformers) that supports `transformers` library
- **Flexible Output**: JSON, plain text, and count output formats
- **Model History**: Track your usage with `--history` and `--history-clear`
- **Configuration**: Persistent configuration for default model and settings
- **Text Analysis**: Token count, word count, character count, and token frequency
- **Cross-platform**: Works on Windows, macOS, and Linux
- **99% local**: Works fully locally on device (besides initial model load)

---

## Installation

```bash
pip install tokker
```

That's it! The `tok` command is now available in your terminal.

---

## Command Reference

```
usage: tok [-h] [--model MODEL] [--output {json,plain,count,table}]
           [--model-default MODEL_DEFAULT] [--models]
           [--history] [--history-clear]
           [text]

positional arguments:
  text                  Text to tokenize (or read from stdin if not provided)

options:
  -h, --help           Show this help message and exit
  --model MODEL        Model to use (overrides default). Use --models to see available options
  --output {json,plain,count,table}
                       Output format (default: json)
  --model-default MODEL_DEFAULT
                       Set the default model in configuration. Use --models to see available options
  --models             List all available models with descriptions
  --history            Show history of used models, with most recent on top
  --history-clear      Clear model usage history
```

## Usage

Tip: When using `bash` or `zsh`, wrap input text in single quotes ('like this'). Double quotes cause issues with special characters such as `!` (used for history expansion).

### Tokenize Text

```bash
# Tokenize with default model
tok 'Hello world'

# Get a specific output format
tok 'Hello world' --output plain

# Use a specific model
tok 'Hello world' --model gpt2

# Get just the counts
tok 'Hello world' --output count
```

### Pipeline Usage

```bash
# Process files
cat document.txt | tok --model gpt2 --output count

# Chain with other tools
curl -s https://example.com | tok --model bert-base-uncased

# Compare models
echo "Machine learning is awesome" | tok --model gpt2
echo "Machine learning is awesome" | tok --model bert-base-uncased
```

### List Available Models

```bash
# See all available models
tok --models
```

Output:
```
--- OpenAI ---
cl100k_base     ->   used in GPT-3.5 (late), GPT-4
o200k_base      ->   used in GPT-4o, o-family (o1, o3, o4)
p50k_base       ->   used in GPT-3.5 (early)
p50k_edit       ->   used in GPT-3 edit models for text and code (text-davinci, code-davinci)
r50k_base       ->   used in GPT-3 base models (davinci, curie, babbage, ada)

--- HuggingFace ---
BYOM - Bring You Own Model:
1. Go to   ->   https://huggingface.co/models?library=transformers
2. Search any model with TRANSFORMERS library support
3. Copy its `USER/MODEL-NAME` into your command:

deepseek-ai/DeepSeek-R1
google-bert/bert-base-uncased
google/gemma-3n-E4B-it
meta-llama/Meta-Llama-3.1-405B
mistralai/Devstral-Small-2507
moonshotai/Kimi-K2-Instruct
Qwen/Qwen3-Coder-480B-A35B-Instruct
Etc.

--- Related Commands ---
`--model-default 'model-name'` to set default model
`--history` to view all models you have used
```

### Set Default Model

```bash
# Set your preferred model
tok --model-default o200k_base
```

### History

```bash
# View your model usage history with date/time
tok --history

# Clear your history
tok --history-clear
```

History is stored locally in `~/.config/tokker/history.json`.


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
  "model": "o200k_base",
  "provider": "OpenAI"
}
```

### Plain Text Output

```bash
$ tok 'Hello world' --output plain
Hello⎮ world
```

### Count Output

```bash
$ tok 'Hello world' --output count
{
  "token_count": 2,
  "word_count": 2,
  "char_count": 11,
  "model": "o200k_base"
}
```

---

## Configuration

Your configuration is stored locally in `~/.config/tokker/config.json`:

```json
{
  "default_model": "o200k_base",
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
result = tokker.tokenize_text("Hello world", "gpt2")
print(result["token_count"])

# Word and character counts
words = tokker.count_words("Hello world")
chars = tokker.count_characters("Hello world")
print(f"Words: {words}, Characters: {chars}")
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
