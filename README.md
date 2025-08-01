# Tokker

Tokker is a fast local-first CLI tool for tokenizing text with all the best models in one place.

---

## Features

- **Simple Usage**: Just `tok 'your text'` - that's it!
- **Models**:
  - OpenAI: GPT-3, GPT-3.5, GPT-4, GPT-4o, o-family (o1, o3, o4)
  - Google: the entire Gemini family
  - HuggingFace: select literally [any model](https://huggingface.co/models?library=transformers) that supports `transformers` library
- **Flexible Output**: JSON, plain text, count, and pivot output formats
- **Model History**: Track your usage with `--history` and `--history-clear`
- **Configuration**: Persistent configuration for default model and settings
- **Text Analysis**: Token count, word count, character count, and token frequency
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Local-first**: Works locally on device (except Google)

---

## Installation

```bash
pip install tokker
```

That's it! The `tok` command is now available in your terminal.

---

## Command Reference

```
usage: tok [-h] [-m MODEL] [-o {json,plain,count,pivot}]
           [-D MODEL_DEFAULT] [-M]
           [-H] [-X]
           [text]

positional arguments:
  text                                    text to tokenize (or read from stdin)

options:
  -h, --help                              show this help message and exit
  -m, --model MODEL                       model to use (overrides default)
  -o, --output {json,plain,count,pivot}   output format (default: json)
  -D, --model-default MODEL_DEFAULT       set default model
  -M, --models                            list all available models
  -H, --history                           show history of used models
  -X, --history-clear                     clear history
```

## Usage

### Tokenize Text

Tip: When using `bash` or `zsh`, wrap input text in single quotes ('like this'). Double quotes cause issues with special characters such as `!` (used for history expansion).

```bash
# Tokenize with default model
tok 'Hello world'

# Get a specific output format
tok 'Hello world' -o plain

# Use a specific model
tok 'Hello world' -m deepseek-ai/DeepSeek-R1

# Get just the counts
tok 'Hello world' -m gemini-2.5-pro -o count
```

### Pipeline Usage

```bash
# Process files
cat document.txt | tok -m gpt2 -o count

# Chain with other tools
curl -s https://example.com | tok -m bert-base-uncased

# Compare models
echo "Machine learning is awesome" | tok -m gpt2
echo "Machine learning is awesome" | tok -m bert-base-uncased
```

### List Available Models

```bash
# See all available models
tok -M
```

Output:
```
============
OpenAI:

  cl100k_base           used in GPT-3.5 (late), GPT-4
  o200k_base            used in GPT-4o, o-family (o1, o3, o4)
  p50k_base             used in GPT-3.5 (early)
  p50k_edit             used in GPT-3 edit models (text-davinci, code-davinci)
  r50k_base             used in GPT-3 base models (davinci, curie, babbage, ada)
------------
Google:

  gemini-2.5-pro
  gemini-2.5-flash
  gemini-2.5-flash-lite
  gemini-2.0-flash
  gemini-2.0-flash-lite

Auth setup required   ->   https://github.com/igoakulov/tokker/blob/main/tokker/google-auth-guide.md
------------
HuggingFace (BYOM - Bring You Own Model):

  1. Go to   ->   https://huggingface.co/models?library=transformers
  2. Search any model with TRANSFORMERS library support
  3. Copy its `USER/MODEL` into your command like:

  deepseek-ai/DeepSeek-R1
  google-bert/bert-base-uncased
  google/gemma-3n-E4B-it
  meta-llama/Meta-Llama-3.1-405B
  mistralai/Devstral-Small-2507
  moonshotai/Kimi-K2-Instruct
  Qwen/Qwen3-Coder-480B-A35B-Instruct
============
```

### Set Default Model

```bash
# Set your preferred model
tok -D o200k_base
```

### History

```bash
# View your model usage history with date/time
tok -H

# Clear your history (will prompt for confirmation)
tok -X
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
  "char_count": 11
}
```

### Plain Text Output

```bash
$ tok 'Hello world' -o plain
Hello⎮ world
```

### Count Output

```bash
$ tok 'Hello world' -o count
{
  "token_count": 2,
  "word_count": 2,
  "char_count": 11
}
```

### Pivot Output

The pivot output prints a JSON object with token frequencies, sorted by highest count first, then by token (A–Z).

Example:
```bash
$ tok 'never gonna give you up neve gonna let you down' -m cl100k_base -o pivot
{
  " gonna": 2,
  " you": 2,
  " down": 1,
  " give": 1,
  " let": 1,
  " ne": 1,
  " up": 1,
  "never": 1,
  "ve": 1
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Issues and pull requests are welcome! Visit the [GitHub repository](https://github.com/igoakulov/tokker).

---

## Acknowledgments

- OpenAI for the tiktoken library
- HuggingFace for the transformers library
- Google for the Gemini models and APIs
