# Small Language Model (SLM) v2

A lightweight GPT-style, character-agnostic language model built with PyTorch. Version 2 replaces the original character-level tokenizer with GPT-2 BPE tokenization and uses a configurable four-block decoder-only Transformer.

This model is trained on the Tiny Shakespeare corpus for testing and showing and generates Shakespeare-like text autoregressively. It is designed for CPU-friendly local experimentation. You can replace the dataset with your own text to train a custom model.

Read the original project article for v1 on [Medium](https://medium.com/@anant07singh2004/building-an-slm-00e691b8496b?sharedUserId=anant07singh2004).

## What's improved in v2

- **BPE tokenization:** Uses the pretrained GPT-2 tokenizer from `tiktoken` instead of a character dictionary.
- **Deeper Transformer:** Supports a configurable stack of Transformer decoder blocks.
- **Configuration file:** Model, training, and dataset settings live in `configs/default_config.json`.
- **Validation metrics:** Training reports cross-entropy loss and perplexity for training and validation splits.
- **Sampling controls:** Generation uses temperature scaling and top-k sampling.
- **Saved v2 checkpoint:** The repository includes a ready-to-use `checkpoints/v2_model.pt` model.

## Architecture

- GPT-2-style BPE tokenization with a 50,257-token vocabulary
- Token embeddings and learned positional embeddings
- Multi-head causal self-attention
- Feed-forward projection and residual Transformer blocks
- Final layer normalization and a language-modeling output head
- Autoregressive next-token prediction

The default configuration is:

| Setting | Value |
|---|---:|
| Embedding size | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Context length | 64 tokens |
| Dropout | 0.1 |
| Batch size | 16 |
| Learning rate | 0.0003 |
| Training steps | 1,000 |

## Project Structure

```text
slm/
├── configs/
│   └── default_config.json       # Model and training configuration
├── data/
│   └── input.txt                 # Tiny Shakespeare training corpus
├── checkpoints/
│   ├── tiny_shakespeare_model.pt # v1 checkpoint
│   └── v2_model.pt               # v2 checkpoint
├── notebooks/
│   └── slm.ipynb                 # Interactive notebook version
└── src/
    ├── tokenizer.py              # GPT-2 BPE tokenizer wrapper
    ├── model.py                  # Transformer language model
    ├── evaluate.py               # Loss and perplexity evaluation
    ├── train.py                  # Training loop and checkpoint saving
    └── generate.py               # Interactive text generation
```

## Setup

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core runtime dependencies are PyTorch and `tiktoken`.

### 3. Prepare the dataset

Place the dataset text at `data\` and rename file as `input.txt`. On Windows PowerShell:
Here is a command to download the Tiny Shakespeare dataset directly:
```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" `
  -OutFile "data\input.txt"
```

On Linux/macOS:

```bash
curl -L "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -o data/input.txt
```

## Training v2

Run the training script from the repository root so that relative paths in the config resolve correctly:

```bash
python src/train.py
```

The script:

1. Loads GPT-2 BPE tokens from `data/input.txt`.
2. Splits the data into 90% training and 10% validation.
3. Evaluates loss and perplexity at configured intervals.
4. Trains with AdamW.
5. Saves the final weights to `checkpoints/v2_model.pt`.

You can change the architecture, training length, batch size, learning rate, or dataset path in `configs/default_config.json`.

## Text Generation

The v2 generator loads `checkpoints/v2_model.pt` by default and prompts for a starting string:

```bash
python src/generate.py
```

Example:

```text
Enter starting prompt: ROMEO:
Generating...
```

Generation uses a 64-token context window, temperature `0.8`, and top-k `40`. These defaults are set in `src/generate.py`; you can use the exported `generate_text()` function to request different values.
