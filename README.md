# Small Language Model

A lightweight, character-level language model built from scratch using PyTorch. This implements a GPT-style, decoder-only Transformer architecture using a single transformer block to autoregressively generate text resembling the works of William Shakespeare in gibberish.

This repository serves as a foundational implementation of modern language modeling mechanics, optimized for local CPU training and experimentation.
Read the full article on [Medium](https://medium.com/@anant07singh2004/building-an-slm-00e691b8496b?sharedUserId=anant07singh2004).


## Architecture

The model is built on a custom Transformer architecture featuring:
* **Character-Level Tokenization:** A custom dictionary-based tokenizer mapping unique characters to integer IDs.
* **Embeddings:** Combined learnable token embeddings and positional encodings to provide the model with spatial awareness.
* **Causal Self-Attention:** Utilizes PyTorch's `nn.TransformerEncoderLayer` modified with a strict causal mask (`generate_square_subsequent_mask`) to block future-token visibility, converting it into a purely autoregressive decoder block.
* **Language Modeling Head:** A linear projection layer mapping hidden states back to vocabulary logits for next-character prediction.

## Project Structure

The codebase is modularized for clean execution and separation of concerns:

```text
├── data/
│   └── input.txt                 # The Tiny Shakespeare training corpus
├── checkpoints/                  # Directory for saved model weights (.pt files)
├── src/
│   ├── tokenizer.py              # Character-level tokenizer class (encode/decode)
│   ├── model.py                  # PyTorch nn.Module defining the SLM architecture
│   ├── train.py                  # Data loading, AdamW optimization, and training loop
│   └── generate.py               # Model inference and autoregressive text generation 
```

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the Tiny Shakespeare training text at:

```text
data/input.txt
```

The tokenizer is generated automatically from the text in this file.

## Training

From the src dir , run:

```bash
python train.py
```

Model weights/parameters are saved in the `checkpoints/` directory.

## Generation

From the src dir, run:

```bash
python generate.py
```

Ensure `src/generate.py` is configured to load a checkpoint saved by the training script.
