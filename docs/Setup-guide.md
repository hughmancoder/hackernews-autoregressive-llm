# Setup Guide

## Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Optional)* Task runner and zx for the task workflows:
```bash
brew install go-task
npm install -g zx
```

## Download Dataset
```bash
cd src && python3 download_dataset.py && cd ..
```

## Dev Container (optional)
1. Start Docker: `open -a Docker`
2. Open the project in VS Code
3. `Cmd + Shift + P` → **Dev Containers: Reopen in Container**

## Commands

Run `make` targets from the repo root (see [makefile](../makefile)):

| Command | Description |
|---------|-------------|
| `make train_local` | Train with default settings |
| `make train_custom` | Train with custom hyperparameters |
| `make analyse_train_run` | Print the baseline log |
| `make test` | Run all tests |
| `make generate_report` | Build the PDF report |

## Files

- `src/data.py`: dataset loading, BPE tokenizer training, batching.
- `src/train.py`: training loop and hyperparameters.
- `src/llama_model.py`: Llama-style transformer.
- `src/gpt_model.py`: GPT-2 style baseline model.
