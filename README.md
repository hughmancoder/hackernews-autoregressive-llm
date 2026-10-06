# HackerNews Autoregressive LLM

A small autoregressive transformer trained on Hacker News headlines. Both GPT-style and LLama-style architectures were experimented with to minimise validation loss. 

## LLM Architecture performance

| Model | Val loss |
|-------|----------|
| GPT-2 style baseline (SGD) | 1.754 |
| GPT-2 + AdamW (exp-002) | 1.29 |
| Llama-style (exp-004) | 1.234 |
| Llama + dropout 0.05 (exp-006) | 1.22 |
| **Llama + block_size 128 (exp-007)** | **1.188** |

The best run uses a Llama-style architecture (~23M parameters) with a 16k BPE vocabulary, `block_size=128`, AdamW, warmup + cosine schedule and light dropout. See the [Training Log](docs/Training-log.md) for all experiments and the [Report](Report.pdf) for analysis.

## Constraints

- Fixed number of epochs, random seed, dataset and validation fraction.
- No pre-trained weights or data augmentation.
- `evaluate()` is unchanged.
- Everything else (architecture, tokenizer, optimizer, scheduler, hyperparameters) is open.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make train_local        # train
make test               # run tests
```

See the [Setup Guide](docs/Setup-guide.md) for details.

## Layout

```
src/        model, tokenizer, training loop, logs
tests/      unit tests
docs/       training log, setup guide, report source
scripts/    checkpoint, report generation
```