# Training Log

Here is the corrected table with the validation loss updated to match your notes.

| Exp ID | Description | LR | Layers / Heads / Dim | Train Loss | Val Loss |
|---|---|---|---|---|---|
| Baseline | Default config | 0.006 | 6 / 8 / 512 | 8.179 | 1.754 |

## Experiment Logs

### Baseline

`mainrun/logs/baseline.log`

**Results**
- `Final Val Loss`: 1.753
- `Final Train Loss`: 8.179
- `Runtime`: ~39.5 min 

**Trainign Hyper Parameters**
- `lr`: 0.006 
- `weight_decay`: 0.0
- `dropout`: 0.1
- `batch_size`: 64

**Architecture**
- `d_model`: 512 
- `n_layer`: 6 
- `n_head`: 8
- `block_size`: 128 
- `vocab_size`: 16000 
- `parameters`: ~27.17M

### Experiment 001 