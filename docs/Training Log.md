# Training Log


| Exp ID | Description | LR | Layers / Heads / Dim | Train Loss | Val Loss | 
|---|---|---|---|---|---|
| Baseline | Default config (SGD, vocab 16k, block 128) | 0.006 | 6 / 8 / 512 | 8.179 | 1.754 | 
| exp-001-hp | SGD, Lower LR (1e-3), Vocab 8k, Block 64, Batch 128, WD 0.1, Dropout 0 | 0.001 | 6 / 8 / 512 | 8.223 | 2.025 | 
| exp-002-hp | AdamW, Vocab 16k, Block 128, Batch 64, WD 0.1, Dropout 0.1 | 6e-4 | 6 / 8 / 512 | 5.802 | 1.291 | 
| exp-003-hp | AdamW, Vocab 8k, Block 32, Batch 128, WD 0.1, Dropout 0.0 | 8e-4 | 6 / 8 / 512 | *Pending* | *Pending* | 

## Experiment Logs

### Baseline

`mainrun/logs/baseline.log`

**Results**
- `Final Val Loss`: 1.754
- `Final Train Loss`: 8.179
- `Runtime`: ~39.5 min (CPU)

**Training Hyperparameters**
- `optimizer`: SGD (momentum=0.0)
- `lr`: 0.006 
- `weight_decay`: 0.0
- `dropout`: 0.1
- `batch_size`: 64
- `block_size`: 128
- `vocab_size`: 16000

**Architecture**
- `d_model`: 512 
- `n_layer`: 6 
- `n_head`: 8
- `parameters`: ~27.17M

---

### exp-001-hp

**Results**
- `Runtime`: ~13.1 min (783.4s on Apple Silicon MPS)

**Training Hyperparameters**
- `optimizer`: SGD (momentum=0.0)
- `lr`: 0.001 
- `weight_decay`: 0.1
- `dropout`: 0.0
- `batch_size`: 128
- `block_size`: 64
- `vocab_size`: 8192

**Architecture**
- `d_model`: 512 
- `n_layer`: 6 
- `n_head`: 8


**Observations**
- Under-optimisation with SGD: Lowering learning rate from `0.006` to `0.001` with vanilla SGD caused may have resulted in slower gradient descent progress, leading to underfitting.
- Vocab Shrinkage: Halved `block_size` to 64 and `vocab_size` to 8192 reduced model capacity and context window.

### exp-002-hp

**Training Hyperparameters**
- `optimizer`: AdamW
- `lr`: 0.0006 
- `weight_decay`: 0.1
- `dropout`: 0.1
- `batch_size`: 64
- `block_size`: 128
- `vocab_size`: 16_000

**Architecture**
- `d_model`: 512 
- `n_layer`: 6 
- `n_head`: 8

**Loss**
- `Train loss`: 5.80
- `Val Loss`: 1.29

**Observations**
- AdamW leads to improve gradient descent for 7 epochs compared to vanilla SGD.
- Lower learning rate prevents divergence and allows AdamW to adapt learning rate. 

### exp-003-hp

**Training Hyperparameters**
- `optimizer`: AdamW
- `lr`: 0.0008
- `weight_decay`: 0.1
- `dropout`: 0.0
- `batch_size`: 128
- `block_size`: 32
- `vocab_size`: 8000

**Architecture**
- `d_model`: 512 
- `n_layer`: 6 
- `n_head`: 8

**Loss**
- `Train loss`: 4.48
- `Val Loss`: 1.27

**Observations**
- Doubling batch size to 128 may provide smoother update steps per iteration and improve gpu throughput leading to reduced training time
- With the titles not exceeding 100 characters in length in the dataset which may correspond to 20-30 tokens, a shorter context window (block_size: 32) may confine the attention to headline context and reduce cross-sample attention noise. It also allows a larger batch size. 
- Weight decay: multiplies and shrinks a neural network's weights toward zero at every training step to prevent overfitting