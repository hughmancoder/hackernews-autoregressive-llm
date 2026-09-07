# Training Log

## Llama Transfomer Architecture

### Llama model without prenorm scaling (exp-004-llama)

`mainrun/logs/exp-004-llama.log`

- `Final Val Loss` : 1.234
- `Final Train Loss` : 3.988
- `Runtime` : 906s
- `Total Steps` : 2247
- `Paramaters`: 22.97M

same parameters are used as the previous experiment

### Llama model with warmup_scheduler and cosine_scheduler optimisation suggested from `LLaMA: Open and Efficient Foundation Language Models` (exp-005-llama)

`mainrun/logs/exp-005-llama.log`

- `Final Val Loss` : 1.252
- `Final Train Loss` : 3.656
- `Runtime` : 911s 
- `Total Steps` : 2247
- `Parameters`: 22.97M


**Observations**
Step : 1712	Epoch 5	Validation loss: 1.240.  Has lowest validation loss.  
As the validation loss increases, while training loss decreases this suggests mild overfitting.
A mild dropout may help prevent this

### exp-006-llama

- `Final Val Loss` : 1.22
- `Final Train Loss` : 4.035
- `Runtime` : 939s 
- `Total Steps` : 2247
- `Parameters`: 22.97M

**Observations**
Added dropout 0.05 to help mitigate slight overfitting in last training run


### exp-007-llama

Validation loss=1.187967 
**Observations**
Increased block size. block_size = 128 tokens. 

## GPT-2 Style Transfomer Architecture
### Baseline

`mainrun/logs/baseline.log`

**Results**
- `Final Val Loss`: 1.754
- `Final Train Loss`: 8.179
- `Runtime`: ~39.5 mi\n (CPU)

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

### exp-001-hp
`mainrun/logs/exp-001-hp.log`

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

`mainrun/logs/exp-002-hp.log`
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

`mainrun/logs/exp-003-hp.log`

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