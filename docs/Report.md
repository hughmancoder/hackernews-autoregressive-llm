# Report

A short report.pdf that walks through the changes you made, your reasoning, and their effect on the training curves. Place the report in the mainrun folder before submitting.


## Changes made and effects on training

Switched from vanilla Stochastic Gradient Descent (SGD) to [AdamW](https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html), the industry-standard optimiser for training Transformers to predict token in a sequence

```python
opt = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=args.weight_decay
)
```

- Modified train.py to use Apple Silicon MPS support to train on Macbook m4 gpu instead of cpu. Led to ~5x faster training times compared to docker container running on the cpu. This allows me to run more experiments in a shorter time frame.

**Hyperparameter Selection**

- `lr: 0.0006`: a suggested standard learning rate using AdamW. Higher learning rates can lead to instability.
- `batch_size: 64`: Better for epoch restriction (7). A smaller batch size means your model takes twice as many gradient steps per epoch which may lead to improved learning
- `Block size: 128`: appears to be a long enough context window for longer titles in the dataset
- `dropout 0.1`: The system deactivates a random 10% subset of nodes in a neural network layer and a new random 10% is chosen for every training batch. As the model appears to learn much faster with Adam loss, and may be more prone to overfitting the relatively small 100k training set. Dropout forces the network to learn generalized patterns rather than memorising exact training phrases which may reduce overfitting. 


Refer to `docs/Training Log.md` for a more detailed list of training runs

## Resources

- [A Survey on Efficient Training of Transformers](https://arxiv.org/pdf/2302.01107)