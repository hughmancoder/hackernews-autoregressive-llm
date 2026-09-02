"""
Training module.
Contains hyperparameters, logging configuration, and the core training loop.
"""
import utils
import argparse
import random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm
import structlog

from data import get_titles, get_batch, iter_full_split, train_tokenizer, BPETokenizer
from model import GPTConfig, GPT

@dataclass
class Hyperparameters:
    block_size: int = 32
    batch_size: int = 128
    vocab_size: int = 8_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0
    lr: float = 8e-4
    weight_decay: float = 0.1
    evals_per_epoch: int = 3
    
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"


def get_args() -> Hyperparameters:
    hp = Hyperparameters()
    parser = argparse.ArgumentParser(description="Train GPT on Hacker News titles")
    parser.add_argument("--lr", type=float, default=hp.lr, help=f"Learning rate (default: {hp.lr})")
    parser.add_argument("--vocab_size", type=int, default=hp.vocab_size, help=f"Vocab size (default: {hp.vocab_size})")
    parser.add_argument("--batch_size", type=int, default=hp.batch_size, help=f"Batch size (default: {hp.batch_size})")
    parser.add_argument("--block_size", type=int, default=hp.block_size, help=f"Block size (default: {hp.block_size})")
    parser.add_argument("--weight_decay", type=float, default=hp.weight_decay, help=f"Weight decay (default: {hp.weight_decay})")
    parser.add_argument("--dropout", type=float, default=hp.dropout, help=f"Dropout (default: {hp.dropout})")
    parser.add_argument("--n_layer", type=int, default=hp.n_layer, help=f"Number of layers (default: {hp.n_layer})")
    parser.add_argument("--n_head", type=int, default=hp.n_head, help=f"Number of heads (default: {hp.n_head})")
    parser.add_argument("--d_model", type=int, default=hp.d_model, help=f"Model dimension (default: {hp.d_model})")
    parser.add_argument("--evals_per_epoch", type=int, default=hp.evals_per_epoch, help=f"Evals per epoch (default: {hp.evals_per_epoch})")
    parser.add_argument("--log_file", type=str, default=hp.log_file, help=f"Log file path (default: {hp.log_file})")
    args, _ = parser.parse_known_args()
    
    for k, v in vars(args).items():
        setattr(hp, k, v)
    return hp

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    
    # NOTE: do not change
    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'ptr': ptr,
        'epoch': epoch,
    }

    Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, './checkpoints/checkpoint_best.pt')
    

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
