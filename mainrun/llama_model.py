"""
Defines the Llama style transformer model and  configuration
Based on paper: LLaMA: Open and Efficient Foundation Language Models by Meta AI
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class LlamaConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        # ensures the full embedding dimension can be evenly divided across all attention head
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head

        #  single linear layer computes the Queries, Keys, and Values simultaneously by projecting the input to 3 times the d_model siz
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Initialises dropout layers to randomly zero out elements during training to mitigate overfitting
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)

        # Rotary embeddings
        self.rotary_emb = RotaryPositionalEmbeddingFrequencies(self.head_dim, max_seq_len=cfg.block_size)
        # Creates a lower triangular matrix of ones and saves it as a PyTorch buffer. During the forward pass, this mask is used to force all future token relationships to negative infinity so the model relies on past and present tokens.
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        # attention mechanism
        B, T, C = x.size()
        # permute to shape (3, B, n_head, T, head_dim)
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE embeddings
        cos, sin = self.rotary_emb(T)
        cos, sin = cos.to(q.device), sin.to(q.device)
        q, k = apply_rotary_positional_embeddings(q, k, cos, sin)

        # scaled dot product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # upper triangular matrix mask to prevent attending to future tokens
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

def apply_rotary_positional_embeddings(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Rotates query and key tensors
    Expects q and k to be shaped (B, n_head, T, head_dim)
    Expects cos and sin to be shaped (1, 1, T, head_dim)
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        dim_half = x.shape[-1] // 2
        # split features in half
        x1, x2 = x[..., :dim_half], x[..., dim_half:]
        # corresponds to 90 degree rotation
        return torch.cat((-x2, x1), dim=-1)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated

class RotaryPositionalEmbeddingFrequencies(nn.Module):
    """
    Precomputes and caches sinusoidal rotary frequency bands (cos and sin)
    up to max_seq_len for Rotary Position Embeddings.
    Cached as buffers of shape (1, 1, max_seq_len, head_dim) 
    """
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        dim_half = head_dim // 2
        
        # -math.log(10000.0) = -ln(10000)
        # torch.arange(0, dim_half) / dim_half = i / (d/2). Equivalent to 10000^(-2i/d)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, dim_half).float() / dim_half) 
        t = torch.arange(max_seq_len).float() 
        
        # Position angle matrix m*Theta. Outer product to construct the angle grid of shape (max_seq_len, dim_half)
        freqs = torch.outer(t, freqs) 
        
        # Duplicate frequencies for both halves of the dimension to give the full shape (max_seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1) 
        
        # The tensors cos_cached and sin_cached store the precomputed numerical values of cos and sin for every timestep t and frequency channel i.
        #  (1, 1, max_seq_len, head_dim) to broadcasts to (B, n_head, T, head_dim)
        self.register_buffer("cos_cached", emb.cos().view(1, 1, max_seq_len, head_dim), persistent=False)
        self.register_buffer("sin_cached", emb.sin().view(1, 1, max_seq_len, head_dim), persistent=False)

    def forward(self, seq_len: int):
        # Slice the precomputed cache up to the current temporal sequence length T
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

class LlamaTransfomerDecoderBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        # define two separate instances  so each layer can learn independent scaling weights
        self.rms_1 = nn.RMSNorm(cfg.d_model)
        self.rms_2 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = LlamaMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x

class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([LlamaTransfomerDecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.RMSNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # B (Batch Size): The number of independent sequences processed at the same time. T (Sequence Length): The number of tokens in each sequence
        B, T = idx.size()
        tok = self.token_emb(idx)
        x = self.drop(tok)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        hidden_dim = int(getattr(cfg, "hidden_dim", (8 / 3) * cfg.d_model))
        self.swiglu = SwiGLU(cfg.d_model, hidden_dim)

    def forward(self, x): 
        return self.swiglu(x)

class SwiGLU(nn.Module):
    """
    Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x)) 
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


    
 

        

