"""
Defines the Llama style transformer model and  configuration
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)"""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # intialise weight as vector  of empty ones
        self.weight = nn.Paramater(torch.ones(dimension));

    def forward(x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square along the feature dimension (dim=-1)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        
        x_norm =  x * torch.rsqrt(variance + self.eps)
        # normalise the weights
        return x_norm * self.weight
        


        
class SwiGLU(nn.Module):
    """
    Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x)) with biases 
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # Linear projection into gating branch
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # Linear projection into data branch
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # Projection back down to residual stream dimension
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        # Project back down to (batch, seq_len, d_model)
        return self.down_proj(gated)

