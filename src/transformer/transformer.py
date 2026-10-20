
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional

from transformer.modules import Embedding, RMSNorm, MultiHeadSelfAttentionRoPE, SiGLU, Linear, softmax

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int, max_seq_len: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.norm1 = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttentionRoPE(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ff = SiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x
        batch_size, seq_len, d_model = x.shape
        attn = x + self.attention(self.norm1(x), torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1))
        ff = self.ff(self.norm2(attn))
        return ff + attn
        
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta
        self.device = device
        self.dtype = dtype
        
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        out = self.lm_head(out)
        return out
