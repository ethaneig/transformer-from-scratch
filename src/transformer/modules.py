
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from collections.abc import Iterable
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        sigma = 2 / (self.in_features + self.out_features)
        self.weights = torch.nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-(3*sigma), b=(3*sigma))
        
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
        
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weights = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids):
        return self.weights[token_ids]

def silu(x):
    return x * torch.sigmoid(x)

class SiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        self.w1_weights = torch.nn.Parameter(torch.zeros(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weights = torch.nn.Parameter(torch.zeros(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weights = torch.nn.Parameter(torch.zeros(d_ff, d_model, device=device, dtype=dtype))
        
        torch.nn.init.trunc_normal_(self.w1_weights, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w2_weights, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w3_weights, mean=0, std=1, a=-3, b=3)
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        out1 = einsum(self.w1_weights, x, "d_ff d_model, ... d_model -> ... d_ff")
        left = out1 * torch.sigmoid(out1)
        right = einsum(self.w3_weights, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2_weights, left * right, "d_model d_ff, ... d_ff -> ... d_model")

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, "... d_model"]:
        intype = x.dtype
        x = x.to(torch.float32)
        s = einsum(x * x, "... d_model -> ...")
        rms = torch.sqrt((1 / self.d_model) * s + self.eps).unsqueeze(-1)
        result = einsum(x / rms, self.gain, "... d_model, d_model -> ... d_model") 
        return result.to(intype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: int, d_k: int, max_seq_len: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        ks = torch.arange(1, self.d_k/2+1)
        theta_ik = torch.arange(self.max_seq_len).unsqueeze(-1) / (self.theta ** ((2 * ks - 2) / self.d_k))
        
        self.sin_cached = torch.sin(theta_ik).to(device)
        self.cos_cached = torch.cos(theta_ik).to(device)
        self.register_buffer("sin", self.sin_cached, persistent=False)
        self.register_buffer("cos", self.cos_cached, persistent=False)
        indices = []
        for i in range(0, self.d_k, 2):
            indices.append(i + 1)
            indices.append(i)
        self.neg_indices = torch.tensor(indices, device=device)
        self.neg_mask = torch.tensor([-1, 1] * (d_k // 2), device=device)

                
    def forward(self, query: Float[Tensor, " ... sequence_length d_k"], positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:        
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        seq_len = len(positions[0])
        
        sins = torch.repeat_interleave(self.sin_cached, repeats=2, dim=1)[:seq_len, :]
        coss = torch.repeat_interleave(self.cos_cached, repeats=2, dim=1)[:seq_len, :]
        
        neg_query = query[..., self.neg_indices] * self.neg_mask
        
        return query * coss + neg_query * sins
        
def softmax(x, i):
    rescaled = x - torch.max(x, dim=i).values.unsqueeze(-1)
    exp = torch.exp(rescaled)
    return exp / torch.sum(exp, dim=i).unsqueeze(-1)
    
def scaled_dot_product_attention(query, key, value, mask=None):
    pre = einsum(query, key, "b ... queries d_k, b ... keys d_k -> b ... queries keys") / torch.sqrt(torch.tensor([query.shape[-1]], device=query.device))
    if mask is not None:
        pre.masked_fill_(~mask, -torch.inf)
    post = softmax(pre, -1)
    result = einsum(post, value, "b ... queries keys, b ... keys d_k -> b ... queries d_k")
    return result
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 256, device: str = "cpu", dtype: str | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        self.d_head = int(self.d_model / self.num_heads)
        self.mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, device=device)) > 0
        self.w_q = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_k = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_v = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_o = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        
        torch.nn.init.trunc_normal_(self.w_q, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w_k, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w_v, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w_o, mean=0, std=1, a=-3, b=3)
        
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_len = x.shape[-2]
        
        Q = einsum(self.w_q, x, "... d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        V = einsum(self.w_v, x, "... d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        K = einsum(self.w_k, x, "... d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        
        Q = rearrange(Q, "b seq_len (num_heads d_head) -> b num_heads seq_len d_head", num_heads=self.num_heads)
        V = rearrange(V, "b seq_len (num_heads d_head) -> b num_heads seq_len d_head", num_heads=self.num_heads)
        K = rearrange(K, "b seq_len (num_heads d_head) -> b num_heads seq_len d_head", num_heads=self.num_heads)
  
        mask = self.mask[:seq_len, :seq_len]
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "b num_heads seq_len d_head -> b seq_len (num_heads d_head)")
        result = einsum(self.w_o, attention, "d_out d_in, ... d_in -> ... d_out")
        return result
        

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: int, max_seq_len: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        self.d_head = int(self.d_model / self.num_heads)
        self.mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, device=device)) > 0
        self.w_q = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_k = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_v = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        self.w_o = torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, device=device, dtype=dtype))
        
        torch.nn.init.trunc_normal_(self.w_q, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w_k, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w_v, mean=0, std=1, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.w_o, mean=0, std=1, a=-3, b=3)
        
        self.RoPE = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device, dtype=dtype)
        
    def forward(self, x: Float[Tensor, " ... sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_len = x.shape[-2]
        
        Q = einsum(self.w_q, x, "... d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        V = einsum(self.w_v, x, "... d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        K = einsum(self.w_k, x, "... d_out d_in, ... seq_len d_in -> ... seq_len d_out")

        Q = rearrange(Q, "b seq_len (num_heads d_head) -> b num_heads seq_len d_head", num_heads=self.num_heads)
        V = rearrange(V, "b seq_len (num_heads d_head) -> b num_heads seq_len d_head", num_heads=self.num_heads)
        K = rearrange(K, "b seq_len (num_heads d_head) -> b num_heads seq_len d_head", num_heads=self.num_heads)
                
        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)
        
        mask = self.mask[:seq_len, :seq_len]
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "b num_heads seq_len d_head -> b seq_len (num_heads d_head)")
        result = einsum(self.w_o, attention, "d_out d_in, ... d_in -> ... d_out")
        return result