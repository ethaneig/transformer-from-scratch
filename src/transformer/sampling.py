

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Optional

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
import numpy as np
from tqdm import tqdm, trange
import wandb
import random

from transformer.tokenization import train_bpe, Tokenizer, from_files
from transformer.modules import Linear, Embedding, silu, SiGLU, RMSNorm, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, MultiHeadSelfAttention, MultiHeadSelfAttentionRoPE
from transformer.transformer import TransformerBlock, Transformer
from transformer.utils import cross_entropy, AdamW, lr_cosine_schedule, gradient_clip, get_batch, save_checkpoint, load_checkpoint

BASE_DIR = "."
DATA_DIR = f"{BASE_DIR}/data"

def sample_from_transformer(transformer: Transformer, intext: torch.Tensor, top_k: Optional[int] = None) -> int:
    logits = transformer(intext).reshape(-1, transformer.vocab_size)[-1]
    rand = random.random()
    next_token = None
    if top_k is not None:
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        top_k_probs = softmax(top_k_logits, -1)
        cum = 0
        idx = 0
        while cum < rand:
            cum += top_k_probs[idx]
            idx += 1
        next_token = top_k_indices[idx - 1]
    else:
        probs = softmax(logits[0, -1], -1)
        cum = 0
        idx = 0
        while cum < rand:
            cum += probs[idx]
            idx += 1
        next_token = idx - 1
    return next_token
    
def generate_text(tokenizer: Tokenizer, transformer: Transformer, prompt: Optional[str] = None, top_k: Optional[int] = None) -> None:
    print(prompt, end="")
    
    endoftext = tokenizer.rev_vocab[b'<|endoftext|>']
    if prompt is not None:
        text = torch.tensor(tokenizer.encode(prompt))
    else:
        text = torch.empty()
    
    while len(text) < transformer.context_length:
        sample = sample_from_transformer(transformer, text.unsqueeze(0), top_k).item()
        if sample == endoftext:
            break
        print(tokenizer.vocab[sample].decode('utf-8'), end="", flush=True)
        text = torch.cat((text, torch.tensor([sample])))
    
def generate():
    tokenizer = from_files(
        f"{DATA_DIR}/tokenizer_vocab_10000.json",
        f"{DATA_DIR}/tokenizer_merges_10000.txt",
        ["<|endoftext|>"]
    )
    model = Transformer(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
    )
    optimizer = AdamW(model.parameters())
    
    model, optimizer, _ = load_checkpoint(
        f"{BASE_DIR}/checkpoints/transformer_final.pth",
        model,
        optimizer
    )
    
    generate_text(
        tokenizer=tokenizer,
        transformer=model,
        prompt="Once upon a time",
        top_k=10
    )

def main():
    generate()

if __name__ == "__main__":
    main()
