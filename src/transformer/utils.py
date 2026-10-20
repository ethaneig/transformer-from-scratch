
import numpy as np
import torch
import math
from transformer.modules import softmax
from jaxtyping import Bool, Float, Int
from collections.abc import Iterable
from torch import Tensor
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], 
                       lr: float = 1e-3, 
                       betas: tuple[float, float] = (0.9, 0.999), 
                       weight_decay: float = 0.1, 
                       eps: float = 1e-8, 
                       device: str = "cpu"):
                       
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "lambd": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lambd = group["lambd"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                grad = p.grad.data
                p.data -= lr * lambd * p.data

                alpha = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                
                p.data -= alpha * m / (torch.sqrt(v) + eps) 
                
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss

def cross_entropy(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, " batch_size"]:
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return -torch.sum(log_probs) / log_probs.numel()

def perplexity(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> float:
    ce = cross_entropy(logits, targets)
    return torch.exp(ce)

def lr_cosine_schedule(t: int, a_max: float, a_min: float, tw: int, tc: int) -> float:
    if t < tw:
        return t / tw * a_max
    elif t > tc:
        return a_min
    return a_min + 0.5 * (1 + math.cos(math.pi * (t - tw) / (tc - tw))) * (a_max - a_min)

def gradient_clip(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    grads = torch.stack(grads)
    norm = torch.linalg.norm(grads)
    if norm > max_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad = p.grad * max_norm / (norm + eps)
            
def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    starting_indices = np.random.randint(0, len(x) - context_length, batch_size)
    idxs = starting_indices[:, None] + np.arange(context_length)
    samps = x[idxs]
    targets = x[idxs + 1]
    return (torch.tensor(samps, device=device), torch.tensor(targets, device=device))
    
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str) -> None:
    state_dict = {}
    state_dict["model"] = model.state_dict()
    state_dict["optimizer"] = optimizer.state_dict()
    state_dict["iteration"] = iteration
    torch.save(state_dict, out)

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]
