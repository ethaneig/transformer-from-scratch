import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Optional
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
import numpy as np
from tqdm import tqdm, trange
import wandb

from transformer.tokenization import train_bpe, Tokenizer, from_files
from transformer.modules import Linear, Embedding, silu, SiGLU, RMSNorm, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, MultiHeadSelfAttention, MultiHeadSelfAttentionRoPE
from transformer.transformer import TransformerBlock, Transformer
from transformer.utils import cross_entropy, perplexity, AdamW, lr_cosine_schedule, gradient_clip, get_batch, save_checkpoint, load_checkpoint


BASE_DIR = "."
DATA_DIR = f"{BASE_DIR}/data"

def train(cfg: dict[str, Any]) -> None: 
    device = cfg["global_args"]["device"]
    experiment_name = cfg["global_args"]["experiment_name"]
    experiment_path = f"{BASE_DIR}/experiments/{experiment_name}"
    checkpoint_path = f"{experiment_path}/checkpoints/"
    seed = cfg["global_args"]["seed"]
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    tokenizer_path = f"{BASE_DIR}/{cfg["tokenizer_args"]["tokenizer_path"]}"
    vocab_size = cfg["tokenizer_args"]["vocab_size"]
    special_tokens = [token for token in cfg["tokenizer_args"]["special_tokens"]]
    vocab_filepath = f"{tokenizer_path}/{cfg["tokenizer_args"]["vocab_filename"]}"
    merges_filepath = f"{tokenizer_path}/{cfg["tokenizer_args"]["merges_filename"]}"

    num_steps = cfg["train_args"]["num_steps"]
    learning_rate = cfg["train_args"]["learning_rate"]
    beta1 = cfg["train_args"]["beta1"]
    beta2 = cfg["train_args"]["beta2"]
    eps = cfg["train_args"]["eps"]
    weight_decay = cfg["train_args"]["weight_decay"]
    check_freq = cfg["train_args"]["check_freq"]
    val_freq = cfg["train_args"]["val_freq"]
    log_freq = cfg["train_args"]["log_freq"]

    context_length = cfg["model_args"]["context_length"]
    d_model = cfg["model_args"]["d_model"]
    d_ff = cfg["model_args"]["d_ff"]
    num_layers = cfg["model_args"]["num_layers"]
    num_heads = cfg["model_args"]["num_heads"]
    rope_theta = cfg["model_args"]["rope_theta"]

    batch_size = cfg["dataset_args"]["batch_size"]
    train_dataset_path = f"{DATA_DIR}/{cfg["dataset_args"]["train_dataset_filename"]}"
    tokenized_train_path = f"{DATA_DIR}/{cfg["dataset_args"]["tokenized_train_filename"]}"
    val_dataset_path = f"{DATA_DIR}/{cfg["dataset_args"]["val_dataset_filename"]}"
    tokenized_valid_path = f"{DATA_DIR}/{cfg["dataset_args"]["tokenized_val_filename"]}"
    
    OmegaConf.save(config=cfg, f=f"{experiment_path}/config.yaml")
    
    print("Total_tokens to be processed:", batch_size * num_steps * context_length)
    torch.manual_seed(seed)
    
    if os.path.exists(vocab_filepath) and os.path.exists(merges_filepath):
        print("Loading tokenizer...")
        tokenizer = from_files(vocab_filepath, merges_filepath, special_tokens)
    else:
        print("Training tokenizer...")
        vocab, merges = train_bpe(train_dataset_path, vocab_size, special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        tokenizer.to_files(vocab_filepath, merges_filepath)
    
    print("Loading training dataset...")
    if os.path.exists(tokenized_train_path):
        train_dataset = np.load(tokenized_train_path)
    else:
        print("Tokenizing training dataset...")
        train_dataset = tokenizer.encode_iterable(open(train_dataset_path, "r", encoding="utf-8"))
        np.save(tokenized_train_path, np.array(train_dataset))
    
    print("Loading validation dataset...")
    if os.path.exists(tokenized_valid_path):
        val_dataset = np.load(tokenized_valid_path)
    else:
        print("Tokenizing validation dataset...")
        val_dataset = tokenizer.encode_iterable(open(val_dataset_path, "r", encoding="utf-8"))
        np.save(tokenized_valid_path, np.array(val_dataset))
    
    print("Loading model...")
    model = Transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device)
    optimizer = AdamW(model.parameters())
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pth")]
    best_model = [f for f in checkpoint_files if f.startswith("best_model")]
    if not checkpoint_files and not best_model:
        print("No checkpoints found. Starting training from scratch.")
        start_step = 0
    else:
        if best_model:
            checkpoint_files = best_model
        elif checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_files[-1])
        start_step = load_checkpoint(checkpoint_filepath, model, optimizer)
        print(f"Loading checkpoint from {checkpoint_filepath}, resuming training from step {start_step}...")

    torch.compile(model, backend='aot_eager')

    print("Starting training...")
    wandb.login()
    wandb.init(project="cs336-assignment1-basics", config=cfg.train_args, name=experiment_name)
    
    best = float("inf")
    for step in trange(start_step, num_steps):
        batch, targets = get_batch(train_dataset, batch_size, context_length, device)
        
        optimizer.zero_grad()
        logits = model(batch)
        targets = targets
        
        loss = cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()
        
        if step % log_freq == 0:
            wandb.log({"loss": loss.item(), "step": step})
        if step != 0 and step % check_freq == 0:
            save_checkpoint(model, optimizer, step, f"{checkpoint_path}/step_{step}.pth")
            print(f"Epoch {step}/{num_steps} Loss {loss.item() if 'loss' in locals() else 'N/A'}")
            if loss.item() < best:
                best = loss.item()
                save_checkpoint(model, optimizer, step, f"{checkpoint_path}/best_model.pth")
        if step != 0 and step % val_freq == 0:
            print(f"Validation at step {step}...")
            val_batch, val_targets = get_batch(val_dataset, batch_size, context_length, device)
            with torch.no_grad():
                val_logits = model(val_batch)
                val_loss = perplexity(val_logits, val_targets).mean()
                wandb.log({"val_loss": val_loss.item(), "step": step})
                print(f"Validation Loss: {val_loss.item() if 'val_loss' in locals() else 'N/A'}")
    print(f"Training completed. Final Loss: {loss.item() if 'loss' in locals() else 'N/A'}")
    print(f"Saving final model checkpoint to {checkpoint_path}/best_model.pth")
    save_checkpoint(model, optimizer, step, f"{checkpoint_path}/best_model.pth")    

@hydra.main(config_path="cfgs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()