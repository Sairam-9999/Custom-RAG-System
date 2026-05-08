"""GPT-2 style model used for local RAG fine-tuning.

This file keeps your original architecture, and adds:
1. GPT-2 checkpoint weight loading helpers
2. text generation helper
3. small config utilities
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

MODEL_CONFIGS = {
    "124M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "355M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "774M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "1558M": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Input length {seq_len} exceeds context length {self.cfg['context_length']}"
            )

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def _assign(left: torch.nn.Parameter, right: np.ndarray, name: str) -> None:
    """Copy numpy GPT-2 checkpoint array into a PyTorch parameter."""
    right_t = torch.tensor(right, dtype=left.dtype)
    if left.shape != right_t.shape:
        raise ValueError(f"Shape mismatch for {name}: left={left.shape}, right={right_t.shape}")
    with torch.no_grad():
        left.copy_(right_t)


def load_weights_into_gpt(model: GPTModel, params: dict) -> None:
    """Load OpenAI GPT-2 TensorFlow checkpoint params into this PyTorch GPT model.

    Expected params format comes from gpt_download_updated.download_and_load_gpt2().
    """
    _assign(model.pos_emb.weight, params["wpe"], "pos_emb.weight")
    _assign(model.tok_emb.weight, params["wte"], "tok_emb.weight")

    for b in range(len(params["blocks"])):
        block = model.trf_blocks[b]
        src = params["blocks"][b]

        # TensorFlow GPT-2 stores QKV combined as [emb_dim, 3 * emb_dim].
        q_w, k_w, v_w = np.split(src["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(src["attn"]["c_attn"]["b"], 3, axis=-1)

        _assign(block.att.W_query.weight, q_w.T, f"blocks.{b}.att.W_query.weight")
        _assign(block.att.W_key.weight, k_w.T, f"blocks.{b}.att.W_key.weight")
        _assign(block.att.W_value.weight, v_w.T, f"blocks.{b}.att.W_value.weight")
        _assign(block.att.W_query.bias, q_b, f"blocks.{b}.att.W_query.bias")
        _assign(block.att.W_key.bias, k_b, f"blocks.{b}.att.W_key.bias")
        _assign(block.att.W_value.bias, v_b, f"blocks.{b}.att.W_value.bias")

        _assign(block.att.out_proj.weight, src["attn"]["c_proj"]["w"].T, f"blocks.{b}.att.out_proj.weight")
        _assign(block.att.out_proj.bias, src["attn"]["c_proj"]["b"], f"blocks.{b}.att.out_proj.bias")

        _assign(block.ff.layers[0].weight, src["mlp"]["c_fc"]["w"].T, f"blocks.{b}.ff.fc.weight")
        _assign(block.ff.layers[0].bias, src["mlp"]["c_fc"]["b"], f"blocks.{b}.ff.fc.bias")
        _assign(block.ff.layers[2].weight, src["mlp"]["c_proj"]["w"].T, f"blocks.{b}.ff.proj.weight")
        _assign(block.ff.layers[2].bias, src["mlp"]["c_proj"]["b"], f"blocks.{b}.ff.proj.bias")

        _assign(block.norm1.scale, src["ln_1"]["g"], f"blocks.{b}.norm1.scale")
        _assign(block.norm1.shift, src["ln_1"]["b"], f"blocks.{b}.norm1.shift")
        _assign(block.norm2.scale, src["ln_2"]["g"], f"blocks.{b}.norm2.scale")
        _assign(block.norm2.shift, src["ln_2"]["b"], f"blocks.{b}.norm2.shift")

    _assign(model.final_norm.scale, params["g"], "final_norm.scale")
    _assign(model.final_norm.shift, params["b"], "final_norm.shift")

    # GPT-2 ties output projection to token embedding weights.
    with torch.no_grad():
        model.out_head.weight.copy_(model.tok_emb.weight)


@torch.no_grad()
def generate_text_simple(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.3,
    top_k: int | None = 40,
    repetition_penalty: float = 1.0,
    device: str = "cpu",
) -> str:
    """Small generation helper for testing the fine-tuned checkpoint."""
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    context_length = model.cfg["context_length"]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)[:, -1, :]

        # Apply repetition penalty to discourage repeated tokens
        if repetition_penalty != 1.0:
            # Get token counts in the current sequence
            unique_tokens, counts = torch.unique(idx, return_counts=True)
            # Create a penalty mask
            penalty = torch.ones_like(logits)
            for token, count in zip(unique_tokens, counts):
                penalty[0, token] = repetition_penalty ** count
            logits = logits / penalty

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                top_vals, _ = torch.topk(logits, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val, torch.tensor(-float("inf"), device=device), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

        # GPT-2 endoftext token id
        if next_id.item() == 50256:
            break

    return tokenizer.decode(idx[0].tolist())
