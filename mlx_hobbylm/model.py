"""HobbyLM's inference graph implemented directly in Apple MLX.

The module mirrors ``hobbylm.model`` closely so a parity test can compare the
PyTorch reference implementation and this Apple-Silicon runtime layer by layer.
Training-only router losses and bias updates are deliberately omitted.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class HobbyLMConfig:
    vocab_size: int = 50304
    d_model: int = 768
    n_layers: int = 16
    n_dense_layers: int = 1
    n_q_heads: int = 12
    n_kv_heads: int = 3
    head_dim: int = 128
    qk_norm: bool = True
    rope_theta: float = 1_000_000.0
    dense_ffn: int = 2304
    expert_ffn: int = 320
    n_experts: int = 36
    top_k: int = 6
    n_shared: int = 1
    gating: str = "sigmoid"
    norm_topk_prob: bool = False
    tie_embeddings: bool = True
    scale_embeddings: bool = False
    logit_softcap: float = 0.0
    rms_eps: float = 1e-6

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HobbyLMConfig":
        allowed = {f.name for f in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        return cls(**values)


def _rotate_half_rope(x: mx.array, *, offset: int, theta: float) -> mx.array:
    """Apply the same split-half RoPE convention used by the PyTorch model."""
    length, dim = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (
        theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    )
    positions = mx.arange(offset, offset + length, dtype=mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs).astype(x.dtype)[None, None, :, :]
    sin = mx.sin(freqs).astype(x.dtype)[None, None, :, :]
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return mx.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1)


class Attention(nn.Module):
    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(
            cfg.d_model,
            (cfg.n_q_heads + 2 * cfg.n_kv_heads) * cfg.head_dim,
            bias=False,
        )
        self.proj = nn.Linear(cfg.n_q_heads * cfg.head_dim, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
            self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_eps)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        cfg = self.cfg
        joined = self.qkv(x)
        q_end = cfg.n_q_heads * cfg.head_dim
        k_end = q_end + cfg.n_kv_heads * cfg.head_dim
        q = joined[..., :q_end]
        k = joined[..., q_end:k_end]
        v = joined[..., k_end:]
        q = q.reshape(batch, length, cfg.n_q_heads, cfg.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, cfg.n_kv_heads, cfg.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, cfg.n_kv_heads, cfg.head_dim).transpose(0, 2, 1, 3)
        if cfg.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = _rotate_half_rope(q, offset=0, theta=cfg.rope_theta)
        k = _rotate_half_rope(k, offset=0, theta=cfg.rope_theta)
        repeat = cfg.n_q_heads // cfg.n_kv_heads
        if repeat > 1:
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length).astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=cfg.head_dim**-0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.proj(out)


class DenseFFN(nn.Module):
    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.dense_ffn, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.dense_ffn, bias=False)
        self.down_proj = nn.Linear(cfg.dense_ffn, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertBank(nn.Module):
    """Sparse selected-expert SwiGLU backed by MLX gather matrix multiplies."""

    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU

        self.layers = SwitchGLU(cfg.d_model, cfg.expert_ffn, cfg.n_experts)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return self.layers(x, indices)


class SharedExpert(nn.Module):
    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.expert_ffn, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.expert_ffn, bias=False)
        self.down_proj = nn.Linear(cfg.expert_ffn, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = ExpertBank(cfg)
        self.expert_bias = mx.zeros((cfg.n_experts,), dtype=mx.float32)
        if cfg.n_shared:
            if cfg.n_shared != 1:
                raise ValueError("The first MLX release supports one shared expert")
            self.shared = SharedExpert(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        cfg = self.cfg
        logits = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(logits) if cfg.gating == "sigmoid" else mx.softmax(logits, axis=-1)
        selection = scores + self.expert_bias
        indices = mx.stop_gradient(
            mx.argpartition(-selection, kth=cfg.top_k - 1, axis=-1)[..., : cfg.top_k]
        )
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if cfg.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)
        routed = self.experts(x, indices)
        out = (routed * weights.astype(routed.dtype)[..., None]).sum(axis=-2)
        if cfg.n_shared:
            out = out + self.shared(x)
        return out


class Block(nn.Module):
    def __init__(self, cfg: HobbyLMConfig, layer_index: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn = DenseFFN(cfg) if layer_index < cfg.n_dense_layers else SparseMoE(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class HobbyLM(nn.Module):
    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg, index) for index in range(cfg.n_layers)]
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> mx.array:
        x = self.embed(token_ids)
        if self.cfg.scale_embeddings:
            x = x * (self.cfg.d_model**0.5)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.embed.as_linear(x) if self.cfg.tie_embeddings else self.lm_head(x)
        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = cap * mx.tanh(logits / cap)
        return logits
