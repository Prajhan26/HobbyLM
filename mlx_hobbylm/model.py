"""HobbyLM's inference graph implemented directly in Apple MLX.

The module mirrors ``hobbylm.model`` closely so a parity test can compare the
PyTorch reference implementation and this Apple-Silicon runtime layer by layer.
Training-only router losses and bias updates are deliberately omitted.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeAlias

import mlx.core as mx
import mlx.nn as nn


KVCache: TypeAlias = tuple[mx.array, mx.array]


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
    max_position_embeddings: int = 1024
    routed_scaling_factor: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HobbyLMConfig":
        # Public Hugging Face exports use Transformers names, while native
        # HobbyLM checkpoints use the shorter training-config names above.
        aliases = {
            "hidden_size": "d_model",
            "num_hidden_layers": "n_layers",
            "first_k_dense_replace": "n_dense_layers",
            "num_attention_heads": "n_q_heads",
            "num_key_value_heads": "n_kv_heads",
            "use_qk_norm": "qk_norm",
            "intermediate_size": "dense_ffn",
            "moe_intermediate_size": "expert_ffn",
            "num_local_experts": "n_experts",
            "num_experts_per_tok": "top_k",
            "n_shared_experts": "n_shared",
            "tie_word_embeddings": "tie_embeddings",
            "rms_norm_eps": "rms_eps",
        }
        raw = {aliases.get(key, key): value for key, value in raw.items()}
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

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array | tuple[mx.array, KVCache]:
        batch, length, _ = x.shape
        cfg = self.cfg
        offset = 0 if cache is None else cache[0].shape[-2]
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
        q = _rotate_half_rope(q, offset=offset, theta=cfg.rope_theta)
        k = _rotate_half_rope(k, offset=offset, theta=cfg.rope_theta)
        if cache is not None:
            k = mx.concatenate((cache[0], k), axis=-2)
            v = mx.concatenate((cache[1], v), axis=-2)
        next_cache = (k, v)
        repeat = cfg.n_q_heads // cfg.n_kv_heads
        if repeat > 1:
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)
        key_length = k.shape[-2]
        mask = None
        if length > 1:
            blocked = mx.arange(key_length)[None, :] > (
                offset + mx.arange(length)[:, None]
            )
            mask = mx.where(blocked, -mx.inf, 0.0).astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=cfg.head_dim**-0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        out = self.proj(out)
        return (out, next_cache) if use_cache else out


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
        self.last_topi = None
        self.last_topv = None
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
        weights = weights * cfg.routed_scaling_factor
        self.last_topi = indices
        self.last_topv = weights
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

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array | tuple[mx.array, KVCache]:
        if use_cache:
            attn_out, next_cache = self.attn(
                self.attn_norm(x), cache, use_cache=True
            )
        else:
            attn_out = self.attn(self.attn_norm(x))
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return (x, next_cache) if use_cache else x


class HobbyLM(nn.Module):
    def __init__(self, cfg: HobbyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg, index) for index in range(cfg.n_layers)]
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | None]:
        return [None] * self.cfg.n_layers

    def __call__(
        self,
        token_ids: mx.array,
        cache: list[KVCache | None] | None = None,
    ) -> mx.array:
        if cache is not None and len(cache) != self.cfg.n_layers:
            raise ValueError(f"Expected {self.cfg.n_layers} cache entries, got {len(cache)}")
        x = self.embed(token_ids)
        if self.cfg.scale_embeddings:
            x = x * (self.cfg.d_model**0.5)
        for index, block in enumerate(self.blocks):
            if cache is None:
                x = block(x)
            else:
                x, cache[index] = block(x, cache[index], use_cache=True)
        x = self.final_norm(x)
        logits = self.embed.as_linear(x) if self.cfg.tie_embeddings else self.lm_head(x)
        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = cap * mx.tanh(logits / cap)
        return logits
