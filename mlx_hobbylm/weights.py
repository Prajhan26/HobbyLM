"""Load and adapt public HobbyLM Safetensors weights for the MLX graph."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .model import HobbyLM, HobbyLMConfig


def _adapt_native_weights(raw: dict[str, mx.array], cfg: HobbyLMConfig) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    weights["embed.weight"] = raw["embed.weight"]
    weights["final_norm.weight"] = raw["final_norm.weight"]
    if not cfg.tie_embeddings:
        weights["lm_head.weight"] = raw["lm_head.weight"]

    for layer in range(cfg.n_layers):
        old = f"blocks.{layer}."
        for name in (
            "attn_norm.weight",
            "attn.qkv.weight",
            "attn.q_norm.weight",
            "attn.k_norm.weight",
            "attn.proj.weight",
            "ffn_norm.weight",
        ):
            weights[old + name] = raw[old + name]

        if layer < cfg.n_dense_layers:
            fused = raw[old + "ffn.w13.weight"]
            weights[old + "ffn.gate_proj.weight"] = fused[: cfg.dense_ffn]
            weights[old + "ffn.up_proj.weight"] = fused[cfg.dense_ffn :]
            weights[old + "ffn.down_proj.weight"] = raw[old + "ffn.w2.weight"]
            continue

        weights[old + "ffn.gate.weight"] = raw[old + "ffn.gate.weight"]
        weights[old + "ffn.expert_bias"] = raw[old + "ffn.expert_bias"]
        fused = raw[old + "ffn.experts.w13"]
        down = raw[old + "ffn.experts.w2"]
        weights[old + "ffn.experts.layers.gate_proj.weight"] = fused[..., : cfg.expert_ffn].transpose(0, 2, 1)
        weights[old + "ffn.experts.layers.up_proj.weight"] = fused[..., cfg.expert_ffn :].transpose(0, 2, 1)
        weights[old + "ffn.experts.layers.down_proj.weight"] = down.transpose(0, 2, 1)

        if cfg.n_shared:
            shared = raw[old + "ffn.shared.w13"][0]
            weights[old + "ffn.shared.gate_proj.weight"] = shared[:, : cfg.expert_ffn].T
            weights[old + "ffn.shared.up_proj.weight"] = shared[:, cfg.expert_ffn :].T
            weights[old + "ffn.shared.down_proj.weight"] = raw[old + "ffn.shared.w2"][0].T
    return weights


def _adapt_hf_weights(raw: dict[str, mx.array], cfg: HobbyLMConfig) -> dict[str, mx.array]:
    """Map the official ``HobbyLMForCausalLM`` Safetensors layout to MLX."""
    weights: dict[str, mx.array] = {
        "embed.weight": raw["model.embed_tokens.weight"],
        "final_norm.weight": raw["model.norm.weight"],
    }
    if not cfg.tie_embeddings:
        weights["lm_head.weight"] = raw["lm_head.weight"]

    for layer in range(cfg.n_layers):
        src = f"model.layers.{layer}."
        dst = f"blocks.{layer}."
        weights[dst + "attn_norm.weight"] = raw[src + "input_layernorm.weight"]
        weights[dst + "ffn_norm.weight"] = raw[src + "post_attention_layernorm.weight"]
        weights[dst + "attn.qkv.weight"] = mx.concatenate(
            (
                raw[src + "self_attn.q_proj.weight"],
                raw[src + "self_attn.k_proj.weight"],
                raw[src + "self_attn.v_proj.weight"],
            ),
            axis=0,
        )
        weights[dst + "attn.proj.weight"] = raw[src + "self_attn.o_proj.weight"]
        if cfg.qk_norm:
            weights[dst + "attn.q_norm.weight"] = raw[src + "self_attn.q_norm.weight"]
            weights[dst + "attn.k_norm.weight"] = raw[src + "self_attn.k_norm.weight"]

        if layer < cfg.n_dense_layers:
            weights[dst + "ffn.gate_proj.weight"] = raw[src + "mlp.gate_proj.weight"]
            weights[dst + "ffn.up_proj.weight"] = raw[src + "mlp.up_proj.weight"]
            weights[dst + "ffn.down_proj.weight"] = raw[src + "mlp.down_proj.weight"]
            continue

        weights[dst + "ffn.gate.weight"] = raw[src + "mlp.gate.weight"]
        weights[dst + "ffn.expert_bias"] = raw[src + "mlp.gate.expert_bias"]
        gate_up = raw[src + "mlp.experts.gate_up_proj"]
        weights[dst + "ffn.experts.layers.gate_proj.weight"] = gate_up[:, : cfg.expert_ffn, :]
        weights[dst + "ffn.experts.layers.up_proj.weight"] = gate_up[:, cfg.expert_ffn :, :]
        weights[dst + "ffn.experts.layers.down_proj.weight"] = raw[src + "mlp.experts.down_proj"]

        if cfg.n_shared:
            shared = raw[src + "mlp.shared_experts.experts.gate_up_proj"][0]
            weights[dst + "ffn.shared.gate_proj.weight"] = shared[: cfg.expert_ffn]
            weights[dst + "ffn.shared.up_proj.weight"] = shared[cfg.expert_ffn :]
            weights[dst + "ffn.shared.down_proj.weight"] = raw[
                src + "mlp.shared_experts.experts.down_proj"
            ][0]
    return weights


def _adapt_weights(raw: dict[str, mx.array], cfg: HobbyLMConfig) -> dict[str, mx.array]:
    if "model.embed_tokens.weight" in raw:
        return _adapt_hf_weights(raw, cfg)
    if "embed.weight" in raw:
        return _adapt_native_weights(raw, cfg)
    raise ValueError("Unrecognized HobbyLM weight layout")


def load_local(model_dir: str | Path) -> tuple[HobbyLM, HobbyLMConfig]:
    model_dir = Path(model_dir)
    cfg = HobbyLMConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
    model = HobbyLM(cfg)
    raw = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(_adapt_weights(raw, cfg).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, cfg


def download(repo_id: str, cache_dir: str | Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=["config.json", "generation_config.json", "model.safetensors"],
    )
    return Path(path)


def load(repo_or_path: str, cache_dir: str | Path | None = None):
    path = Path(repo_or_path).expanduser()
    if not path.exists():
        path = download(repo_or_path, cache_dir)
    return load_local(path)
