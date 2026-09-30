"""Load and adapt public HobbyLM Safetensors weights for the MLX graph."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .model import HobbyLM, HobbyLMConfig


def _adapt_weights(raw: dict[str, mx.array], cfg: HobbyLMConfig) -> dict[str, mx.array]:
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
        allow_patterns=["config.json", "model.safetensors"],
    )
    return Path(path)


def load(repo_or_path: str, cache_dir: str | Path | None = None):
    path = Path(repo_or_path).expanduser()
    if not path.exists():
        path = download(repo_or_path, cache_dir)
    return load_local(path)
