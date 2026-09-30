"""Numerical smoke test: the same tiny random HobbyLM in PyTorch and MLX."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hobbylm.config import ModelConfig
from hobbylm.model import MoETransformer


def as_hf_export(state: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Create the public HF tensor schema from a tiny native state dict."""
    out = {
        "model.embed_tokens.weight": state["embed.weight"],
        "lm_head.weight": state["embed.weight"].clone(),
        "model.norm.weight": state["final_norm.weight"],
    }
    q = cfg.n_q_heads * cfg.head_dim
    kv = cfg.n_kv_heads * cfg.head_dim
    for layer in range(cfg.n_layers):
        src = f"blocks.{layer}."
        dst = f"model.layers.{layer}."
        out[dst + "input_layernorm.weight"] = state[src + "attn_norm.weight"]
        out[dst + "post_attention_layernorm.weight"] = state[src + "ffn_norm.weight"]
        qkv = state[src + "attn.qkv.weight"]
        out[dst + "self_attn.q_proj.weight"] = qkv[:q]
        out[dst + "self_attn.k_proj.weight"] = qkv[q : q + kv]
        out[dst + "self_attn.v_proj.weight"] = qkv[q + kv :]
        out[dst + "self_attn.o_proj.weight"] = state[src + "attn.proj.weight"]
        out[dst + "self_attn.q_norm.weight"] = state[src + "attn.q_norm.weight"]
        out[dst + "self_attn.k_norm.weight"] = state[src + "attn.k_norm.weight"]
        if layer < cfg.n_dense_layers:
            fused = state[src + "ffn.w13.weight"]
            out[dst + "mlp.gate_proj.weight"] = fused[: cfg.dense_ffn]
            out[dst + "mlp.up_proj.weight"] = fused[cfg.dense_ffn :]
            out[dst + "mlp.down_proj.weight"] = state[src + "ffn.w2.weight"]
            continue
        out[dst + "mlp.gate.weight"] = state[src + "ffn.gate.weight"]
        out[dst + "mlp.gate.expert_bias"] = state[src + "ffn.expert_bias"]
        out[dst + "mlp.experts.gate_up_proj"] = state[src + "ffn.experts.w13"].transpose(1, 2).contiguous()
        out[dst + "mlp.experts.down_proj"] = state[src + "ffn.experts.w2"].transpose(1, 2).contiguous()
        out[dst + "mlp.shared_experts.experts.gate_up_proj"] = state[src + "ffn.shared.w13"].transpose(1, 2).contiguous()
        out[dst + "mlp.shared_experts.experts.down_proj"] = state[src + "ffn.shared.w2"].transpose(1, 2).contiguous()
    return out


def hf_config(cfg: ModelConfig) -> dict:
    return {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "num_hidden_layers": cfg.n_layers,
        "first_k_dense_replace": cfg.n_dense_layers,
        "num_attention_heads": cfg.n_q_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "use_qk_norm": cfg.qk_norm,
        "rope_theta": cfg.rope_theta,
        "intermediate_size": cfg.dense_ffn,
        "moe_intermediate_size": cfg.expert_ffn,
        "num_local_experts": cfg.n_experts,
        "num_experts_per_tok": cfg.top_k,
        "n_shared_experts": cfg.n_shared,
        "gating": cfg.gating,
        "norm_topk_prob": cfg.norm_topk_prob,
        "tie_word_embeddings": cfg.tie_embeddings,
        "scale_embeddings": cfg.scale_embeddings,
        "logit_softcap": cfg.logit_softcap,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1024,
        "routed_scaling_factor": 1.0,
    }


def main() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=3,
        n_dense_layers=1,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=8,
        dense_ffn=64,
        expert_ffn=16,
        n_experts=4,
        top_k=2,
        n_shared=1,
        expert_backend="bmm",
        tie_embeddings=True,
    )
    torch_model = MoETransformer(cfg).eval().float()
    token_ids = torch.tensor([[1, 17, 4, 91, 3]], dtype=torch.long)
    with torch.no_grad():
        expected = torch_model(token_ids)[0].numpy()

    state = {name: value.detach().cpu().contiguous().clone() for name, value in torch_model.state_dict().items()}
    import json
    import mlx.core as mx
    from mlx_hobbylm.weights import load_local

    layouts = {
        "native": (cfg.to_dict(), state),
        "huggingface": (hf_config(cfg), as_hf_export(state, cfg)),
    }
    for layout, (config, tensors) in layouts.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(json.dumps(config))
            save_file(tensors, path / "model.safetensors")
            mlx_model, _ = load_local(path)
            actual = mlx_model(mx.array(token_ids.numpy()))
            mx.eval(actual)
            actual_np = np.asarray(actual)

        max_error = float(np.max(np.abs(expected - actual_np)))
        mean_error = float(np.mean(np.abs(expected - actual_np)))
        print(f"{layout}: max_abs_error={max_error:.8f} mean_abs_error={mean_error:.8f}")
        if not np.allclose(expected, actual_np, rtol=2e-4, atol=2e-4):
            raise SystemExit(f"FAIL: {layout} MLX output does not match the PyTorch reference")
        print(f"PASS: {layout} MLX output matches the PyTorch reference")


if __name__ == "__main__":
    main()
