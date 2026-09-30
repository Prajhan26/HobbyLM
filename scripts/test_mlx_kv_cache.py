"""Check cached decoding against full-context MLX decoding."""
from __future__ import annotations

from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_hobbylm.model import HobbyLM, HobbyLMConfig


def route_snapshot(model: HobbyLM) -> list[np.ndarray]:
    routes = []
    for block in model.blocks:
        if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None:
            mx.eval(block.ffn.last_topi)
            routes.append(np.sort(np.asarray(block.ffn.last_topi)[-1, -1]))
    return routes


def main() -> None:
    mx.random.seed(7)
    cfg = HobbyLMConfig(
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
        max_position_embeddings=8,
    )
    model = HobbyLM(cfg)
    mx.eval(model.parameters())
    ids = [1, 17, 4, 91, 3]
    cache = model.make_cache()
    cached_input = mx.array([ids], dtype=mx.int32)

    for step in range(6):
        full_input = mx.array([ids[-cfg.max_position_embeddings :]], dtype=mx.int32)
        full_logits = model(full_input)[0, -1].astype(mx.float32)
        mx.eval(full_logits)
        full_routes = route_snapshot(model)

        cached_logits = model(cached_input, cache=cache)[0, -1].astype(mx.float32)
        mx.eval(cached_logits)
        cached_routes = route_snapshot(model)

        max_error = float(mx.max(mx.abs(full_logits - cached_logits)).item())
        if not mx.allclose(full_logits, cached_logits, rtol=2e-4, atol=2e-4).item():
            raise SystemExit(f"FAIL: cached logits differ at step {step}: {max_error:.8f}")
        if any(not np.array_equal(a, b) for a, b in zip(full_routes, cached_routes)):
            raise SystemExit(f"FAIL: cached expert routes differ at step {step}")

        token = int(mx.argmax(full_logits).item())
        ids.append(token)
        cached_length = cache[0][0].shape[-2]
        if cached_length >= cfg.max_position_embeddings:
            cache = model.make_cache()
            cached_input = mx.array(
                [ids[-cfg.max_position_embeddings :]], dtype=mx.int32
            )
        else:
            cached_input = mx.array([[token]], dtype=mx.int32)

    print("PASS: cached logits, tokens, and expert routes match full decoding")


if __name__ == "__main__":
    main()
