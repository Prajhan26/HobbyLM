"""Exercise quantized HobbyLM construction and strict weight round-tripping."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_hobbylm.model import HobbyLM, HobbyLMConfig
from mlx_hobbylm.weights import load_local, quantize_model


def main() -> None:
    cfg = HobbyLMConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_dense_layers=1,
        n_q_heads=2,
        n_kv_heads=1,
        head_dim=32,
        dense_ffn=128,
        expert_ffn=32,
        n_experts=4,
        top_k=2,
        max_position_embeddings=64,
    )
    quantization = {"group_size": 32, "bits": 8, "mode": "affine"}
    model = HobbyLM(cfg)
    quantize_model(model, quantization)
    mx.eval(model.parameters())

    assert model.blocks[1].ffn.gate.weight.dtype == mx.float32
    assert isinstance(model.blocks[1].ffn.experts.layers.gate_proj, QuantizedSwitchLinear)

    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    expected = model(tokens)
    mx.eval(expected)
    model.blocks[1].ffn.last_topi = None
    model.blocks[1].ffn.last_topv = None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        config = asdict(cfg) | {
            "quantization": quantization,
            "mlx_router_precision": "float32",
        }
        (path / "config.json").write_text(json.dumps(config))
        model.save_weights(str(path / "model.safetensors"))
        loaded, _ = load_local(path)
        actual = loaded(tokens)
        mx.eval(actual)

    error = float(mx.max(mx.abs(expected - actual)).item())
    if error != 0.0:
        raise SystemExit(f"Quantized round-trip changed logits: max error {error}")
    print("PASS: quantized strict load, FP32 router, and logits round-trip")


if __name__ == "__main__":
    main()
