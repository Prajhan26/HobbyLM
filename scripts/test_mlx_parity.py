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

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        import json

        (path / "config.json").write_text(json.dumps(cfg.to_dict()))
        state = {name: value.detach().cpu().contiguous().clone() for name, value in torch_model.state_dict().items()}
        save_file(state, path / "model.safetensors")

        import mlx.core as mx
        from mlx_hobbylm.weights import load_local

        mlx_model, _ = load_local(path)
        actual = mlx_model(mx.array(token_ids.numpy()))
        mx.eval(actual)
        actual_np = np.asarray(actual)

    max_error = float(np.max(np.abs(expected - actual_np)))
    mean_error = float(np.mean(np.abs(expected - actual_np)))
    print(f"max_abs_error={max_error:.8f}")
    print(f"mean_abs_error={mean_error:.8f}")
    if not np.allclose(expected, actual_np, rtol=2e-4, atol=2e-4):
        raise SystemExit("FAIL: MLX output does not match the PyTorch reference")
    print("PASS: MLX output matches the PyTorch reference")


if __name__ == "__main__":
    main()
