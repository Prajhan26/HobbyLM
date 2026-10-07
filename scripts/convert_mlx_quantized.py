"""Create a local weight-only MLX quantization of a HobbyLM checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_hobbylm.weights import load, quantize_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="harims95/hobbylm-1b-broad-sft-3450-hf")
    parser.add_argument("--source-model-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--scope",
        choices=(
            "all_weights_except_moe_router",
            "routed_experts_only",
            "routed_expert_down_only",
        ),
        default="routed_experts_only",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model, _ = load(args.model)
    quantization = {
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": "affine",
        "scope": args.scope,
    }
    quantize_model(model, quantization)
    mx.eval(model.parameters())
    for block in model.blocks:
        if hasattr(block.ffn, "last_topi"):
            block.ffn.last_topi = None
            block.ffn.last_topv = None
    model.save_weights(str(output / "model.safetensors"))

    source = Path(args.model).expanduser()
    if not source.exists():
        from mlx_hobbylm.weights import download

        source = download(args.model)
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    config["mlx_router_precision"] = "float32"
    config["mlx_quantization_scope"] = args.scope
    config["mlx_source_model"] = args.source_model_id or args.model
    if args.source_revision:
        config["mlx_source_revision"] = args.source_revision
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("generation_config.json", "README.md"):
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, output / name)

    weights = dict(tree_flatten(model.parameters(), destination=[]))
    router_dtypes = {
        value.dtype for key, value in weights.items() if key.endswith("ffn.gate.weight")
    }
    if router_dtypes != {mx.float32}:
        raise SystemExit(f"Router precision changed unexpectedly: {router_dtypes}")

    weight_file = output / "model.safetensors"
    manifest = {
        "source_model": args.source_model_id or args.model,
        "source_revision": args.source_revision,
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": "affine",
        "scope": args.scope,
        "weight_only": True,
        "router_precision": "float32",
        "router_quantized": False,
        "weight_bytes": weight_file.stat().st_size,
        "weight_sha256": sha256(weight_file),
    }
    (output / "quantization-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
