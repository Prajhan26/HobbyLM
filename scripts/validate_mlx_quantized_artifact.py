"""Validate the structure and provenance of a quantized MLX artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx

from mlx_hobbylm.weights import load_local


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.model / "config.json"
    weights_path = args.model / "model.safetensors"
    manifest_path = args.model / "quantization-manifest.json"
    for path in (config_path, weights_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"Missing required artifact file: {path}")

    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    quantization = config.get("quantization")
    failures: list[str] = []
    if quantization != {"group_size": 32, "bits": 8, "mode": "affine"}:
        failures.append(f"unexpected quantization config: {quantization}")
    if config.get("mlx_router_precision") != "float32":
        failures.append("config does not declare an FP32 router")
    if manifest.get("router_quantized") is not False:
        failures.append("manifest does not explicitly exclude router quantization")

    actual_hash = sha256(weights_path)
    if manifest.get("weight_sha256") != actual_hash:
        failures.append("weight SHA-256 does not match manifest")
    if manifest.get("weight_bytes") != weights_path.stat().st_size:
        failures.append("weight byte count does not match manifest")

    raw = mx.load(str(weights_path))
    diagnostic = sorted(
        key for key in raw if key.endswith("last_topi") or key.endswith("last_topv")
    )
    if diagnostic:
        failures.append(f"diagnostic tensors leaked into weights: {diagnostic}")
    routers = {
        key: str(value.dtype)
        for key, value in raw.items()
        if key.endswith("ffn.gate.weight")
    }
    if len(routers) != 19:
        failures.append(f"expected 19 sparse routers, found {len(routers)}")
    if {value for value in routers.values()} != {"mlx.core.float32"}:
        failures.append(f"router weights are not uniformly FP32: {routers}")

    # Strict loading proves that there are no unresolved missing or unexpected
    # tensors after reconstructing quantized modules from config.
    model, _ = load_local(args.model)
    del model
    mx.clear_cache()

    report = {
        "status": "pass" if not failures else "fail",
        "model": str(args.model),
        "source_model": manifest.get("source_model"),
        "source_revision": manifest.get("source_revision"),
        "quantization": quantization,
        "router_precision": "float32",
        "router_count": len(routers),
        "strict_load": True,
        "diagnostic_tensor_count": len(diagnostic),
        "weight_bytes": weights_path.stat().st_size,
        "weight_sha256": actual_hash,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("FAIL: quantized artifact structure is invalid")


if __name__ == "__main__":
    main()
