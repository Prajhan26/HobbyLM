"""Research-only long-context parity probe for the step-3450 SFT export.

The published HF config declares 1024 positions. This script deliberately
tests a longer input without modifying that config or the shipping CLI.
It does not certify long-context retrieval or a 4K release.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODEL = "harims95/hobbylm-1b-broad-sft-3450-hf"
FIXTURE = Path("artifacts/hobbylm-4k-candidate-reference.npz")
REPORT = Path("artifacts/hobbylm-4k-candidate-mlx-report.json")


def prompt_ids(length: int, variant: str = "archive") -> np.ndarray:
    tokenizer = tiktoken.get_encoding("gpt2")
    prefix = tokenizer.encode_ordinary(
        "SYSTEM: You are a helpful and concise assistant.\nUSER: "
    )
    suffix = tokenizer.encode_ordinary("\nASSISTANT:")
    fillers = {
        "archive": "The archive contains short notes about trees, weather, and maps. ",
        "science": "A field notebook records observations about stars, rivers, and minerals. ",
        "code": "The program reads a value, checks its type, and returns a result. ",
    }
    filler = tokenizer.encode_ordinary(fillers[variant] * length)
    available = length - len(prefix) - len(suffix)
    if available < 0:
        raise ValueError("Prompt length is too short")
    return np.asarray(prefix + filler[:available] + suffix, dtype=np.int32)


def create_reference(model_id: str, length: int, output: Path, variant: str = "archive") -> None:
    import torch
    from transformers import AutoModelForCausalLM

    inputs = prompt_ids(length, variant)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(
            input_ids=torch.from_numpy(inputs.astype(np.int64)).unsqueeze(0),
            use_cache=False,
        ).logits[0, -1].float().cpu().numpy()
        routes = np.stack(
            [
                layer.mlp.gate.last_topi[-1].cpu().numpy()
                for layer in model.model.layers
                if getattr(layer, "is_moe", False)
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, prompt_ids=inputs, logits=logits, routes=routes, variant=variant)
    print(
        json.dumps(
            {
                "status": "research_candidate_only",
                "backend": "pytorch",
                "model": model_id,
                "tokens": len(inputs),
                "variant": variant,
                "next_token": int(np.argmax(logits[:50257])),
                "load_seconds": load_seconds,
                "prefill_seconds": time.perf_counter() - started,
                "fixture": str(output),
            },
            indent=2,
        )
    )


def compare_mlx(model_id: str, fixture: Path, output: Path) -> None:
    import mlx.core as mx
    from mlx_hobbylm.weights import load

    with np.load(fixture) as reference:
        inputs = reference["prompt_ids"]
        expected_logits = reference["logits"]
        expected_routes = reference["routes"]
        variant = str(reference["variant"]) if "variant" in reference else "archive"
    started = time.perf_counter()
    model, cfg = load(model_id)
    load_seconds = time.perf_counter() - started
    mx.reset_peak_memory()
    started = time.perf_counter()
    logits = model(mx.array(inputs[None, :], dtype=mx.int32))[0, -1].astype(mx.float32)
    mx.eval(logits)
    routes = []
    for block in model.blocks:
        if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None:
            mx.eval(block.ffn.last_topi)
            routes.append(np.asarray(block.ffn.last_topi)[-1, -1])
    actual_logits = np.asarray(logits)
    actual_routes = np.stack(routes)
    delta = np.abs(actual_logits - expected_logits)
    expected_token = int(np.argmax(expected_logits[:50257]))
    actual_token = int(np.argmax(actual_logits[:50257]))
    route_equal = np.sort(actual_routes, axis=-1) == np.sort(expected_routes, axis=-1)
    mismatched_layers = []
    for layer_index, (actual, expected) in enumerate(zip(actual_routes, expected_routes)):
        if set(map(int, actual)) != set(map(int, expected)):
            mismatched_layers.append({
                "moe_layer_index": layer_index,
                "pytorch_experts": sorted(map(int, expected)),
                "mlx_experts": sorted(map(int, actual)),
            })
    report = {
        "status": "research_candidate_only",
        "model": model_id,
        "platform": platform.platform(),
        "published_max_position_embeddings": cfg.max_position_embeddings,
        "probe_tokens": int(len(inputs)),
        "variant": variant,
        "expected_token": expected_token,
        "actual_token": actual_token,
        "token_match": expected_token == actual_token,
        "router_agreement": float(route_equal.mean()),
        "expert_set_agreement": 1.0 - len(mismatched_layers) / len(expected_routes),
        "mismatched_layers": mismatched_layers,
        "max_logit_error": float(delta.max()),
        "mean_logit_error": float(delta.mean()),
        "load_seconds": load_seconds,
        "prefill_seconds": time.perf_counter() - started,
        "peak_memory_gib": mx.get_peak_memory() / (1024**3),
        "does_not_certify_retrieval": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["token_match"] or mismatched_layers:
        raise SystemExit("FAIL: long-context token or expert routes differ")
    print("PASS: long-context next token and expert sets match PyTorch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["pytorch", "mlx"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokens", type=int, default=3900)
    parser.add_argument("--variant", choices=["archive", "science", "code"], default="archive")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if not 32 <= args.tokens <= 4096:
        parser.error("--tokens must be between 32 and 4096")
    if args.backend == "pytorch":
        create_reference(args.model, args.tokens, args.fixture, args.variant)
    else:
        compare_mlx(args.model, args.fixture, args.report)


if __name__ == "__main__":
    main()
