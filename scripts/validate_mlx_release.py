"""Validate the public HobbyLM-1B SFT against a saved PyTorch fixture."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_hobbylm.weights import load


DEFAULT_MODEL = "harims95/hobbylm-1b-broad-sft-3450-hf"


def route_snapshot(model) -> np.ndarray:
    rows = []
    for block in model.blocks:
        if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None:
            mx.eval(block.ffn.last_topi)
            rows.append(np.asarray(block.ffn.last_topi)[-1, -1])
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reference", default="artifacts/hobbylm-1b-sft-reference.npz")
    parser.add_argument("--report", default="artifacts/hobbylm-1b-sft-mlx-report.json")
    args = parser.parse_args()

    if platform.machine() != "arm64":
        raise SystemExit("MLX validation requires an Apple-Silicon Mac (arm64)")
    reference = np.load(args.reference)
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, cfg = load(args.model)
    load_seconds = time.perf_counter() - started

    expected = {
        "vocab_size": 50304,
        "d_model": 1024,
        "n_layers": 20,
        "n_experts": 64,
        "top_k": 8,
        "max_position_embeddings": 1024,
    }
    mismatches = {k: (getattr(cfg, k), v) for k, v in expected.items() if getattr(cfg, k) != v}
    if mismatches:
        raise SystemExit(f"Unexpected model configuration: {mismatches}")

    prompt_ids = reference["prompt_ids"].tolist()
    ids = list(prompt_ids)
    cache = model.make_cache()
    current = mx.array([ids], dtype=mx.int32)
    first_logits = None
    first_routes = None
    generation_started = time.perf_counter()
    for step in range(len(reference["generated_ids"])):
        logits = model(current, cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        if step == 0:
            first_logits = np.asarray(logits)
            first_routes = route_snapshot(model)
        token = int(mx.argmax(logits[:50257]).item())
        ids.append(token)
        current = mx.array([[token]], dtype=mx.int32)
        if token == 50256:
            break
    generation_seconds = time.perf_counter() - generation_started
    actual_ids = np.asarray(ids[len(prompt_ids):], dtype=np.int32)

    logits_delta = np.abs(first_logits - reference["first_logits"])
    # torch.topk sorts by score while MLX argpartition does not promise order;
    # routing correctness is equality of the selected expert set per layer.
    route_equal = np.sort(first_routes, axis=-1) == np.sort(reference["first_routes"], axis=-1)
    token_equal = np.array_equal(actual_ids, reference["generated_ids"])
    report = {
        "model": args.model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": getattr(mx, "__version__", "unknown"),
        "kv_cache": True,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generated_tokens": int(len(actual_ids)),
        "tokens_per_second": len(actual_ids) / generation_seconds if generation_seconds else 0.0,
        "peak_memory_gib": mx.get_peak_memory() / (1024**3),
        "max_logit_error": float(logits_delta.max()),
        "mean_logit_error": float(logits_delta.mean()),
        "router_agreement": float(route_equal.mean()),
        "token_sequence_match": bool(token_equal),
        "generated_ids": actual_ids.tolist(),
        "expected_ids": reference["generated_ids"].tolist(),
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not token_equal or report["router_agreement"] != 1.0:
        raise SystemExit("FAIL: MLX tokens or expert routes differ from PyTorch")
    print("PASS: MLX token sequence and expert routing match PyTorch")


if __name__ == "__main__":
    main()
