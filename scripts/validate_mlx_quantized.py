"""Compare a quantized HobbyLM MLX candidate with the FP32 MLX model."""
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


def routes(model) -> np.ndarray:
    result = []
    for block in model.blocks:
        if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None:
            mx.eval(block.ffn.last_topi)
            result.append(np.asarray(block.ffn.last_topi)[-1, -1])
    return np.stack(result)


def run(
    model,
    prompt: list[int],
    tokens: int,
    *,
    forced_tokens: list[int] | None = None,
) -> tuple[list[int], list[np.ndarray], float]:
    generated: list[int] = []
    route_steps: list[np.ndarray] = []
    cache = model.make_cache()
    current = mx.array([prompt], dtype=mx.int32)
    started = time.perf_counter()
    for _ in range(tokens):
        logits = model(current, cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        route_steps.append(routes(model))
        predicted = int(mx.argmax(logits[:50257]).item())
        token = forced_tokens[len(generated)] if forced_tokens is not None else predicted
        generated.append(token)
        if token == 50256:
            break
        current = mx.array([[token]], dtype=mx.int32)
    elapsed = time.perf_counter() - started
    return generated, route_steps, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", default="harims95/hobbylm-1b-broad-sft-3450-hf")
    parser.add_argument("--quantized-model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    prompt_text = (
        "SYSTEM: You are a helpful and concise assistant.\n"
        "USER: Explain sparse routing in one sentence.\nASSISTANT:"
    )
    prompt = tiktoken.get_encoding("gpt2").encode_ordinary(prompt_text)

    reference, _ = load(args.reference_model)
    reference_ids, reference_routes, _ = run(reference, prompt, args.max_tokens)
    del reference
    mx.clear_cache()

    mx.reset_peak_memory()
    started = time.perf_counter()
    candidate, _ = load(args.quantized_model)
    load_seconds = time.perf_counter() - started
    candidate_ids, candidate_routes, elapsed = run(candidate, prompt, args.max_tokens)

    # Route comparisons are meaningful only while both backends receive the
    # same tokens. Force the FP32 sequence through the candidate separately.
    _, forced_candidate_routes, _ = run(
        candidate,
        prompt,
        len(reference_ids),
        forced_tokens=reference_ids,
    )

    compared_steps = min(len(reference_routes), len(candidate_routes))
    agreements = [
        np.mean(
            np.sort(reference_routes[i], axis=-1)
            == np.sort(forced_candidate_routes[i], axis=-1)
        )
        for i in range(compared_steps)
    ]
    first_mismatch = next(
        (index for index, pair in enumerate(zip(reference_ids, candidate_ids)) if pair[0] != pair[1]),
        None,
    )
    report = {
        "status": "candidate_not_release_approved",
        "reference_model": args.reference_model,
        "quantized_model": args.quantized_model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": getattr(mx, "__version__", "unknown"),
        "prompt": prompt_text,
        "reference_ids": reference_ids,
        "quantized_ids": candidate_ids,
        "token_sequence_match": reference_ids == candidate_ids,
        "matching_prefix_tokens": first_mismatch if first_mismatch is not None else len(reference_ids),
        "router_expert_set_agreement_on_identical_tokens": (
            float(np.mean(agreements)) if agreements else None
        ),
        "load_seconds": load_seconds,
        "generation_seconds": elapsed,
        "generated_tokens": len(candidate_ids),
        "tokens_per_second": len(candidate_ids) / elapsed if elapsed else 0.0,
        "peak_memory_gib": mx.get_peak_memory() / (1024**3),
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
