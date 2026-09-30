"""Run the frozen MLX v1 prompt suite and preserve raw generations."""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import tiktoken

from mlx_hobbylm.chat import EOT, GPT2_VALID, generate
from mlx_hobbylm.weights import load


DEFAULT_MODEL = "harims95/hobbylm-1b-broad-sft-3450-hf"
DEFAULT_PROMPTS = Path("eval/mlx_v1_prompts.json")
DEFAULT_OUTPUT = Path("artifacts/hobbylm-1b-mlx-v1-eval.json")
SYSTEM = "You are a helpful and concise assistant."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
    if args.limit is not None:
        prompts = prompts[: args.limit]
    model, cfg = load(args.model)
    tokenizer = tiktoken.get_encoding("gpt2")
    results = []
    started = time.perf_counter()

    for item in prompts:
        serialized = f"SYSTEM: {SYSTEM}\nUSER: {item['prompt']}\nASSISTANT:"
        prompt_ids = tokenizer.encode_ordinary(serialized)
        output_ids, elapsed = generate(
            model,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=0.0,
            repetition_penalty=1.0,
            context=cfg.max_position_embeddings,
        )
        valid_ids = [
            token for token in output_ids if token < GPT2_VALID and token != EOT
        ]
        text = tokenizer.decode(valid_ids).strip()
        counts = Counter(valid_ids)
        repeat_fraction = max(counts.values(), default=0) / max(len(valid_ids), 1)
        severe_repetition = len(valid_ids) >= 8 and repeat_fraction > 0.5
        results.append(
            {
                **item,
                "output": text,
                "output_ids": output_ids,
                "tokens": len(output_ids),
                "seconds": elapsed,
                "tokens_per_second": len(output_ids) / elapsed if elapsed else 0.0,
                "nonempty": bool(text),
                "repeat_fraction": repeat_fraction,
                "severe_repetition": severe_repetition,
            }
        )
        print(f"{item['id']}: {text[:100]}")

    summary = {
        "model": args.model,
        "suite": str(args.prompts),
        "platform": platform.platform(),
        "prompt_count": len(results),
        "nonempty_count": sum(row["nonempty"] for row in results),
        "high_repetition_count": sum(row["severe_repetition"] for row in results),
        "manual_quality_review_required": True,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if summary["nonempty_count"] != summary["prompt_count"]:
        raise SystemExit("FAIL: one or more prompts produced empty output")
    if summary["high_repetition_count"]:
        raise SystemExit("FAIL: one or more prompts triggered severe token repetition")
    print("PASS: frozen MLX v1 execution-health gate")


if __name__ == "__main__":
    main()
