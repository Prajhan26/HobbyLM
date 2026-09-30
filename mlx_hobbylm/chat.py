"""A minimal, private terminal chat for HobbyLM on Apple Silicon."""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
import tiktoken

from . import __version__
from .weights import load


EOT = 50256
GPT2_VALID = 50257


def _next_token(logits: mx.array, previous: list[int], temperature: float, repetition_penalty: float) -> int:
    values = np.asarray(logits.astype(mx.float32))
    values[GPT2_VALID:] = -np.inf
    if repetition_penalty != 1.0:
        for token in set(previous):
            values[token] = values[token] / repetition_penalty if values[token] > 0 else values[token] * repetition_penalty
    if temperature <= 0:
        return int(values.argmax())
    scaled = mx.array(values / temperature)
    return int(mx.random.categorical(scaled).item())


def generate(model, prompt_ids: list[int], *, max_tokens: int, temperature: float,
             repetition_penalty: float, context: int) -> tuple[list[int], float]:
    ids = list(prompt_ids)
    cache = model.make_cache()
    current = mx.array([ids[-context:]], dtype=mx.int32)
    started = time.perf_counter()
    for _ in range(max_tokens):
        logits = model(current, cache=cache)[0, -1]
        mx.eval(logits)
        token = _next_token(logits, ids, temperature, repetition_penalty)
        ids.append(token)
        if token == EOT:
            break
        cached_length = cache[0][0].shape[-2]
        if cached_length >= context:
            cache = model.make_cache()
            current = mx.array([ids[-context:]], dtype=mx.int32)
        else:
            current = mx.array([[token]], dtype=mx.int32)
    return ids[len(prompt_ids) :], time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HobbyLM locally with Apple MLX")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--model",
        default="harims95/hobbylm-1b-broad-sft-3450-hf",
        help="HF repo or local model directory",
    )
    parser.add_argument("--prompt", help="Single prompt; omit for an interactive session")
    parser.add_argument("--system", default="", help="Optional system instruction")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    args = parser.parse_args()

    print(f"Loading {args.model} on Apple Silicon…", flush=True)
    model, cfg = load(args.model)
    tokenizer = tiktoken.get_encoding("gpt2")
    context = min(args.context or cfg.max_position_embeddings, cfg.max_position_embeddings)
    print(f"Ready — {cfg.n_layers} layers, {cfg.n_experts} experts, top-{cfg.top_k}. Processing stays on this Mac.\n")

    def answer(question: str) -> None:
        system = f"SYSTEM: {args.system.strip()}\n" if args.system.strip() else ""
        prompt = f"{system}USER: {question.strip()}\nASSISTANT:"
        prompt_ids = tokenizer.encode_ordinary(prompt)
        output_ids, elapsed = generate(
            model,
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            context=context,
        )
        output = tokenizer.decode(
            [token for token in output_ids if token < GPT2_VALID and token != EOT]
        )
        rate = len(output_ids) / elapsed if elapsed else 0.0
        print(f"\nHobbyLM: {output.strip()}\n\n[{len(output_ids)} tokens · {rate:.1f} tok/s]\n")

    if args.prompt:
        answer(args.prompt)
        return
    print("Type /quit to exit.\n")
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.strip().lower() in {"/quit", "/exit"}:
            return
        if question.strip():
            answer(question)


if __name__ == "__main__":
    main()
