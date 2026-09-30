"""Create a deterministic PyTorch fixture for the public HobbyLM-1B SFT.

Run this once before the MLX validator. PyTorch and MLX are intentionally run
in separate processes so a 16/32 GB Mac never holds both models at once.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
from transformers import AutoModelForCausalLM


DEFAULT_MODEL = "harims95/hobbylm-1b-broad-sft-3450-hf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="artifacts/hobbylm-1b-sft-reference.npz")
    parser.add_argument("--prompt", default="Explain sparse routing in one sentence.")
    parser.add_argument("--system", default="You are a helpful and concise assistant.")
    parser.add_argument("--generate-tokens", type=int, default=8)
    args = parser.parse_args()

    serialized = f"SYSTEM: {args.system}\nUSER: {args.prompt}\nASSISTANT:"
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt_ids = tokenizer.encode_ordinary(serialized)

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    load_seconds = time.perf_counter() - started

    ids = list(prompt_ids)
    first_logits = None
    first_routes = None
    generation_started = time.perf_counter()
    with torch.no_grad():
        for step in range(args.generate_tokens):
            inputs = torch.tensor([ids], dtype=torch.long)
            output = model(input_ids=inputs, use_cache=False)
            logits = output.logits[0, -1].float().cpu()
            if step == 0:
                first_logits = logits.numpy()
                first_routes = np.stack([
                    layer.mlp.gate.last_topi[-1].cpu().numpy()
                    for layer in model.model.layers
                    if getattr(layer, "is_moe", False)
                ])
            token = int(logits[:50257].argmax())
            ids.append(token)
            if token == 50256:
                break
    generation_seconds = time.perf_counter() - generation_started

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
        first_logits=first_logits,
        first_routes=first_routes,
        generated_ids=np.asarray(ids[len(prompt_ids):], dtype=np.int32),
    )
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "system": args.system,
        "serialized_prompt": serialized,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generated_text": tokenizer.decode([x for x in ids[len(prompt_ids):] if x < 50257]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Reference fixture written to {output}")


if __name__ == "__main__":
    main()
