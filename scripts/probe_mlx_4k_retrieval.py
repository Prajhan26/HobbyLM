"""Small research-only probe of shallow versus deep 4K recall."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_hobbylm.weights import load


MODEL = "harims95/hobbylm-1b-broad-sft-3450-hf"
OUTPUT = Path("artifacts/hobbylm-4k-candidate-retrieval.json")
LENGTH = 3900
CODE = "BIRCH-741"


def make_prompt(tokenizer, needle_position: int) -> list[int]:
    prefix = tokenizer.encode_ordinary(
        "SYSTEM: Answer using only facts from the archive.\nUSER: Archive:\n"
    )
    filler = tokenizer.encode_ordinary(
        "Archive entry: The weather is ordinary and the map is in the cabinet.\n"
    )
    needle = tokenizer.encode_ordinary(
        f"Important archive fact: The access code is {CODE}.\n"
    )
    suffix = tokenizer.encode_ordinary(
        "What is the access code? Answer only the code.\nASSISTANT:"
    )
    before = needle_position - len(prefix)
    after = LENGTH - len(prefix) - before - len(needle) - len(suffix)
    if before < 0 or after < 0:
        raise ValueError("Needle position does not fit")
    ids = (
        prefix
        + (filler * (before // len(filler) + 1))[:before]
        + needle
        + (filler * (after // len(filler) + 1))[:after]
        + suffix
    )
    assert len(ids) == LENGTH
    return ids


def main() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    model, cfg = load(MODEL)
    rows = []
    for position in (256, 1920, 3584):
        ids = make_prompt(tokenizer, position)
        cache = model.make_cache()
        current = mx.array([ids], dtype=mx.int32)
        output = []
        started = time.perf_counter()
        for _ in range(24):
            logits = model(current, cache=cache)[0, -1].astype(mx.float32)
            mx.eval(logits)
            token = int(mx.argmax(logits[:50257]).item())
            if token == 50256:
                break
            output.append(token)
            current = mx.array([[token]], dtype=mx.int32)
        answer = tokenizer.decode(output).strip()
        row = {
            "needle_token_position": position,
            "prompt_tokens": len(ids),
            "expected_code": CODE,
            "answer": answer,
            "exact_code_present": CODE in answer,
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(json.dumps(row))
    report = {
        "status": "research_candidate_only",
        "model": MODEL,
        "published_max_position_embeddings": cfg.max_position_embeddings,
        "does_not_certify_4k": True,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
