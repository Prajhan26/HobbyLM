"""Frozen research-only needle retrieval grid for the local 4K candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_hobbylm.weights import load_local


DEFAULT_MODEL = Path("artifacts/hobbylm-1b-sft-3450-4k-candidate")
DEFAULT_SUITE = Path("eval/mlx_4k_research_suite.json")
DEFAULT_OUTPUT = Path("artifacts/mlx-4k-research-retrieval.json")
FILLER = "Archive entry: The weather is ordinary and the map is in the cabinet.\n"


def prompt(tokenizer, length: int, fraction: float, code: str) -> tuple[list[int], int]:
    prefix = tokenizer.encode_ordinary(
        "SYSTEM: Answer using only facts from the archive.\nUSER: Archive:\n"
    )
    filler = tokenizer.encode_ordinary(FILLER)
    needle = tokenizer.encode_ordinary(f"Important archive fact: The access code is {code}.\n")
    suffix = tokenizer.encode_ordinary(
        "What is the access code? Answer only the code.\nASSISTANT:"
    )
    position = round(length * fraction)
    before = position - len(prefix)
    after = length - position - len(needle) - len(suffix)
    if before < 0 or after < 0:
        raise ValueError(f"Needle does not fit in {length} tokens at {fraction}")
    ids = (
        prefix + (filler * (before // len(filler) + 1))[:before]
        + needle + (filler * (after // len(filler) + 1))[:after] + suffix
    )
    if len(ids) != length:
        raise AssertionError("Retrieval prompt length changed")
    return ids, position


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Research smoke test; omit for full grid")
    parser.add_argument("--length", type=int, help="Select one frozen prompt length")
    parser.add_argument("--position", type=float, help="Select one frozen needle fraction")
    parser.add_argument("--code", help="Select one frozen access code")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    manifest = json.loads((args.model / "candidate-manifest.json").read_text())
    if manifest["source_revision"] != suite["source_revision"]:
        raise ValueError("Candidate revision differs from frozen suite")
    model, cfg = load_local(args.model)
    if cfg.max_position_embeddings != suite["candidate_context"]:
        raise ValueError("Candidate context differs from frozen suite")
    tokenizer = tiktoken.get_encoding("gpt2")
    grid = [
        (length, fraction, code)
        for length in suite["retrieval_lengths"]
        for fraction in suite["retrieval_positions"]
        for code in suite["retrieval_codes"]
    ]
    if args.length is not None:
        grid = [case for case in grid if case[0] == args.length]
    if args.position is not None:
        grid = [case for case in grid if case[1] == args.position]
    if args.code is not None:
        grid = [case for case in grid if case[2] == args.code]
    if not grid:
        parser.error("No frozen cases match the selected filters")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        grid = grid[:args.limit]
    rows = []
    for length, fraction, code in grid:
        ids, position = prompt(tokenizer, length, fraction, code)
        mx.reset_peak_memory()
        started = time.perf_counter()
        cache = model.make_cache()
        current = mx.array([ids], dtype=mx.int32)
        output_ids = []
        for _ in range(suite["retrieval_max_new_tokens"]):
            logits = model(current, cache=cache)[0, -1].astype(mx.float32)
            mx.eval(logits)
            token = int(mx.argmax(logits[:50257]).item())
            if token == 50256:
                break
            output_ids.append(token)
            current = mx.array([[token]], dtype=mx.int32)
        answer = tokenizer.decode(output_ids).strip()
        row = {
            "prompt_tokens": length,
            "needle_fraction": fraction,
            "needle_token_position": position,
            "expected_code": code,
            "answer": answer,
            "exact_match": answer.strip().rstrip(".") == code,
            "code_present": code in answer,
            "generated_tokens": len(output_ids),
            "seconds": time.perf_counter() - started,
            "peak_memory_gib": mx.get_peak_memory() / (1024**3),
            "prompt_sha256": hashlib.sha256(bytes().join(
                int(token).to_bytes(4, "little") for token in ids
            )).hexdigest(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "status": "research_only_not_release_validation",
        "model": str(args.model),
        "platform": platform.platform(),
        "suite": str(args.suite),
        "cases_completed": len(rows),
        "cases_in_full_suite": (
            len(suite["retrieval_lengths"]) * len(suite["retrieval_positions"])
            * len(suite["retrieval_codes"])
        ),
        "exact_match_count": sum(row["exact_match"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.length is not None or args.position is not None or args.code is not None:
        suffix = f"-{args.length or 'all'}-{args.position or 'all'}-{args.code or 'all'}"
        args.output = args.output.with_name(args.output.stem + suffix + args.output.suffix)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
