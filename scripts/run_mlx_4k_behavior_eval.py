"""Run the frozen 30-case 4K MLX behavior suite on a pinned HF revision.

Research only: this does not validate a 4K MLX release (see
docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md). It loads the immutable, 4096-configured
HF revision, generates greedy completions for every case in
eval/mlx_4k_behavior_prompts.json, and writes raw prompts, prompt token
hashes, outputs, the exact model revision, backend/version info, load time,
peak memory, and generation tokens/second. It does not score or apply
thresholds; run scripts/score_mlx_4k_behavior_eval.py on its output for that.

Does not touch the shipping 1K MLX default, mlx_hobbylm runtime code, or the
Hugging Face Hub (read-only download of an already-published, immutable
revision).
"""
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
from mlx_hobbylm import __version__ as mlx_hobbylm_version
from mlx_hobbylm.chat import EOT, GPT2_VALID, generate, serialize_prompt
from mlx_hobbylm.weights import load_local

DEFAULT_REPO_ID = "harims95/hobbylm-1b-broad-sft-3450-hf"
DEFAULT_REVISION = "ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5"
EXPECTED_WEIGHT_SHA256 = "d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89"
DEFAULT_PROMPTS = Path("eval/mlx_4k_behavior_prompts.json")
DEFAULT_OUTPUT = Path("artifacts/mlx-4k-behavior-eval/raw.json")
DEFAULT_MAX_TOKENS = 48
EXPECTED_CONTEXT = 4096


def download_pinned(repo_id: str, revision: str, cache_dir: str | None) -> Path:
    """Fetch an immutable revision snapshot without touching mlx_hobbylm.weights.download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["config.json", "generation_config.json", "model.safetensors"],
    )
    return Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_case(case: dict, tokenizer, system: str, context: int) -> list[int]:
    if case["format"] == "tools":
        tools_json = json.dumps(case["tools"], separators=(",", ":"))
        serialized = f"TOOLS: {tools_json}\nUSER: {case['prompt']}\nASSISTANT:"
        return tokenizer.encode_ordinary(serialized)
    return serialize_prompt(
        case["prompt"], system=system, history=[], tokenizer=tokenizer, context=context
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                         help="Fallback budget for cases without a per-case max_tokens")
    parser.add_argument("--limit", type=int, help="Run only the first N cases (smoke test)")
    parser.add_argument("--case-id", action="append", dest="case_ids",
                         help="Run only this case id; repeatable")
    args = parser.parse_args()

    if platform.machine() != "arm64":
        raise SystemExit("This runner targets Apple Silicon (arm64) MLX; got " + platform.machine())

    suite = json.loads(args.prompts.read_text())
    suite_sha256 = sha256_file(args.prompts)
    if suite.get("source_revision") != args.revision:
        raise SystemExit(
            f"Suite source_revision {suite.get('source_revision')!r} does not match "
            f"--revision {args.revision!r}; the frozen suite and pinned run must agree."
        )
    cases = suite["cases"]
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected")

    print(f"Downloading pinned revision {args.revision} of {args.repo_id} (read-only)...", flush=True)
    model_dir = download_pinned(args.repo_id, args.revision, args.cache_dir)
    weight_sha256 = sha256_file(model_dir / "model.safetensors")
    if weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise SystemExit(f"Unexpected pinned weight SHA-256: {weight_sha256}")

    mx.reset_peak_memory()
    started = time.perf_counter()
    model, cfg = load_local(model_dir)  # strict weight loading; raises on missing/unexpected weights
    load_seconds = time.perf_counter() - started
    peak_memory_after_load_gib = mx.get_peak_memory() / (1024**3)

    if cfg.max_position_embeddings != EXPECTED_CONTEXT:
        raise SystemExit(
            f"Expected max_position_embeddings={EXPECTED_CONTEXT} at revision {args.revision}, "
            f"got {cfg.max_position_embeddings}. Refusing to run a research suite built for 4K "
            "against an unexpected config."
        )

    tokenizer = tiktoken.get_encoding("gpt2")
    system = suite.get("default_system", "")
    results = []
    suite_started = time.perf_counter()

    for case in cases:
        prompt_ids = serialize_case(case, tokenizer, system, cfg.max_position_embeddings)
        max_tokens = case.get("max_tokens", args.max_tokens)
        output_ids, elapsed = generate(
            model,
            prompt_ids,
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=1.0,
            context=cfg.max_position_embeddings,
        )
        valid_ids = [t for t in output_ids if t < GPT2_VALID and t != EOT]
        text = tokenizer.decode(valid_ids).strip()
        row = {
            "id": case["id"],
            "category": case["category"],
            "format": case["format"],
            "raw_prompt": case["prompt"],
            "tools": case.get("tools"),
            "prompt_tokens": len(prompt_ids),
            "prompt_sha256": hashlib.sha256(
                b",".join(str(t).encode() for t in prompt_ids)
            ).hexdigest(),
            "output_text": text,
            "output_ids": output_ids,
            "generated_tokens": len(output_ids),
            "seconds": elapsed,
            "tokens_per_second": len(output_ids) / elapsed if elapsed else 0.0,
        }
        results.append(row)
        print(f"{case['id']} [{case['category']}]: {text[:100]!r}", flush=True)

    suite_seconds = time.perf_counter() - suite_started
    report = {
        "status": "research_only_not_validated",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "weight_sha256": weight_sha256,
        "model_dir": str(model_dir),
        "config": {
            "max_position_embeddings": cfg.max_position_embeddings,
            "n_layers": cfg.n_layers,
            "n_experts": cfg.n_experts,
            "top_k": cfg.top_k,
        },
        "backend": "mlx",
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_hobbylm_version": mlx_hobbylm_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "load_seconds": load_seconds,
        "peak_memory_after_load_gib": peak_memory_after_load_gib,
        "peak_memory_overall_gib": mx.get_peak_memory() / (1024**3),
        "suite_seconds": suite_seconds,
        "case_count": len(results),
        "prompts_file": str(args.prompts),
        "prompts_sha256": suite_sha256,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print(f"Wrote {len(results)} raw case results to {args.output}")


if __name__ == "__main__":
    main()
