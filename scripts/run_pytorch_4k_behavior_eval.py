"""PyTorch CPU baseline for the frozen 4K MLX behavior prompts.

Uses the identical pinned HF weights, tokenizer, serialized prompt bytes,
greedy token selection, and EOS rule as the MLX runner. Research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import tiktoken
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM


REPO = "harims95/hobbylm-1b-broad-sft-3450-hf"
REVISION = "ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5"
WEIGHT_SHA256 = "d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89"
PROMPTS = Path("eval/mlx_4k_behavior_prompts.json")
OUTPUT = Path("artifacts/mlx-4k-behavior-eval/pytorch-raw.json")
EOS = 50256
VALID_VOCAB = 50257


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_case(case: dict, tokenizer, system: str) -> list[int]:
    if case["format"] == "tools":
        tools_json = json.dumps(case["tools"], separators=(",", ":"))
        prompt = f"TOOLS: {tools_json}\nUSER: {case['prompt']}\nASSISTANT:"
    else:
        prompt = f"SYSTEM: {system.strip()}\nUSER: {case['prompt'].strip()}\nASSISTANT:"
    return tokenizer.encode_ordinary(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, help="Run first N cases for a smoke test")
    args = parser.parse_args()
    suite = json.loads(args.prompts.read_text())
    if suite.get("source_revision") != REVISION:
        raise SystemExit("Frozen prompt revision differs from pinned model revision")
    cases = suite["cases"][: args.limit] if args.limit else suite["cases"]
    if not cases:
        raise SystemExit("No cases selected")
    snapshot = Path(snapshot_download(
        REPO, revision=REVISION,
        allow_patterns=["config.json", "generation_config.json", "model.safetensors", "*.py"],
    ))
    weight_sha256 = file_sha256(snapshot / "model.safetensors")
    if weight_sha256 != WEIGHT_SHA256:
        raise SystemExit(f"Unexpected pinned weight SHA-256: {weight_sha256}")
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    load_seconds = time.perf_counter() - started
    if model.config.max_position_embeddings != 4096:
        raise SystemExit("Pinned model config is not 4096")
    tokenizer = tiktoken.get_encoding("gpt2")
    rows = []
    suite_started = time.perf_counter()
    for case in cases:
        prompt_ids = serialize_case(case, tokenizer, suite.get("default_system", ""))
        max_tokens = case.get("max_tokens", 48)
        output_ids = []
        current = torch.tensor([prompt_ids], dtype=torch.long)
        past = None
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(max_tokens):
                result = model(input_ids=current, past_key_values=past, use_cache=True)
                token = int(result.logits[0, -1, :VALID_VOCAB].argmax().item())
                output_ids.append(token)
                if token == EOS:
                    break
                past = result.past_key_values
                current = torch.tensor([[token]], dtype=torch.long)
        seconds = time.perf_counter() - started
        row = {
            "id": case["id"],
            "category": case["category"],
            "format": case["format"],
            "raw_prompt": case["prompt"],
            "tools": case.get("tools"),
            "prompt_tokens": len(prompt_ids),
            "prompt_sha256": hashlib.sha256(
                b",".join(str(token).encode() for token in prompt_ids)
            ).hexdigest(),
            "output_text": tokenizer.decode(
                [token for token in output_ids if token < VALID_VOCAB and token != EOS]
            ).strip(),
            "output_ids": output_ids,
            "generated_tokens": len(output_ids),
            "seconds": seconds,
            "tokens_per_second": len(output_ids) / seconds if seconds else 0.0,
        }
        rows.append(row)
        print(f"{case['id']} [{case['category']}]: {row['output_text'][:100]!r}", flush=True)
    report = {
        "status": "research_only_not_validated",
        "repo_id": REPO,
        "revision": REVISION,
        "weight_sha256": weight_sha256,
        "model_dir": str(snapshot),
        "config": {"max_position_embeddings": model.config.max_position_embeddings},
        "backend": "pytorch_cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "load_seconds": load_seconds,
        "suite_seconds": time.perf_counter() - suite_started,
        "case_count": len(rows),
        "prompts_file": str(args.prompts),
        "prompts_sha256": file_sha256(args.prompts),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(rows)} PyTorch baseline cases to {args.output}")


if __name__ == "__main__":
    main()
