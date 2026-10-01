"""Frozen, research-only final-position PyTorch/MLX parity grid.

Run once with ``pytorch`` in the reference venv, then ``mlx`` in the MLX venv.
Failures remain failures; no tie tolerance is applied to expert-set parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mlx_4k_candidate import prompt_ids


DEFAULT_MODEL = Path("artifacts/hobbylm-1b-sft-3450-4k-candidate")
DEFAULT_SUITE = Path("eval/mlx_4k_research_suite.json")
DEFAULT_OUTPUT = Path("artifacts/mlx-4k-research-parity")


def cases(suite: dict) -> list[tuple[str, int, str]]:
    return [
        (f"{variant}-{length}", length, variant)
        for variant in suite["parity_variants"]
        for length in suite["parity_lengths"]
    ]


def check_candidate(model_dir: Path, suite: dict) -> None:
    manifest = json.loads((model_dir / "candidate-manifest.json").read_text())
    config = json.loads((model_dir / "config.json").read_text())
    if manifest["source_revision"] != suite["source_revision"]:
        raise ValueError("Candidate revision differs from frozen suite")
    if config["max_position_embeddings"] != suite["candidate_context"]:
        raise ValueError("Candidate context differs from frozen suite")
    if config.get("torch_dtype") != "float32":
        raise ValueError("Candidate must remain FP32")


def pytorch_run(model_dir: Path, output: Path, selected: list) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    rows = []
    for case_id, length, variant in selected:
        ids = prompt_ids(length, variant)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(
                input_ids=torch.from_numpy(ids.astype(np.int64)).unsqueeze(0),
                use_cache=False,
            ).logits[0, -1].float().cpu().numpy()
            routes = np.stack([
                layer.mlp.gate.last_topi[-1].cpu().numpy()
                for layer in model.model.layers if getattr(layer, "is_moe", False)
            ])
        fixture = output / f"{case_id}.npz"
        np.savez_compressed(fixture, prompt_ids=ids, logits=logits, routes=routes)
        row = {
            "case": case_id, "tokens": length, "variant": variant,
            "prompt_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "next_token": int(np.argmax(logits[:50257])),
            "seconds": time.perf_counter() - started,
            "fixture": str(fixture),
        }
        rows.append(row)
        (output / f"{case_id}.pytorch.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    return rows


def mlx_run(model_dir: Path, output: Path, selected: list) -> list[dict]:
    import mlx.core as mx
    from mlx_hobbylm.weights import load_local

    model, _ = load_local(model_dir)
    rows = []
    for case_id, length, variant in selected:
        with np.load(output / f"{case_id}.npz") as reference:
            ids = reference["prompt_ids"]
            expected_logits = reference["logits"]
            expected_routes = reference["routes"]
        if not np.array_equal(ids, prompt_ids(length, variant)):
            raise ValueError(f"Fixture prompt changed: {case_id}")
        mx.reset_peak_memory()
        started = time.perf_counter()
        logits = model(mx.array(ids[None, :], dtype=mx.int32))[0, -1].astype(mx.float32)
        mx.eval(logits)
        actual_logits = np.asarray(logits)
        mx.eval(*[
            block.ffn.last_topi for block in model.blocks
            if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None
        ])
        actual_routes = np.stack([
            np.asarray(block.ffn.last_topi)[-1, -1]
            for block in model.blocks if hasattr(block.ffn, "last_topi")
            and block.ffn.last_topi is not None
        ])
        mismatch = [
            index for index, (actual, expected) in enumerate(zip(actual_routes, expected_routes))
            if set(map(int, actual)) != set(map(int, expected))
        ]
        delta = np.abs(actual_logits - expected_logits)
        expected_token = int(np.argmax(expected_logits[:50257]))
        actual_token = int(np.argmax(actual_logits[:50257]))
        row = {
            "case": case_id, "tokens": length, "variant": variant,
            "prompt_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "expected_token": expected_token, "actual_token": actual_token,
            "token_match": expected_token == actual_token,
            "expert_sets_matching": len(expected_routes) - len(mismatch),
            "expert_sets_total": len(expected_routes),
            "mismatched_moe_indices": mismatch,
            "strict_parity_pass": expected_token == actual_token and not mismatch,
            "mean_logit_error": float(delta.mean()),
            "max_logit_error": float(delta.max()),
            "peak_memory_gib": mx.get_peak_memory() / (1024**3),
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        (output / f"{case_id}.mlx.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["pytorch", "mlx"])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Research smoke test; omit for full grid")
    parser.add_argument("--case", help="Run one frozen case in an isolated process")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    check_candidate(args.model, suite)
    selected = cases(suite)
    if args.case:
        selected = [case for case in selected if case[0] == args.case]
        if not selected:
            parser.error(f"Unknown frozen case: {args.case}")
        if args.limit is not None:
            parser.error("Use --case or --limit, not both")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        selected = selected[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = (pytorch_run if args.backend == "pytorch" else mlx_run)(args.model, args.output, selected)
    summary = {
        "status": "research_only_not_release_validation",
        "backend": args.backend,
        "model": str(args.model),
        "platform": platform.platform(),
        "suite": str(args.suite),
        "cases_completed": len(rows),
        "cases_in_full_suite": len(cases(suite)),
        "all_strict_parity_pass": all(row["strict_parity_pass"] for row in rows) if args.backend == "mlx" else None,
        "rows": rows,
    }
    summary_name = f"{args.backend}-{args.case}-summary.json" if args.case else f"{args.backend}-summary.json"
    (args.output / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if args.backend == "mlx" and not summary["all_strict_parity_pass"]:
        raise SystemExit("FAIL: at least one strict token/expert-set parity case failed")


if __name__ == "__main__":
    main()
