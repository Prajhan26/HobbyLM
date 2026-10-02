"""Replay frozen 3900-token retrieval prompts in PyTorch without changing the suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import tiktoken
import torch
from transformers import AutoModelForCausalLM


FILLER = "Archive entry: The weather is ordinary and the map is in the cabinet.\n"
VALID_VOCAB = 50257
EOS = 50256


def prompt(tokenizer, length: int, fraction: float, code: str) -> list[int]:
    prefix = tokenizer.encode_ordinary("SYSTEM: Answer using only facts from the archive.\nUSER: Archive:\n")
    filler = tokenizer.encode_ordinary(FILLER)
    needle = tokenizer.encode_ordinary(f"Important archive fact: The access code is {code}.\n")
    suffix = tokenizer.encode_ordinary("What is the access code? Answer only the code.\nASSISTANT:")
    position = round(length * fraction)
    before = position - len(prefix)
    after = length - position - len(needle) - len(suffix)
    if before < 0 or after < 0:
        raise ValueError("Needle does not fit")
    ids = prefix + (filler * (before // len(filler) + 1))[:before]
    ids += needle + (filler * (after // len(filler) + 1))[:after] + suffix
    if len(ids) != length:
        raise AssertionError("Prompt length changed")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--m1-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=3900)
    args = parser.parse_args()
    suite = json.loads(Path("eval/mlx_4k_research_suite.json").read_text())
    tokenizer = tiktoken.get_encoding("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    if model.config.max_position_embeddings != 4096:
        raise SystemExit("Expected a 4096-configured candidate")
    rows = []
    for fraction in suite["retrieval_positions"]:
        for code in suite["retrieval_codes"]:
            path = args.m1_reports / f"mlx-4k-research-retrieval-{args.length}-{fraction}-{code}.json"
            m1 = json.loads(path.read_text())["rows"][0]
            ids = prompt(tokenizer, args.length, fraction, code)
            digest = hashlib.sha256(b"".join(int(t).to_bytes(4, "little") for t in ids)).hexdigest()
            if digest != m1["prompt_sha256"]:
                raise SystemExit(f"Prompt differs from M1 report: {path}")
            generated = []
            past = None
            current = torch.tensor([ids], dtype=torch.long)
            started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(suite["retrieval_max_new_tokens"]):
                    result = model(input_ids=current, past_key_values=past, use_cache=True)
                    token = int(result.logits[0, -1, :VALID_VOCAB].argmax().item())
                    if token == EOS:
                        break
                    generated.append(token)
                    past = result.past_key_values
                    current = torch.tensor([[token]], dtype=torch.long)
            answer = tokenizer.decode(generated).strip()
            row = {
                "prompt_tokens": args.length,
                "needle_fraction": fraction,
                "expected_code": code,
                "prompt_sha256": digest,
                "pytorch_answer": answer,
                "pytorch_exact_match": answer.rstrip(".") == code,
                "mlx_m1_answer": m1["answer"],
                "mlx_m1_exact_match": m1["exact_match"],
                "pytorch_output_ids": generated,
                "pytorch_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "status": "research_only_not_release_validation",
        "model": str(args.model),
        "reference": "pytorch_cpu",
        "m1_report_dir": str(args.m1_reports),
        "cases": len(rows),
        "pytorch_exact": sum(row["pytorch_exact_match"] for row in rows),
        "mlx_m1_exact": sum(row["mlx_m1_exact_match"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
