"""Score a raw run from run_mlx_4k_behavior_eval.py against frozen rubrics.

Research only: this reports gate numbers against the PROPOSED, UNAPPROVED
thresholds in docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md. It never claims a
"validated 4K MLX release" and never adjusts a threshold based on what a run
produced. The threshold constants below are copied once from that draft and
must not be edited to make a specific run pass; if Hari/the user approve
different numbers, that is a deliberate, separately reviewed edit to this
file BEFORE the next run, not a response to a run's output.

Usage:
    python scripts/score_mlx_4k_behavior_eval.py \
        --raw artifacts/mlx-4k-behavior-eval/raw.json \
        --prompts eval/mlx_4k_behavior_prompts.json \
        --report artifacts/mlx-4k-behavior-eval/report.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Proposed, unapproved gates from docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md.
# Frozen here; do not change after seeing a run's results.
GATES = {
    "calling": {"min_pass": 8, "total": 10},
    "instruction_following": {"min_pass": 8, "total": 10},
    "repetition_continuation": {"min_pass": 9, "total": 10, "no_runaway_required": True},
}


def _words(text: str) -> list[str]:
    return text.split()


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _parse_tool_calls(text: str) -> tuple[list[dict], str]:
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        try:
            node = ast.parse(text.strip(), mode="eval").body
        except SyntaxError as exc:
            raise ValueError("output is neither JSON nor a Python-like call list") from exc
        if not isinstance(node, ast.List) or not node.elts:
            raise ValueError("Python-like output must be a non-empty call list")
        parsed = []
        for item in node.elts:
            if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
                raise ValueError("call list must contain named calls with keyword arguments only")
            arguments = {}
            for keyword in item.keywords:
                if keyword.arg is None:
                    raise ValueError("expanded keyword arguments are not allowed")
                try:
                    arguments[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
                    raise ValueError("tool arguments must be literal values") from exc
            parsed.append({"name": item.func.id, "arguments": arguments})
        return parsed, "pythonic"
    if not isinstance(calls, list) or not calls:
        raise ValueError("JSON output must be a non-empty call array")
    return calls, "json"


def score_tool_call(text: str, rubric: dict) -> tuple[bool, str]:
    try:
        calls, fmt = _parse_tool_calls(text)
    except ValueError as exc:
        return False, str(exc)
    if len(calls) != 1:
        return False, f"expected exactly one call, got {len(calls)}"
    call = calls[0]
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return False, "call is missing a valid name"
    name = call["name"]
    if name != rubric["expected_tool"]:
        return False, f"expected {rubric['expected_tool']!r}, got {name!r}"
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return False, "arguments must be an object"
    for key, expected in rubric.get("required_args", {}).items():
        if key not in args or args[key] in (None, ""):
            return False, f"missing required argument {key!r}"
        if expected is not None and str(expected).lower() not in str(args[key]).lower():
            return False, f"argument {key!r} does not match expected value"
    return True, f"ok ({fmt})"


def score_word_count(text: str, rubric: dict) -> tuple[bool, str]:
    n = len(_words(text))
    ok = rubric["min_words"] <= n <= rubric["max_words"]
    return ok, f"word_count={n}"


def score_line_count(text: str, rubric: dict) -> tuple[bool, str]:
    n = len(_lines(text))
    ok = rubric["min_lines"] <= n <= rubric["max_lines"]
    return ok, f"line_count={n}"


def score_sentence_count(text: str, rubric: dict) -> tuple[bool, str]:
    n = len(_sentences(text))
    ok = rubric["min_sentences"] <= n <= rubric["max_sentences"]
    return ok, f"sentence_count={n}"


def score_json_only(text: str, rubric: dict) -> tuple[bool, str]:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False, "output is not valid JSON"
    return True, "ok"


def score_all_uppercase(text: str, rubric: dict) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    ok = bool(letters) and all(c.isupper() for c in letters)
    return ok, "ok" if ok else "contains lowercase letters or has no letters"


def score_starts_with(text: str, rubric: dict) -> tuple[bool, str]:
    ok = text.strip().lower().startswith(rubric["prefix"].lower())
    return ok, "ok" if ok else f"does not start with {rubric['prefix']!r}"


def score_forbidden_words(text: str, rubric: dict) -> tuple[bool, str]:
    lowered = text.lower()
    hit = [w for w in rubric["forbidden"] if re.search(rf"\b{re.escape(w.lower())}\b", lowered)]
    return (not hit), ("ok" if not hit else f"contains forbidden word(s): {hit}")


def score_contains_all(text: str, rubric: dict) -> tuple[bool, str]:
    lowered = text.lower()
    missing = [s for s in rubric["required"] if s.lower() not in lowered]
    return (not missing), ("ok" if not missing else f"missing required substring(s): {missing}")


def max_consecutive_ngram_repeats(token_ids: list[int], n: int = 3) -> int:
    if len(token_ids) < n * 2:
        return 1
    best = 1
    i = 0
    while i < len(token_ids) - n:
        gram = tuple(token_ids[i : i + n])
        run = 1
        j = i + n
        while tuple(token_ids[j : j + n]) == gram and j + n <= len(token_ids):
            run += 1
            j += n
        best = max(best, run)
        i += 1
    return best


def score_repetition(output_ids: list[int], rubric: dict) -> tuple[bool, str]:
    tokens = [token for token in output_ids if token != 50256]
    if len(tokens) < rubric.get("min_tokens_checked", 8):
        return False, f"too few non-EOS tokens to evaluate ({len(tokens)})"
    counts = Counter(tokens)
    repeat_fraction = max(counts.values()) / len(tokens)
    runaway = max_consecutive_ngram_repeats(tokens)
    ok = (
        repeat_fraction <= rubric["max_repeat_fraction"]
        and runaway <= rubric["max_consecutive_ngram_repeats"]
    )
    return ok, f"repeat_fraction={repeat_fraction:.2f}, max_consecutive_ngram_repeats={runaway}"


SCORERS = {
    "tool_call": score_tool_call,
    "word_count": score_word_count,
    "line_count": score_line_count,
    "sentence_count": score_sentence_count,
    "json_only": score_json_only,
    "all_uppercase": score_all_uppercase,
    "starts_with": score_starts_with,
    "forbidden_words": score_forbidden_words,
    "contains_all": score_contains_all,
}


def score_case(result: dict, rubric: dict) -> tuple[bool, str]:
    rubric_type = rubric["type"]
    if rubric_type == "repetition":
        return score_repetition(result["output_ids"], rubric)
    return SCORERS[rubric_type](result["output_text"], rubric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("eval/mlx_4k_behavior_prompts.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/mlx-4k-behavior-eval/report.json"))
    args = parser.parse_args()

    raw = json.loads(args.raw.read_text())
    prompts_sha256 = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    if raw.get("prompts_sha256") != prompts_sha256:
        raise SystemExit("Raw run and prompt fixture hashes differ; refusing to rescore")
    suite = json.loads(args.prompts.read_text())
    rubrics = {c["id"]: c["rubric"] for c in suite["cases"]}
    if len(rubrics) != len(suite["cases"]):
        raise SystemExit("Duplicate case IDs in prompt fixture")

    per_case = []
    by_category = defaultdict(list)
    for result in raw["results"]:
        if result["id"] not in rubrics:
            raise SystemExit(f"Unknown case ID in raw run: {result['id']}")
        rubric = rubrics[result["id"]]
        passed, detail = score_case(result, rubric)
        row = {
            "id": result["id"],
            "category": result["category"],
            "rubric_type": rubric["type"],
            "pass": passed,
            "detail": detail,
            "output_text": result["output_text"],
        }
        if rubric["type"] == "repetition":
            tokens = [token for token in result["output_ids"] if token != 50256]
            row["runaway_ngram"] = (
                max_consecutive_ngram_repeats(tokens)
                > rubric["max_consecutive_ngram_repeats"]
            )
        per_case.append(row)
        by_category[result["category"]].append(row)

    gates = {}
    for category, gate in GATES.items():
        rows = by_category.get(category, [])
        passes = sum(r["pass"] for r in rows)
        gate_report = {
            "cases_run": len(rows),
            "cases_expected": gate["total"],
            "passes": passes,
            "proposed_min_pass": gate["min_pass"],
            "meets_proposed_threshold": len(rows) == gate["total"] and passes >= gate["min_pass"],
        }
        if gate.get("no_runaway_required"):
            runaway_cases = [r["id"] for r in rows if r.get("runaway_ngram")]
            gate_report["any_runaway_repetition_failure"] = bool(runaway_cases)
            gate_report["meets_proposed_threshold"] = (
                gate_report["meets_proposed_threshold"] and not runaway_cases
            )
        gates[category] = gate_report

    report = {
        "status": "research_only_not_validated_release",
        "note": (
            "Gate thresholds are proposals from docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md, "
            "not approved pass/fail criteria. This script does not certify a 4K MLX "
            "release even when every gate below is met."
        ),
        "source_raw": str(args.raw),
        "revision": raw.get("revision"),
        "repo_id": raw.get("repo_id"),
        "backend": raw.get("backend"),
        "mlx_version": raw.get("mlx_version"),
        "load_seconds": raw.get("load_seconds"),
        "peak_memory_after_load_gib": raw.get("peak_memory_after_load_gib"),
        "peak_memory_overall_gib": raw.get("peak_memory_overall_gib"),
        "gates": gates,
        "per_case": per_case,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    for row in per_case:
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"  {mark} {row['id']} ({row['rubric_type']}): {row['detail']}")
    print(f"Wrote scored report to {args.report}")
    sys.exit(0)


if __name__ == "__main__":
    main()
