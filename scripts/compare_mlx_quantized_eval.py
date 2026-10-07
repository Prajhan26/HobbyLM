"""Compare frozen FP32 and quantized MLX prompt-suite reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    reference_rows = {row["id"]: row for row in reference["results"]}
    candidate_rows = {row["id"]: row for row in candidate["results"]}
    if reference_rows.keys() != candidate_rows.keys():
        raise SystemExit("Reference and candidate prompt IDs differ")

    changed = []
    new_empty = []
    new_repetition = []
    for case_id in sorted(reference_rows):
        before = reference_rows[case_id]
        after = candidate_rows[case_id]
        if before["nonempty"] and not after["nonempty"]:
            new_empty.append(case_id)
        if not before["severe_repetition"] and after["severe_repetition"]:
            new_repetition.append(case_id)
        if before["output_ids"] != after["output_ids"]:
            changed.append(
                {
                    "id": case_id,
                    "category": before.get("category"),
                    "prompt": before["prompt"],
                    "reference_output": before["output"],
                    "candidate_output": after["output"],
                    "human_review": "pending",
                }
            )

    hard_failures = []
    if new_empty:
        hard_failures.append("new_empty_outputs")
    if new_repetition:
        hard_failures.append("new_severe_repetition")
    report = {
        "status": "fail" if hard_failures else "manual_review_required",
        "cases": len(reference_rows),
        "exact_output_sequence_matches": len(reference_rows) - len(changed),
        "changed_output_sequences": len(changed),
        "new_empty_outputs": new_empty,
        "new_severe_repetition": new_repetition,
        "hard_failures": hard_failures,
        "changed_cases": changed,
        "release_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "changed_cases"}, indent=2))
    if hard_failures:
        raise SystemExit("FAIL: quantized candidate introduced an execution regression")


if __name__ == "__main__":
    main()
