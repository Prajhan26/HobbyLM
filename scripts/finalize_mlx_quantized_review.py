"""Combine automated results and locked human review into a packaging decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text())
    comparison = json.loads(args.comparison.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    review = json.loads(args.review.read_text())

    failures: list[str] = []
    if artifact["status"] != "pass":
        failures.append("artifact_validation_failed")
    if comparison["hard_failures"]:
        failures.extend(comparison["hard_failures"])
    changed_ids = {case["id"] for case in comparison["changed_cases"]}
    reviewed_ids = {case["id"] for case in review["cases"]}
    if reviewed_ids != changed_ids:
        failures.append("manual_review_does_not_cover_exact_changed_case_set")
    if review["candidate_weight_sha256"] != artifact["weight_sha256"]:
        failures.append("manual_review_targets_a_different_weight_file")
    pass_to_fail = [
        case["id"]
        for case in review["cases"]
        if case["reference_result"] == "pass" and case["candidate_result"] == "fail"
    ]
    if pass_to_fail:
        failures.append("manual_review_found_pass_to_fail_regressions")

    remaining = (
        [
            "package a public-facing model card",
            "upload to the chosen Hugging Face organization",
            "verify a clean download and CLI run from the immutable public revision",
        ]
        if not failures
        else [
            "do not publish this weight file",
            "develop and independently validate a less aggressive quantization recipe",
        ]
    )
    decision = {
        "status": "approved_for_packaging" if not failures else "rejected",
        "public_release_complete": False,
        "candidate_weight_sha256": artifact["weight_sha256"],
        "quantization": artifact["quantization"],
        "strict_load": artifact["strict_load"],
        "router_precision": artifact["router_precision"],
        "frozen_cases": comparison["cases"],
        "exact_output_sequence_matches": comparison["exact_output_sequence_matches"],
        "manually_reviewed_changed_cases": len(review["cases"]),
        "pass_to_fail_regressions": pass_to_fail,
        "router_expert_set_agreement_on_identical_tokens": benchmark[
            "router_expert_set_agreement_on_identical_tokens"
        ],
        "failures": failures,
        "remaining_before_public_release": remaining,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))
    if failures:
        raise SystemExit("FAIL: candidate is not approved for packaging")


if __name__ == "__main__":
    main()
