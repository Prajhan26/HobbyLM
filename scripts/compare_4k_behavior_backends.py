"""Compare frozen PyTorch and MLX behavior without relabeling token mismatches."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-raw", type=Path, required=True)
    parser.add_argument("--pytorch-raw", type=Path, required=True)
    parser.add_argument("--mlx-score", type=Path, required=True)
    parser.add_argument("--pytorch-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlx-4k-behavior-eval/comparison.json"))
    args = parser.parse_args()
    mlx = json.loads(args.mlx_raw.read_text())
    pytorch = json.loads(args.pytorch_raw.read_text())
    mlx_score = json.loads(args.mlx_score.read_text())
    pytorch_score = json.loads(args.pytorch_score.read_text())
    if mlx["revision"] != pytorch["revision"] or mlx["weight_sha256"] != pytorch["weight_sha256"]:
        raise SystemExit("Model revision or weight hash differs between backends")
    if mlx["prompts_sha256"] != pytorch["prompts_sha256"]:
        raise SystemExit("Prompt fixture differs between backends")
    if mlx["backend"] != "mlx" or pytorch["backend"] != "pytorch_cpu":
        raise SystemExit("Expected MLX and PyTorch CPU reports")
    for raw, scored in ((mlx, mlx_score), (pytorch, pytorch_score)):
        if raw["revision"] != scored["revision"]:
            raise SystemExit("Raw and score revision differ")
    mx_rows = {row["id"]: row for row in mlx["results"]}
    pt_rows = {row["id"]: row for row in pytorch["results"]}
    mx_scores = {row["id"]: row for row in mlx_score["per_case"]}
    pt_scores = {row["id"]: row for row in pytorch_score["per_case"]}
    ids = set(mx_rows)
    if not ids or ids != set(pt_rows) or ids != set(mx_scores) or ids != set(pt_scores):
        raise SystemExit("Case sets differ or are empty")
    rows = []
    by_category = defaultdict(list)
    for case_id in sorted(ids):
        a, b = mx_rows[case_id], pt_rows[case_id]
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise SystemExit(f"Prompt token IDs differ for {case_id}")
        if a["category"] != b["category"]:
            raise SystemExit(f"Category differs for {case_id}")
        row = {
            "id": case_id,
            "category": a["category"],
            "prompt_sha256": a["prompt_sha256"],
            "output_ids_match": a["output_ids"] == b["output_ids"],
            "mlx_pass": mx_scores[case_id]["pass"],
            "pytorch_pass": pt_scores[case_id]["pass"],
            "mlx_output": a["output_text"],
            "pytorch_output": b["output_text"],
        }
        rows.append(row)
        by_category[a["category"]].append(row)
    summary = {
        "status": "research_only_not_validated_release",
        "revision": mlx["revision"],
        "weight_sha256": mlx["weight_sha256"],
        "prompts_sha256": mlx["prompts_sha256"],
        "same_input_cases": len(rows),
        "matching_output_sequences": sum(row["output_ids_match"] for row in rows),
        "categories": {
            category: {
                "cases": len(category_rows),
                "mlx_pass": sum(row["mlx_pass"] for row in category_rows),
                "pytorch_pass": sum(row["pytorch_pass"] for row in category_rows),
                "mlx_additional_failures": sum(
                    row["pytorch_pass"] and not row["mlx_pass"] for row in category_rows
                ),
            }
            for category, category_rows in sorted(by_category.items())
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(f"Wrote detailed comparison to {args.output}")


if __name__ == "__main__":
    main()
