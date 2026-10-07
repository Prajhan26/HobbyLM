#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-mlx-release/bin/python}"
REFERENCE_MODEL="${REFERENCE_MODEL:-hobbylm-model}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-work/hobbylm-1b-mlx-8bit-routed-experts}"
REPORT_DIR="${REPORT_DIR:-artifacts/mlx-8bit-routed-experts-release}"

mkdir -p "$REPORT_DIR"

echo "[1/7] Quantized module round-trip"
"$PYTHON_BIN" scripts/test_mlx_quantization.py

echo "[2/7] Artifact structure, provenance, strict loading, and FP32 router"
"$PYTHON_BIN" scripts/validate_mlx_quantized_artifact.py \
  --model "$CANDIDATE_MODEL" \
  --report "$REPORT_DIR/artifact.json"

echo "[3/7] FP32 frozen prompt suite"
"$PYTHON_BIN" scripts/evaluate_mlx_v1.py \
  --model "$REFERENCE_MODEL" \
  --output "$REPORT_DIR/fp32-eval.json" \
  --max-tokens 32

echo "[4/7] 8-bit frozen prompt suite"
"$PYTHON_BIN" scripts/evaluate_mlx_v1.py \
  --model "$CANDIDATE_MODEL" \
  --output "$REPORT_DIR/8bit-eval.json" \
  --max-tokens 32

echo "[5/7] Frozen-suite comparison"
"$PYTHON_BIN" scripts/compare_mlx_quantized_eval.py \
  --reference "$REPORT_DIR/fp32-eval.json" \
  --candidate "$REPORT_DIR/8bit-eval.json" \
  --output "$REPORT_DIR/comparison.json"

echo "[6/7] Deterministic token, routing, memory, and speed comparison"
"$PYTHON_BIN" scripts/validate_mlx_quantized.py \
  --reference-model "$REFERENCE_MODEL" \
  --quantized-model "$CANDIDATE_MODEL" \
  --report "$REPORT_DIR/benchmark.json" \
  --max-tokens 32

echo "[7/7] Locked manual review and packaging decision"
"$PYTHON_BIN" scripts/finalize_mlx_quantized_review.py \
  --artifact "$REPORT_DIR/artifact.json" \
  --comparison "$REPORT_DIR/comparison.json" \
  --benchmark "$REPORT_DIR/benchmark.json" \
  --review eval/mlx_8bit_routed_experts_review.json \
  --output "$REPORT_DIR/decision.json"

echo
echo "Infrastructure validation complete."
echo "Reports: $REPORT_DIR"
echo "See decision.json for packaging approval and remaining public-release work."
