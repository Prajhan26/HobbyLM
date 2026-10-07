#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-mlx-release/bin/python}"
REFERENCE_MODEL="${REFERENCE_MODEL:-hobbylm-model}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-work/hobbylm-1b-mlx-8bit}"
REPORT_DIR="${REPORT_DIR:-artifacts/mlx-8bit-release}"

mkdir -p "$REPORT_DIR"

echo "[1/6] Quantized module round-trip"
"$PYTHON_BIN" scripts/test_mlx_quantization.py

echo "[2/6] Artifact structure, provenance, strict loading, and FP32 router"
"$PYTHON_BIN" scripts/validate_mlx_quantized_artifact.py \
  --model "$CANDIDATE_MODEL" \
  --report "$REPORT_DIR/artifact.json"

echo "[3/6] FP32 frozen prompt suite"
"$PYTHON_BIN" scripts/evaluate_mlx_v1.py \
  --model "$REFERENCE_MODEL" \
  --output "$REPORT_DIR/fp32-eval.json" \
  --max-tokens 32

echo "[4/6] 8-bit frozen prompt suite"
"$PYTHON_BIN" scripts/evaluate_mlx_v1.py \
  --model "$CANDIDATE_MODEL" \
  --output "$REPORT_DIR/8bit-eval.json" \
  --max-tokens 32

echo "[5/6] Frozen-suite comparison"
"$PYTHON_BIN" scripts/compare_mlx_quantized_eval.py \
  --reference "$REPORT_DIR/fp32-eval.json" \
  --candidate "$REPORT_DIR/8bit-eval.json" \
  --output "$REPORT_DIR/comparison.json"

echo "[6/6] Deterministic token, routing, memory, and speed comparison"
"$PYTHON_BIN" scripts/validate_mlx_quantized.py \
  --reference-model "$REFERENCE_MODEL" \
  --quantized-model "$CANDIDATE_MODEL" \
  --report "$REPORT_DIR/benchmark.json" \
  --max-tokens 32

echo
echo "Infrastructure validation complete."
echo "Reports: $REPORT_DIR"
echo "Release status remains manual_review_required until changed outputs are approved."
