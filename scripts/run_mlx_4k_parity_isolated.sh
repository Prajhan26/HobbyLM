#!/usr/bin/env bash
# Research-only: use a fresh MLX process per frozen case to bound memory growth.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
MLX_PYTHON="${HOBBYLM_MLX_PYTHON:-.venv-mlx-release/bin/python}"

passes=0
failures=0
for variant in archive science code; do
  for length in 512 2048 3900 4096; do
    case_id="${variant}-${length}"
    echo "Running frozen parity case: $case_id"
    if "$MLX_PYTHON" scripts/evaluate_mlx_4k_parity.py mlx --case "$case_id"; then
      passes=$((passes + 1))
    else
      failures=$((failures + 1))
    fi
  done
done

echo "Research-only strict parity: $passes passed, $failures failed (12 cases)."
if (( failures > 0 )); then
  exit 1
fi
