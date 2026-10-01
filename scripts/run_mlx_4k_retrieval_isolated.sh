#!/usr/bin/env bash
# Research-only: one process per frozen retrieval case to bound memory use.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1
MLX_PYTHON="${HOBBYLM_MLX_PYTHON:-.venv-mlx-release/bin/python}"

errors=0
for length in 1024 2048 3900; do
  for position in 0.1 0.5 0.9; do
    for code in BIRCH-741 ORBIT-206; do
      echo "Running frozen retrieval case: $length / $position / $code"
      if ! "$MLX_PYTHON" scripts/evaluate_mlx_4k_retrieval.py \
        --length "$length" --position "$position" --code "$code"; then
        errors=$((errors + 1))
      fi
    done
  done
done

echo "Research-only retrieval grid: $((18 - errors)) completed, $errors execution errors."
if (( errors > 0 )); then
  exit 1
fi
