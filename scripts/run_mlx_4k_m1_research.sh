#!/usr/bin/env bash
# Research-only M1 Max run. This script cannot certify a 4K release by itself.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon is required for the MLX half of this run." >&2
  exit 1
fi
FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
if (( FREE_KB < 20971520 )); then
  echo "At least 20 GiB free disk space is required before starting." >&2
  exit 1
fi

MLX_PYTHON="${HOBBYLM_MLX_PYTHON:-$ROOT_DIR/.venv-mlx-release/bin/python}"
HF_PYTHON="${HOBBYLM_HF_PYTHON:-$ROOT_DIR/.venv-hf-reference/bin/python}"
for executable in "$MLX_PYTHON" "$HF_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing environment: $executable" >&2
    echo "Create it and install the matching requirements file before rerunning." >&2
    exit 1
  fi
done
export HOBBYLM_MLX_PYTHON="$MLX_PYTHON"

candidate="artifacts/hobbylm-1b-sft-3450-4k-candidate"
if [[ ! -f "$candidate/candidate-manifest.json" ]]; then
  "$MLX_PYTHON" scripts/prepare_mlx_4k_candidate.py
fi

echo "[1/7] Frozen PyTorch final-position parity references"
"$HF_PYTHON" scripts/evaluate_mlx_4k_parity.py pytorch

echo "[2/7] Frozen MLX parity cases (strict mismatches are expected and remain visible)"
parity_status=0
bash scripts/run_mlx_4k_parity_isolated.sh || parity_status=$?

echo "[3/7] Frozen MLX long-context retrieval"
bash scripts/run_mlx_4k_retrieval_isolated.sh

echo "[4/7] Frozen 30-prompt MLX behavior run on pinned 4096 HF revision"
"$MLX_PYTHON" scripts/run_mlx_4k_behavior_eval.py

echo "[5/7] Same-prompt PyTorch CPU behavior baseline"
"$HF_PYTHON" scripts/run_pytorch_4k_behavior_eval.py

echo "[6/7] Score both backends against frozen per-case rubrics"
"$HF_PYTHON" scripts/score_mlx_4k_behavior_eval.py \
  --raw artifacts/mlx-4k-behavior-eval/raw.json \
  --report artifacts/mlx-4k-behavior-eval/report.json
"$HF_PYTHON" scripts/score_mlx_4k_behavior_eval.py \
  --raw artifacts/mlx-4k-behavior-eval/pytorch-raw.json \
  --report artifacts/mlx-4k-behavior-eval/pytorch-report.json

echo "[7/7] Verify same prompts and compare outcomes"
"$HF_PYTHON" scripts/compare_4k_behavior_backends.py \
  --mlx-raw artifacts/mlx-4k-behavior-eval/raw.json \
  --pytorch-raw artifacts/mlx-4k-behavior-eval/pytorch-raw.json \
  --mlx-score artifacts/mlx-4k-behavior-eval/report.json \
  --pytorch-score artifacts/mlx-4k-behavior-eval/pytorch-report.json

echo
echo "Research run complete. Reports are under artifacts/."
echo "Strict parity runner exit status: $parity_status (nonzero means at least one mismatch or execution error)."
echo "Do not call this a validated 4K release until the proposed gates are approved and reviewed."
