#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This validation package requires an Apple-Silicon Mac (arm64)." >&2
  exit 1
fi

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
if (( FREE_KB < 20971520 )); then
  echo "At least 20 GiB of free disk space is required." >&2
  exit 1
fi

MLX_VENV="${HOBBYLM_MLX_VENV:-.venv-mlx-release}"
HF_VENV="${HOBBYLM_HF_VENV:-.venv-hf-reference}"
MODEL="${HOBBYLM_MODEL:-harims95/hobbylm-1b-broad-sft-3450-hf}"

python3 -m venv "$HF_VENV"
"$HF_VENV/bin/python" -m pip install --upgrade pip
"$HF_VENV/bin/pip" install -r requirements-mlx-reference.txt

python3 -m venv "$MLX_VENV"
"$MLX_VENV/bin/python" -m pip install --upgrade pip
"$MLX_VENV/bin/pip" install -r requirements-mlx.txt

echo "[1/3] Testing native and Hugging Face weight adapters with a tiny model"
"$MLX_VENV/bin/pip" install torch
"$MLX_VENV/bin/python" scripts/test_mlx_parity.py

echo "[2/3] Creating the deterministic PyTorch reference (CPU, no cloud GPU)"
"$HF_VENV/bin/python" scripts/create_hf_reference.py --model "$MODEL"

echo "[3/3] Validating MLX logits, tokens, routes, memory, and speed"
"$MLX_VENV/bin/python" scripts/validate_mlx_release.py --model "$MODEL"

echo
echo "Validation finished. Reports are in artifacts/."
echo "Start local chat with:"
echo "  $MLX_VENV/bin/python -m mlx_hobbylm.chat --model $MODEL"
