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

PYTHON="${HOBBYLM_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))'; then
      PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]] \
  || ! command -v "$PYTHON" >/dev/null 2>&1 \
  || ! "$PYTHON" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))'; then
  echo "Python 3.10-3.13 is required (set HOBBYLM_PYTHON to its executable)." >&2
  exit 1
fi

create_venv() {
  local venv="$1"
  local selected_version existing_version
  selected_version="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ -x "$venv/bin/python" ]]; then
    existing_version="$("$venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$existing_version" != "$selected_version" ]]; then
      echo "Recreating $venv with Python $selected_version (was $existing_version)."
      "$PYTHON" -m venv --clear "$venv"
      return
    fi
  fi
  "$PYTHON" -m venv "$venv"
}

create_venv "$HF_VENV"
"$HF_VENV/bin/python" -m pip install --upgrade pip
"$HF_VENV/bin/pip" install -r requirements-mlx-reference.txt

create_venv "$MLX_VENV"
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
