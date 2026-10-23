#!/usr/bin/env bash
# Hobby setup for Apple Silicon Macs. Downloads pinned model files, clones/builds the pinned macos-use server, builds Swift tools.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q "mlx==0.32.3" "transformers==4.46.3" "huggingface-hub==0.26.2" numpy safetensors
# FP32 Instruct weights (sha256 d0899e0a...) and the MLX release (runtime + tokenizer)
.venv/bin/huggingface-cli download harims95/hobbylm-1B-instruct --revision 8a90f6eb6a3d7c7eb952fd01b512a406821366ab --local-dir models/hobbylm-1B-instruct
.venv/bin/huggingface-cli download harims95/hobbylm-1B-mlx --local-dir models/hobbylm-1B-mlx \
  --include 'runtime/*' 'tokenizer.json' 'tokenizer_config.json' 'vocab.json' 'merges.txt' 'special_tokens_map.json' 'config.json' 'generation_config.json'
# (only the runtime + tokenizer are needed; the 3.7 GB MLX weights are skipped. Hobby runs the FP32 Instruct weights + adapter)
# macos-use (Business Source License 1.1, see its LICENSE), pinned commit
[ -d vendor/mcp-server-macos-use ] || git clone https://github.com/mediar-ai/mcp-server-macos-use vendor/mcp-server-macos-use
git -C vendor/mcp-server-macos-use checkout 205c8880e8c06fa96a3e3a859e6edca728eb6c47
(cd vendor/mcp-server-macos-use && swift build -c release)
# small Swift tools: checkers and system controls
for t in calc_checker brightness media_key checker; do swiftc -O hobby/$t.swift -o data/$t; done
echo "Done. Grant Accessibility to your terminal app, then: .venv/bin/python hobby/hobby_demo.py r4"
