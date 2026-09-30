# HobbyLM on Apple MLX

This workstream gives Mac users a native Apple-Silicon inference path for the
HobbyLM text model. MLX is the numerical runtime; it is not the product UI, a
model store, or a document-retrieval system.

## What a user gets

The first release is a local terminal chat using the confirmed 1B broad-SFT
checkpoint derived from the final step-95,367 annealed Base:

```text
$ python -m mlx_hobbylm.chat
Loading harims95/hobbylm-1b-broad-sft-3450-hf on Apple Silicon…
Ready — 20 layers, 64 experts, top-8. Processing stays on this Mac.

You: Explain sparse routing in one sentence.
HobbyLM: ...
```

On first use, the public checkpoint is downloaded from Hugging Face. After it
is cached, generation runs locally. A later desktop application can wrap this
same runtime, and a later RAG layer can retrieve passages from local files.

## One-command M1 Max validation

Requirements: Apple Silicon, Python 3.10-3.13, and at least 20 GiB free disk.
The script uses separate PyTorch-reference and MLX environments because the
released checkpoint pins Transformers 4.46.3 while current MLX tooling uses a
newer Hugging Face client. It also runs them sequentially so both copies of the
model are never resident together.

```bash
git clone <HobbyLM repository URL>
cd HobbyLM
bash scripts/run_mlx_release_validation.sh
```

It creates:

- `artifacts/hobbylm-1b-sft-reference.npz` — deterministic PyTorch fixture
- `artifacts/hobbylm-1b-sft-reference.json` — prompt and environment metadata
- `artifacts/hobbylm-1b-sft-mlx-report.json` — parity, routing, memory and speed

No cloud GPU and no training are involved. The first run downloads the 4.35 GB
public SFT checkpoint. The verified prompt serialization is literal
`SYSTEM:`, `USER:`, and `ASSISTANT:` text using the GPT-2 tokenizer.

## Manual developer setup

Requirements: an Apple-Silicon Mac and Python 3.10-3.13.

```bash
python3 -m venv .venv-hf-reference
.venv-hf-reference/bin/pip install -r requirements-mlx-reference.txt
.venv-hf-reference/bin/python scripts/create_hf_reference.py

python3 -m venv .venv-mlx-release
.venv-mlx-release/bin/pip install -r requirements-mlx.txt torch
.venv-mlx-release/bin/python scripts/test_mlx_parity.py
.venv-mlx-release/bin/python scripts/validate_mlx_release.py
.venv-mlx-release/bin/python -m mlx_hobbylm.chat --prompt "Give me three ways to take better research notes."
```

The parity test creates a tiny deterministic HobbyLM, loads identical weights
into PyTorch and MLX, and compares the logits. Do not publish an MLX checkpoint
until this passes and the public 1B SFT has also passed fixed-prompt and
expert-routing checks.

## Current scope

- Text-only autoregressive HobbyLM checkpoints
- Native sparse selected-expert computation through MLX gather matrix multiplies
- GPT-2 tokenization and the confirmed `SYSTEM:` / `USER:` / `ASSISTANT:` format
- Greedy or temperature sampling with a repetition penalty

## Deliberately deferred

- RAG and local-document indexing
- Julia Browser automation
- Multimodal and diffusion checkpoints
- A desktop installer
- Quantization
- KV caching and production serving

The initial generator recomputes the current context for every token. It is a
correctness milestone, not the final performance implementation. Add a KV cache
only after parity with the PyTorch reference is established.

## Release gate

1. Tiny-model numerical parity passes.
2. Public HobbyLM-Chat downloads and loads without manual weight editing.
3. Fixed greedy prompts produce matching PyTorch and MLX token sequences.
4. Selected expert IDs agree on the test prompts.
5. Memory and tokens-per-second are recorded on at least one target Mac.
6. The README states the model's capability limits and supported hardware.
