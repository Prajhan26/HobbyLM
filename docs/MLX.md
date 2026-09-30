# HobbyLM on Apple MLX

This workstream gives Mac users a native Apple-Silicon inference path for the
HobbyLM text model. MLX is the numerical runtime; it is not the product UI, a
model store, or a document-retrieval system.

## What a user gets

The first release is a local terminal chat:

```text
$ python -m mlx_hobbylm.chat
Loading rootxhacker/HobbyLM-Chat on Apple Silicon…
Ready — 16 layers, 36 experts, top-6. Processing stays on this Mac.

You: Explain sparse routing in one sentence.
HobbyLM: ...
```

On first use, the public checkpoint is downloaded from Hugging Face. After it
is cached, generation runs locally. A later desktop application can wrap this
same runtime, and a later RAG layer can retrieve passages from local files.

## Developer setup

Requirements: an Apple-Silicon Mac and Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm safetensors huggingface_hub tiktoken torch
python scripts/test_mlx_parity.py
python -m mlx_hobbylm.chat --prompt "Give me three ways to take better research notes."
```

The parity test creates a tiny deterministic HobbyLM, loads identical weights
into PyTorch and MLX, and compares the logits. Do not publish an MLX checkpoint
until this passes and the public 500M checkpoint has also passed fixed-prompt
and expert-routing checks.

## Current scope

- Text-only autoregressive HobbyLM checkpoints
- Native sparse selected-expert computation through MLX gather matrix multiplies
- GPT-2 tokenization and HobbyLM-Chat's `USER:` / `ASSISTANT:` prompt format
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
