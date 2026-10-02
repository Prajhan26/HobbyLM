# 4K MLX M1 Max research run

This is a separate 4K candidate, not a change to the published 1K release. The frozen checks record exact token and expert-set disagreements. A completed run is **not** a validated release; the proposed acceptance gates in `docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md` still need approval.

## Before starting

Use an M1 Max with at least 20 GiB of free disk space, internet access, and the `feat/mlx-4k-behavior-eval` branch. Do not use the stale 1K Hugging Face revision for the behavior run. The scripts pin the 4096-config revision `ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5` and verify its FP32 weight hash.

Create the two environments if this clone does not already have them:

```bash
python3 -m venv .venv-mlx-release
.venv-mlx-release/bin/python -m pip install -r requirements-mlx.txt
python3 -m venv .venv-hf-reference
.venv-hf-reference/bin/python -m pip install -r requirements-mlx-reference.txt
```

Then run from the repository root:

```bash
bash scripts/run_mlx_4k_m1_research.sh
```

If the environments live elsewhere, set `HOBBYLM_MLX_PYTHON` and `HOBBYLM_HF_PYTHON` to their Python executables. The first run downloads the pinned model; do not remove the Hugging Face cache until the reports have been copied safely.

## What success looks like

The command completes all 12 parity cases, 18 retrieval cases, and 30 behavior cases per backend. Reports under `artifacts/` contain the pinned revision, weight and prompt hashes, exact token/routing results, retrieval answers, peak MLX memory, and measured speed. Preserve both raw reports, both rubric reports, the comparison report, and the parity/retrieval reports. A nonzero strict parity status is **not** converted into a pass.

Review calling, instruction-following, repetition, and retrieval against the *approved* gates. If failures remain, keep 4K labeled experimental and document them. Do not change router precision, architecture, weights, or the website to make a gate pass. The local M4 pilot scored 6/10 calling, 3/10 instruction, and 9/10 repetition on both PyTorch and MLX; 26/30 output-ID sequences matched. Those figures are a warning, not M1 Max results.

For a manual candidate smoke test after the research run, install the CLI in a Python 3.10–3.13 MLX environment and pin the same revision:

```bash
.venv-mlx-release/bin/python -m pip install -e .
.venv-mlx-release/bin/python -m mlx_hobbylm.chat \
  --model harims95/hobbylm-1b-broad-sft-3450-hf \
  --revision ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5 \
  --context 4096 --temperature 0 \
  --prompt "Explain what a sparse expert is in one sentence."
```

If the MLX environment uses Python 3.14, make a separate Python 3.13 environment for this package instead; `pyproject.toml` currently supports Python below 3.14. A smoke test is not certification.
