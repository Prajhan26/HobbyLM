# HobbyLM-1B 4K MLX checkpoint audit

Status: Hari confirms 4K SFT training and says the original 1024 HF config was
copied from the pretrained Base export. The HF repo's newer revision now
publishes 4096 metadata with unchanged weights. This is **not** a validated
4K MLX release: strict parity and early-position retrieval have failures, and
Mac-specific behavioral acceptance has not yet been approved or run on M1 Max.

## What is established

- The previously validated public broad-SFT revision is
  `harims95/hobbylm-1b-broad-sft-3450-hf` at
  `f4b8557098c01fd42195f32feee3a3275dfbfbe3`. Its pinned `config.json`
  declares `max_position_embeddings: 1024` and `rope_theta: 10000.0`.
  The MLX v1 validator expects 1024. The mutable repo's newer 4096 config is
  described below; these revisions must not be conflated.
- The public HF model commit says it converted
  `broad_instruct_sft_v1_ckpt_3450.pt` to FP32 HF format.
- The raw training checkpoint is listed in
  `harims95/hobbylm-broad-sft-checkpoints` as
  `checkpoints/broad_instruct_sft_v1_ckpt_3450.pt` (8,513,301,775 bytes).
  Its published SHA-256 sidecar says
  `ef62430d01a551c90f0a7fd9ca3ffad29ec50fea8aaee365de6d334d7a16cc34`.
- A read-only HTTP range inspection of that raw PyTorch ZIP checkpoint's
  `ckpt_3450/data.pkl` found one saved `seq_len` field immediately followed
  by pickle `BININT2` bytes `M\x00\x10`, which encode **4096**. This inspected
  approximately 4.5 MB of metadata and did not execute the pickle or download
  the 8.5 GB tensor/optimizer payload. The saved `rope_theta` field encodes
  10000.0. The published HF config's 1024 value therefore conflicts with the
  original step-3450 training metadata.
- Hari's `HobbyLM → Colander Handoff for Prajan` identifies the 4K R2
  Step150 Base parent as
  `/data/runs/hobbylm_2k_to_4k_ext_v2_short_pythonic_retry_r2/ckpt_150.pt`,
  SHA-256 `4a0c8215317545f8ba54d0c01cc30987ed19d9b271414e03d2d25a1bd7d11639`.
  The handoff's frozen broad-SFT recipe says `CONTEXT_LENGTH: 4096`.
- There was no separately named 4K broad-SFT model in the earlier public
  listing. The same SFT repo now has a newer 4096-configured revision; the
  R2 parent is a Base checkpoint, not a substitute for step-3450 SFT weights.

The raw step-3450 SFT checkpoint was saved with a 4096-token training length.
Hari's later provenance update says the 1024 value was inherited from the
pretrained Base config, rather than being an intentional SFT context cap.
Whether the public HF weights are fully identical to a conversion of that
checkpoint is not yet proven. A model accepting 4096 input is also distinct
from useful retrieval at that depth; Hari's roadmap says deep 4K retrieval
was not certified.

## Hari's later provenance and release guidance

Hari identifies `broad_instruct_sft_v1_ckpt_3450.pt` (SHA-256
`ef62430d01a551c90f0a7fd9ca3ffad29ec50fea8aaee365de6d334d7a16cc34`)
as the intended source. He reports the published HF Safetensors at revision
`f4b8557098c01fd42195f32feee3a3275dfbfbe3` has SHA-256
`d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89`.
His comparison of 496 sections covering portions of 232 of 280 published
tensors found all sampled sections matched. This strongly supports the
lineage, but **does not prove full per-tensor identity**. These hashes and
sample counts are from Hari's update. This audit independently recomputed the
**published HF Safetensors SHA-256** from the local pinned snapshot and got
`d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89`.
It has not independently recomputed the full raw-checkpoint hash or the
sampled conversion comparison.

The exact command that produced the *public* upload has not been recovered.
Hari documented a different, earlier `convert_checkpoint.py` export, with a
converter hash beginning `af0bdc28` and export hash beginning `3298f8e7`;
its recreation command must not be attributed to the later public revision.
The raw SFT training config records `train.seq_len=4096`, plain RoPE at
theta 10000, and no YaRN. Hari said a metadata correction was prepared. A
live HF API check on 2026-10-01 confirmed it is **now published** at revision
`ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5`: its config declares
`max_position_embeddings: 4096`, `rope_theta: 10000.0`, and
`rope_scaling: null`. A direct config diff against the pinned 1024 revision
shows only the context value and explicit null RoPE-scaling field changed;
the current repo's Safetensors LFS SHA-256 is still
`d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89`.
This verifies a **metadata correction with unchanged published weights**,
not 4K MLX behavioral acceptance.

The shipping 1K validator and CLI resolve the model name without an immutable
revision. A fresh default download may now see the 4096 config; the
validator's explicit 1024 expectation will reject it. This is a release
reproducibility issue to handle deliberately. The original M1 Max validation
artifacts correspond to the prior 1024 revision, not mutable repo head.

Hari proposes that exact PyTorch/MLX bit parity is not an automatic Mac
release requirement. Faithful FP32 routing semantics and acceptable measured
Mac behavior are required. Exact ties and numerical near-ties must be
distinguished; token and expert-set mismatches must remain visible, with
routing compared on identical input token sequences. Once generated tokens
diverge, subsequent route comparisons are not same-input comparisons.
Merely reporting mismatches is insufficient: calling accuracy, instruction
following, repetition, and long-context behavior need acceptance tests.
**Hari has not yet approved a Mac-specific tolerance or pass threshold.**
The proposed thresholds and decision process are in
`docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md` and require approval before the M1 Max
release run.

## Research-only long-context probes

The published 1024-configured FP32 SFT export can mechanically process long
inputs, but that does not establish a supported context length. The isolated
`scripts/probe_mlx_4k_candidate.py` compares the final-token logits, next
token, and final-token expert sets from PyTorch and MLX without changing the
shipping model or config:

- At 3900 tokens, both choose next-token ID 383 and all 19 MoE-layer expert
  sets match. Mean absolute logit error is about 0.0135; MLX peak memory on a
  local 16 GiB M4 Mac is about 5.51 GiB.
- At 4096 tokens, both still choose next-token ID 383, but only 18 of 19
  MoE-layer expert sets match. In MoE layer index 16 (transformer layer 17),
  PyTorch selects expert 41 while MLX selects expert 62; the other seven
  experts match. Mean absolute logit error is about 0.0613 and MLX peak
  memory is about 5.55 GiB. This is a **failed parity test**, irrespective
  of the matching next-token ID. Neither backend was changed to mask it.
- A tiny synthetic 3900-token access-code probe retrieves `BIRCH-741` when
  inserted at token positions 1920 and 3584, but fails at position 256. Three
  prompts cannot establish reliable retrieval; this result is a warning, not
  an evaluation score.

Local ignored reports and fixtures live under `artifacts/hobbylm-4k-candidate-*`.
These probes ran on an M4 Mac, not the target M1 Max. The later committed
`archive-512` diagnostic below explains a different short-context divergence;
the 4096-token layer-17 flip has not received the same full diagnosis.

## Separate local candidate (not published)

Run `python scripts/prepare_mlx_4k_candidate.py` in an environment with
`huggingface_hub` installed. It pins the public SFT revision above, creates
`artifacts/hobbylm-1b-sft-3450-4k-candidate/`, copies its HF model code,
changes only a copied `config.json` from 1024 to 4096, and symlinks the same
FP32 Safetensors weight file. A manifest records the source revision and
unresolved raw-checkpoint identity. The source HF repo and 1K CLI default are
untouched. This directory is local and ignored by Git; its weight symlink
depends on the local Hugging Face cache.

The candidate loaded with strict MLX weight matching on the M4 Mac. Against
the saved PyTorch fixture, its 3900-token next-token and all expert sets pass;
its 4096-token run reproduces the same one-layer expert-set failure described
above. A context metadata change alone therefore does not certify 4K.

An expanded, still-small deterministic sample uses three filler variants in
`scripts/probe_mlx_4k_candidate.py`. Each case compares the final prompt
position only, not every token in the prompt:

| Tokens | Variant | Next token | Expert sets | Result |
| ---: | --- | --- | --- | --- |
| 512 | code | 383 on both | 19/19 | pass |
| 2048 | science | 317 on both | 18/19 | fail: MoE index 1 |
| 3900 | archive | 383 on both | 19/19 | pass |
| 4096 | archive | 383 on both | 18/19 | fail: MoE index 16 |
| 4096 | code | 383 on both | 18/19 | fail: MoE index 5 |

The mismatches at 2048 and 4096 occur in different MoE layers. This sample
is too small to estimate a failure rate and does not establish generation
sequence parity, retrieval quality, or M1 Max performance. Matching next
tokens do not override the failed strict expert-set checks. Local fixtures
and reports for these cases are ignored under `artifacts/`.

## Frozen research grids (M4 Mac, 16 GiB)

`eval/mlx_4k_research_suite.json` freezes three prompt variants at four
lengths (12 final-position parity cases) and two access codes at three lengths
and three needle positions (18 retrieval cases). With the local candidate
prepared, the commands are:

```bash
.venv-hf-reference/bin/python scripts/evaluate_mlx_4k_parity.py pytorch
bash scripts/run_mlx_4k_parity_isolated.sh
bash scripts/run_mlx_4k_retrieval_isolated.sh
```

The PyTorch run completed all 12 references. The MLX parity runner uses a
fresh process per case because a single-process batch caused severe swapping
on this 16 GiB machine. It intentionally exits nonzero when any strict case
fails. Ignored per-case fixtures and reports are in
`artifacts/mlx-4k-research-parity/`; retrieval reports are in
`artifacts/mlx-4k-research-retrieval-*.json`.

| Prompt tokens | Strict parity passes | Exact retrieval |
| ---: | ---: | ---: |
| 512 parity / 1024 retrieval | 1/3 | 6/6 |
| 2048 | 0/3 | 6/6 |
| 3900 | 2/3 | 4/6 |
| 4096 parity only | 1/3 | not run |
| Total | **4/12** | **16/18** |

The final prompt-position next token matched in 11/12 parity cases. The
`archive-512` case differs in both next token (PyTorch 383, MLX 921) and one
expert set; the same mismatch reproduces with the original published
1024-configured model. Thus the earlier M1 Max 1K release result remains a
pass for its *specific frozen prompt*, but it does not establish broad 1K
parity. Other cases can match the next token while choosing different expert
sets. These are observations, not a proof of a router implementation bug or
an acceptable tie tolerance.

At 3900 tokens, both codes are retrieved from middle and late positions, but
both fail at the early 10% position. This simple repetitive synthetic task
does not measure real-world chat quality or certify long-context usefulness.
The retrieval grid uses MLX only; it is not a PyTorch/MLX generation-parity
test. All runs were on a local M4, not the target M1 Max. No training,
quantization, router-behavior, or 1K release changes were made.

## archive-512 divergence diagnosis (research only)

Reproduce with the pinned public revision (the 1024-configured model; weights
are identical to the 4K-config candidate):

```bash
.venv-hf-reference/bin/python scripts/diagnose_mlx_layer_divergence.py pytorch --dtype float32
.venv-mlx-release/bin/python  scripts/diagnose_mlx_layer_divergence.py mlx
.venv-hf-reference/bin/python scripts/diagnose_mlx_layer_divergence.py pytorch --dtype float64  # oracle only
.venv-hf-reference/bin/python scripts/diagnose_mlx_layer_divergence.py analyze
```

Raw tensors stay in ignored `artifacts/archive512-divergence/`; the tracked
result is `eval/mlx_archive512_divergence/summary.json`. The float64 run is a
diagnostic oracle (same weights upcast exactly, router also in float64); it is
not a product path and the FP32 router in both backends is untouched. M4 Mac,
torch 2.14 / transformers 4.46.3, MLX 0.32.3, prompt SHA-256 `d3c1d267...30bb`.

Findings:

- **Replication is exact.** The op-by-op MLX replay equals the library forward
  (max logit difference 0.0) and reproduces token 921. PyTorch gives 383.
- **Layer 0 is clean.** Every op matches to about 5e-7 relative, the same noise
  as PyTorch FP32 versus FP64. The first op above 1e-5 is transformer layer 1's
  MoE output (9.5e-3), not attention, norms, RoPE, or the router logits (5e-7).
- **Cause at layer 1: exact FP32 ties.** Router selection is sigmoid score plus
  `expert_bias` (about 97 to 99 here). FP32 spacing at that scale is 7.6e-6, so
  distinct scores collapse into equal selection values. Feeding MLX PyTorch's
  exact router input still changes 322 token/layer expert sets across all MoE
  layers, and all 322 are exact ties at the rank-8/rank-9 boundary. MLX
  `argpartition` kept the lower expert index in 322/322; PyTorch `topk` kept
  the higher index in 263/322 and a mixed choice in the rest. These are
  different tie-break behaviours, not a weight or math error.
- **Scale.** Over 9,728 token/layer pairs, 479 are exact ties in PyTorch, and
  the boundary margin is below 1e-4 for 28% and below 1e-3 for 74%. A free run
  has 651 differing sets; the audit's last-position check saw one layer.
- **Weights and ops exonerated.** With PyTorch's expert indices forced in all
  MoE layers, MLX final logits match PyTorch to 3.1e-5 max and choose 383.
  Forcing only transformer layer 18 does not recover it (still 921).
- **Transformer layer 18 (MoE index 17), last position.** Given PyTorch's
  input, MLX picks PyTorch's set. In the free run upstream drift moves router
  logits by up to 0.37 and the 8th/9th margin is 1.5e-4 (20 FP32 steps),
  so that flip is drift from earlier tie flips, not an exact tie.
- **The token is a near-tie.** Logit gap 383 minus 921: PyTorch FP32 +0.0014,
  MLX FP32 -0.145, FP64 oracle +0.087. FP32 PyTorch and MLX are equally far
  from the oracle after layer 1 (7.07e-3 vs 7.08e-3).

Conclusion: the failure follows from tie-ridden FP32 top-8 selection amplified
by the large router bias, plus a different tie-break rule in each backend. It
is not evidence of a weight-mapping or operator bug.

Remaining uncertainty: which tie-break matches the original CUDA training run
is unknown, and the FP64 oracle is not ground truth for the trained model.
One prompt, final position token only, M4 not M1 Max. No fix is made; any
tie-break alignment would be a behaviour decision to make deliberately, and the
1K release and parity criteria are unchanged.

## Evidence needed before a 4K release

1. Freeze and approve Mac-specific release criteria before using M1 Max
   results to make a release decision. This is still pending.
2. Verify the public HF export provenance further if a definitive lineage
   claim is desired. Sampled tensor agreement and Hari's account are strong
   evidence, but the public converter command and full identity remain open.
3. Preserve strict final-token and expert-set comparisons as diagnostics;
   measure tie and near-tie effects on identical inputs. Do not change FP32
   router or architecture behavior just to make parity pass.
4. Run frozen Mac behavioral and retrieval evaluations on the target M1 Max,
   plus load, memory, speed, and missing/unexpected-weight checks. Keep raw
   outputs and report failures even if the release criteria allow some
   backend-dependent tied-expert selections.
5. Decide whether to validate and distribute the new immutable 4096 HF
   revision as 4K MLX. Do not silently re-label the old M1 Max artifacts or
   call the new metadata a behavioral validation. Preserve a reproducible
   path to the original pinned 1K revision.

Do not treat the newer 4096 HF metadata or a raised CLI cap as a test pass.
No training, fine-tuning, quantization, or Modal work is authorized by this
audit.

## Bounded coding handoff

An implementation agent may build a separate 4K validation path after the
checkpoint provenance gate is resolved. It must use an immutable model revision,
leave the 1024 MLX v1 default unchanged, save raw parity and retrieval artifacts,
and fail closed on missing or unexpected weights. A DeepSeek coding session can
take this task using this audit as its input; it cannot supply missing checkpoint
provenance by inference.
