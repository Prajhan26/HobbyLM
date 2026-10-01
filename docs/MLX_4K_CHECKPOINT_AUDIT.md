# HobbyLM-1B 4K MLX checkpoint audit

Status: 4K training length confirmed in the raw SFT checkpoint; a 4096-token
parity probe fails expert-set agreement, and retrieval is inconsistent. This
does not certify a 4K MLX release.

## What is established

- The current public broad-SFT model is
  `harims95/hobbylm-1b-broad-sft-3450-hf`, revision
  `f4b8557098c01fd42195f32feee3a3275dfbfbe3`. Its published
  `config.json` declares `max_position_embeddings: 1024` and
  `rope_theta: 10000.0`. The MLX v1 validator correctly expects 1024.
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
- A public Hugging Face model listing under `harims95` shows no separately
  published 4K broad-SFT model. The R2 parent is a Base training checkpoint,
  not a replacement for the step-3450 SFT release.

The raw step-3450 SFT checkpoint was saved with a 4096-token training length.
The remaining provenance question is whether the public HF weights were
converted exactly from that checkpoint and whether the 1024 limit was an
intentional export decision. A model accepting 4096 input is also distinct
from useful retrieval at that depth; Hari's roadmap says deep 4K retrieval
was not certified.

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
These probes ran on an M4 Mac, not the target M1 Max. A separate scratchpad
inspection reported a near-tie in layer 17's FP32 router scores; that analysis
is not yet a committed reproducible diagnostic. The HF export's 1024 metadata
remains unexplained.

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

1. Confirm the exact HF export command or converter configuration and explain
   why it wrote `max_position_embeddings: 1024` despite the raw checkpoint's
   `seq_len: 4096`. Record the parent identity and conversion provenance.
2. Confirm that the public HF weights really came from that exact checkpoint.
   Preserve a revision and per-tensor conversion check; do not overwrite the
   current 1024-configured release.
3. Characterize the observed expert-set disagreements without changing the
   FP32 router or architecture semantics. If the original
   checkpoint and export provenance support 4096, publish a separate 4K
   candidate with accurate config. Run PyTorch and MLX parity at
   short, 2K, and near-4K lengths: logits, greedy token IDs, and expert sets.
   Keep router computation FP32 and weight loading strict.
4. Test shallow and deep retrieval separately, alongside memory and tokens per
   second on the target Mac. Report failures as limitations, even if sequence
   acceptance and numerical parity pass.

Do not relabel the current HF config or raise the CLI cap to make a test pass.
No training, fine-tuning, quantization, or Modal work is authorized by this
audit.

## Bounded coding handoff

An implementation agent may build a separate 4K validation path after the
checkpoint provenance gate is resolved. It must use an immutable model revision,
leave the 1024 MLX v1 default unchanged, save raw parity and retrieval artifacts,
and fail closed on missing or unexpected weights. A DeepSeek coding session can
take this task using this audit as its input; it cannot supply missing checkpoint
provenance by inference.
