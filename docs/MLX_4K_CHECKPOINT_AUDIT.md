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
These probes ran on an M4 Mac, not the target M1 Max. The exact router
disagreement cause and the HF export's 1024 metadata remain unresolved.

## Evidence needed before a 4K release

1. Confirm the exact HF export command or converter configuration and explain
   why it wrote `max_position_embeddings: 1024` despite the raw checkpoint's
   `seq_len: 4096`. Record the parent identity and conversion provenance.
2. Confirm that the public HF weights really came from that exact checkpoint.
   Preserve a revision and per-tensor conversion check; do not overwrite the
   current 1024-configured release.
3. Explain and fix the layer-17 4096-token expert-set disagreement without
   changing the FP32 router or architecture semantics. If the original
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
