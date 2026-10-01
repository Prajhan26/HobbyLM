# HobbyLM-1B 4K MLX checkpoint audit

Status: provenance gate open. This document does not certify a 4K MLX release.

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
- Hari's `HobbyLM → Colander Handoff for Prajan` identifies the 4K R2
  Step150 Base parent as
  `/data/runs/hobbylm_2k_to_4k_ext_v2_short_pythonic_retry_r2/ckpt_150.pt`,
  SHA-256 `4a0c8215317545f8ba54d0c01cc30987ed19d9b271414e03d2d25a1bd7d11639`.
  The handoff's frozen broad-SFT recipe says `CONTEXT_LENGTH: 4096`.
- A public Hugging Face model listing under `harims95` shows no separately
  published 4K broad-SFT model. The R2 parent is a Base training checkpoint,
  not a replacement for the step-3450 SFT release.

The unresolved question is whether the step-3450 SFT weights were trained at
4096 and only the HF export metadata says 1024, or whether this exported model
was intentionally limited to 1024. The parent lineage and recipe alone do not
settle that question. A model accepting 4096 input is also distinct from useful
retrieval at that depth; Hari's roadmap says deep 4K retrieval was not certified.

## Evidence needed before a 4K release

1. Inspect the original step-3450 checkpoint's saved model and training config,
   or the exact export command and source config. Record `seq_len`,
   `max_position_embeddings`, `rope_theta`, parent identity, and checkpoint hash.
2. Confirm that the public HF weights really came from that exact checkpoint.
   Preserve a revision and per-tensor conversion check; do not overwrite the
   current 1024-configured release.
3. If the original checkpoint and export provenance support 4096, publish a
   separate 4K candidate with accurate config. Run PyTorch and MLX parity at
   short, 2K, and near-4K lengths: logits, greedy token IDs, and expert sets.
   Keep router computation FP32 and weight loading strict.
4. Test shallow and deep retrieval separately, alongside memory and tokens per
   second on the target Mac. Report failures as limitations, even if sequence
   acceptance and numerical parity pass.

If the original checkpoint config says 1024, stop the 4K release path until an
actual 4K SFT checkpoint is identified. Do not relabel the current HF config or
raise the CLI cap to make a test pass. No training, fine-tuning, quantization,
or Modal work is authorized by this audit.

## Bounded coding handoff

An implementation agent may build a separate 4K validation path after the
checkpoint provenance gate is resolved. It must use an immutable model revision,
leave the 1024 MLX v1 default unchanged, save raw parity and retrieval artifacts,
and fail closed on missing or unexpected weights. A DeepSeek coding session can
take this task using this audit as its input; it cannot supply missing checkpoint
provenance by inference.
