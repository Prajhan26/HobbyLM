# HobbyLM-1B 4K MLX release criteria — draft for approval

Status: **proposed, not approved**. Freeze this document and the prompt fixtures
before the M1 Max release run. Do not revise thresholds after seeing results
and then call the same run a pass. This document does not authorize publishing
or changing the public Hugging Face model.

## What is being evaluated

- Candidate: the 4096-configured HF revision
  `ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5` of the broad-SFT
  step-3450 model, or an explicitly equivalent local copy. Its published
  Safetensors hash matches the prior 1024-configured revision
  `f4b8557098c01fd42195f32feee3a3275dfbfbe3`. Keep the prior 1K
  validation tied to that earlier immutable revision. Record exact config,
  revision, weight hash, code commit, Python/MLX/PyTorch versions, and Mac
  model/RAM.
- Use GPT-2 tokenization, literal `SYSTEM:`, `USER:`, `ASSISTANT:` formatting,
  greedy decoding, and fixed token budgets. Preserve FP32 router calculations,
  sigmoid scores, balancing bias, top-8 selection, and original-score expert
  weighting. Load weights strictly; unresolved missing or unexpected weights
  are a hard failure.
- A frozen 30-prompt behavior set should contain 10 calling/structured-action
  cases, 10 instruction-following cases, and 10 repetition/continuation cases.
  Each case needs a predetermined machine-checkable or blinded human rubric.
  Do not score a capability that the SFT was never intended to support as a
  hidden requirement; document such an exclusion before the run.
- Run the existing 18-case frozen access-code retrieval grid at 1024, 2048,
  and 3900 prompt tokens. Treat it as a narrow synthetic recall test, not a
  complete long-context benchmark. Include at least one 4096-token acceptance
  and generation smoke test, but note that 4096 prompt tokens plus generated
  tokens extends beyond a strict 4096 total context. A 4096-token *total*
  prompt-plus-generation check needs a shorter prefill.

## Proposed behavioral gates (approval needed)

| Gate | Proposed pass threshold | Why it is separate |
| --- | --- | --- |
| Calling/structured action | At least 8/10 pass, with no unsafe or wrong-tool action | Correct action matters more than matching PyTorch wording |
| Instruction following | At least 8/10 pass | Measures whether the Mac backend obeys the task |
| Repetition/continuation | At least 9/10 pass; no runaway repeated span on any case | Catches a user-visible failure mode |
| Mac versus PyTorch behavior | MLX has no more than one additional failure in any category on the same frozen prompts | Prevents accepting a large backend quality regression, while not requiring identical tokens |
| Frozen retrieval | At least 5/6 exact codes at **each** length, and at least 1/2 at each 3900-token needle position | Prevents an aggregate score from hiding early-position collapse |
| Weights and runtime | Strict load, no crash/OOM, all cases complete, memory and timing recorded | Basic reproducibility and usability |

These numeric thresholds are **proposals, not facts about expected model
quality**. The user and Hari should approve or change them before the target
Mac run. A short preview may use different thresholds, but must be labelled
experimental rather than retrospectively called a validated 4K release.

## Parity and routing reporting, not a hidden pass conversion

- Run the frozen 12-case PyTorch/MLX parity grid and report exact next-token
  IDs, logit errors, 19-layer expert-set agreement, and any failures. The
  current M4 result is 4/12 strict passes; M1 Max may differ. **Do not turn
  this number into 12/12 using a tie-aware metric.**
- Classify each route difference as exact FP32 boundary tie, numerical
  near-tie, or unexplained, using scores from the *same token sequence*.
  Compare each backend's routing on identical prompt tokens first. If
  generated tokens diverge, stop comparing later routes as if they had the
  same input; instead report the different sequences and optionally replay
  each backend on a fixed teacher-forced sequence.
- A backend-dependent tie may be accepted only if the behavioral gates pass
  and its effect is disclosed. Unexplained large logit drift, a weight
  mapping error, altered routing semantics, or an unexamined user-visible
  regression requires investigation before a validated release.

## Release decision

1. **Validated 4K MLX:** all approved behavioral, retrieval, load, and
   completion gates pass on M1 Max; exact parity exceptions and performance
   are published as limitations. Do not describe it as PyTorch-identical.
2. **Experimental 4K preview:** some gates fail or remain unapproved, but the
   candidate is still made available with conspicuous limitations. It may say
   "tested on M1 Max" if that run happened, but not "validated 4K". This
   requires a separate distribution decision.
3. **Hold:** crashes, incomplete reports, weight mismatch, material quality
   regression, or unresolved routing/architecture concern.

Hari's 4096 metadata correction is now visible on the Hub at the immutable
revision above. It is a metadata change, **not** a completed MLX release
validation. The pinned 1024 revision must not be silently re-labelled.
Preserve all M1 Max reports and commands.
