# HobbyLM-1B 4K MLX behavior eval — rubric reference (research only)

Status: **frozen, research-only preview**, not an approved release gate. See
`docs/MLX_4K_RELEASE_CRITERIA_DRAFT.md` for the proposed, unapproved
numeric thresholds this suite reports against. This file documents what each
rubric `type` in `eval/mlx_4k_behavior_prompts.json` checks. Freeze the
prompts and this document together; do not add, remove, or loosen a check
after seeing a run's results.

## Case set

30 frozen cases in `eval/mlx_4k_behavior_prompts.json`: `call-01..10`
(calling/structured-action), `instr-01..10` (instruction-following),
`rep-01..10` (repetition/continuation). Each case is fully machine-checkable;
no blinded human rubric is used in this pass (see `exclusions` in the prompt
file for what is intentionally out of scope and why).

## Rubric types (implemented in `scripts/score_mlx_4k_behavior_eval.py`)

- `tool_call` — raw output must parse as exactly one call in either the JSON
  training format (`[{"name":"get_weather","arguments":{"location":"Boston"}}]`)
  or the Python-like format observed in this model's output
  (`[get_weather(location='Boston')]`). The Python-like parser uses a restricted
  syntax tree and never executes the text. Pass requires the expected tool
  name and every required argument with a non-empty value (a `null` rubric
  value means any non-empty value; a non-null value requires a case-insensitive
  substring match). The observed format is reported, not silently converted.
- `word_count` — whitespace-delimited word count of the trimmed output must
  fall within `[min_words, max_words]`.
- `line_count` — non-blank line count must fall within `[min_lines, max_lines]`.
- `sentence_count` — count of `.`/`!`/`?`-terminated segments must fall
  within `[min_sentences, max_sentences]`.
- `json_only` — the entire trimmed output must parse as JSON (nothing else).
- `all_uppercase` — output must contain at least one letter and no lowercase
  letters.
- `starts_with` — trimmed output, lowercased, must start with `prefix`.
- `forbidden_words` — none of `forbidden` may appear (case-insensitive,
  whole-word match).
- `contains_all` — every string in `required` must appear in the output
  (case-insensitive substring match).
- `repetition` — detects the user-visible failure mode, not task
  correctness. EOS is excluded from the token statistics. Output shorter
  than `min_tokens_checked` is marked **not evaluable and fails**, rather
  than being counted as a clean repetition pass. Otherwise it fails if either:
  1. `repeat_fraction` (the most common generated token's count divided by
     total generated tokens, once at least `min_tokens_checked` tokens were
     generated) exceeds `max_repeat_fraction`, or
  2. any contiguous 3-token n-gram repeats immediately back-to-back more
     than `max_consecutive_ngram_repeats` times (a direct "runaway repeated
     span" detector).

## What this suite does not score

- Factual correctness or reasoning quality beyond the literal, stated
  constraint in each instruction-following prompt.
- Multi-turn or multi-call tool use (the SFT tool-calling data is
  single-shot, single-call; see `exclusions` in the prompt file).
- PyTorch-vs-MLX same-prompt behavioral comparison. The "Mac versus PyTorch
  behavior" gate in the release-criteria draft needs a second, PyTorch-backed
  run of this same frozen suite on identical prompts; this repo change adds
  only the MLX-side runner and scorer.
