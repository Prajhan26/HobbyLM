# M1 Max 4K MLX research-run review — 2026-10-02

**Decision: no fully validated 4K release from this run.** The frozen run completed, but the proposed behavior and retrieval gates were missed. The numerical gates in `MLX_4K_RELEASE_CRITERIA_DRAFT.md` were not formally approved; they are used here as prewritten reference points, not retroactive certification rules. The user requested a fully validated release, not an experimental preview.

## Evidence and scope

- Supplied archive: `artifacts-2026-10-02.zip`, SHA-256 `6668cb231d16ec3a238ddedbc911daa13bff6f976afb222e3f67063f738abd88`; 77 files under `artifacts/` (excluding four directory entries). Archive integrity check passed. The archive contains reports and small parity fixtures, not a distributable model-weight copy.
- Target report platform: `macOS-26.2-arm64-arm-64bit-Mach-O` on the friend's M1 Max. MLX 0.32.3 / Python 3.14.2 for behavior; PyTorch 2.14.1 / Python 3.13.12 for the CPU reference. Hardware identification and total RAM are not independently embedded in these reports.
- Behavior pinned HF revision `ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5`, FP32 Safetensors SHA-256 `d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89`, prompt fixture SHA-256 `8df294ad9e233888686b195bac574511253fba28debc8307c0cf4f5b60e34342`.
- The parity/retrieval candidate manifest identifies older immutable HF revision `f4b8557098c01fd42195f32feee3a3275dfbfbe3` with local metadata changed from 1024 to 4096, plain RoPE theta 10000, and a symlink to unchanged FP32 weights. The symlink target and original training-checkpoint identity cannot be reverified from this report-only archive. The behavior runner separately verified the later pinned weight hash and loaded strictly.

## M1 Max results

| Check | Result | Interpretation |
| --- | --- | --- |
| 30-prompt behavior: calling | MLX 6/10, PyTorch 6/10 | Below proposed 8/10 |
| Instruction following | MLX 3/10, PyTorch 3/10 | Below proposed 8/10 |
| Repetition/continuation | MLX 9/10, PyTorch 9/10 | Meets proposed 9/10 |
| Backend behavior comparison | 26/30 exact generated-ID sequences; no additional MLX rubric failures | Does not make the weak model behavior acceptable |
| Strict final-position parity | 4/12 cases; 11/12 next-token IDs match | Eight cases have expert-set differences; one changes next token (`archive-512`, PyTorch 383, MLX 921) |
| Frozen retrieval | 16/18 total; 1024 6/6, 2048 6/6, 3900 4/6 | Misses proposed 5/6 at each length and 1/2 at every 3900 position |
| 3900-token early retrieval | 0/2; `BIRCH-741` became `I`, `ORBIT-206` became `1` | Both failures at 10% needle position |
| Runtime | Behavior load 1.07 s; 30-case suite 5.31 s; 426 generated tokens / 5.30 s case-generation time = 80.4 aggregate tok/s | Aggregate across short cases, not sustained single-chat throughput |
| Peak MLX memory | 4.22 GiB behavior, 5.82 GiB parity, 7.23 GiB retrieval | Largest observed peak is 7.23 GiB |

All 12 parity cases and all 18 retrieval cases produced reports. The parity runner's nonzero exit came from strict mismatches, not execution errors. The 30 behavior cases completed on both backends. Independent re-scoring of the frozen raw reports reproduced the original category counts and the 26/30 exact sequence comparison.

## What this does and does not establish

The 4K-configured model loads and executes on Apple Silicon, and many outputs match PyTorch. This is evidence of a working 4K **candidate**, not a fully validated 4K release. The equal behavior-category totals on PyTorch and MLX suggest that most measured calling/instruction weakness is not an MLX-only regression, but four generated sequences differ and the suite is small. Exact router-set failures remain visible; the reports alone do not classify every one as an exact tie, numerical near-tie, or unexplained drift. No architecture, router-precision, training, quantization, or website change was made to force a pass.

## Next decision

1. Keep the public 4K claim on hold while a fully validated release is required. Do not relabel this run by lowering thresholds after seeing its results.
2. Diagnose the two early 3900-token retrieval failures against PyTorch on identical prompts, and classify the eight routing mismatches using same-token-sequence score margins. Verify the candidate's actual weight target if full provenance is needed.
3. Agree on capability and release criteria **before** a new frozen suite. If a legitimate implementation or evaluation bug is found, fix only that cause and rerun the complete suite on the M1 Max. If the model itself cannot meet the desired quality bar, a new model/capability scope is needed; this run cannot be certified retroactively.

## Follow-up diagnosis (not a rerun of the release suite)

On the same six 3900-token frozen prompts, a PyTorch CPU replay on the M4 used prompt hashes identical to the M1 MLX reports. It also scored 4/6: the two early-position answers were `I` and `1`, and the four middle/late positions returned the correct codes. This is strong evidence that those particular retrieval misses are not MLX-only, though it does not establish retrieval quality beyond this narrow grid.

An M4 FP32 boundary capture reproduced the same eight *indices of mismatched MoE layers* as the M1 reports. Across its 16 mismatched case/layer pairs, six had exact FP32 boundary ties in both backends and ten had nonzero margins with numerical reordering. `code-3900`'s PyTorch expert identities differed from the frozen M1 fixture, so those M4 counts must **not** be presented as definitive M1 tie classifications. The diagnostic needs to run on the target M1 Max.

On the M1 Max, after updating `feat/mlx-4k-behavior-eval`, run from the repository root (use the existing Python 3.13 reference environment):

```bash
.venv-hf-reference-py313/bin/python scripts/diagnose_4k_routing_boundaries.py pytorch \
  --model artifacts/hobbylm-1b-sft-3450-4k-candidate \
  --reference-dir artifacts/mlx-4k-research-parity \
  --output artifacts/mlx-4k-routing-diagnostic
PYTHONPATH="$PWD" .venv-mlx-release/bin/python scripts/diagnose_4k_routing_boundaries.py mlx \
  --model artifacts/hobbylm-1b-sft-3450-4k-candidate \
  --reference-dir artifacts/mlx-4k-research-parity \
  --output artifacts/mlx-4k-routing-diagnostic
.venv-mlx-release/bin/python scripts/diagnose_4k_routing_boundaries.py analyze \
  --model artifacts/hobbylm-1b-sft-3450-4k-candidate \
  --reference-dir artifacts/mlx-4k-research-parity \
  --output artifacts/mlx-4k-routing-diagnostic
```

Send `artifacts/mlx-4k-routing-diagnostic/routing-boundary-report.json` and the two environment JSON files. The script verifies the FP32 weight hash and original prompt hashes. Exact token and expert-set mismatches remain failures; this diagnostic does not change any release criterion.
