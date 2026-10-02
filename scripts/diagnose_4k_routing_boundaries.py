"""Research-only FP32 top-8 boundary capture on frozen M1 parity prompts.

Run ``pytorch`` in the reference environment, then ``mlx`` in the MLX
environment, then ``analyze`` in either. No routing or model code is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

WEIGHT_SHA256 = "d0899e0ac88c90a2c01ca481463bc0f96c6d50337f1b3e0dc56e1d3dc2999e89"


def verify_weights(model_dir: Path) -> None:
    digest = hashlib.sha256()
    with (model_dir / "model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise SystemExit("Candidate FP32 weight SHA-256 differs from frozen reference")


def selected_cases(reference_dir: Path, case_name: str | None = None) -> list[str]:
    names = []
    for path in sorted(reference_dir.glob("*.mlx.json")):
        row = json.loads(path.read_text())
        if not row["strict_parity_pass"] and (case_name is None or row["case"] == case_name):
            names.append(row["case"])
    if not names:
        raise SystemExit("No failed frozen parity cases found")
    return names


def fixture(reference_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(reference_dir / f"{name}.npz") as data:
        ids = data["prompt_ids"].copy()
        routes = data["routes"].copy()
    expected = json.loads((reference_dir / f"{name}.mlx.json").read_text())
    if hashlib.sha256(ids.tobytes()).hexdigest() != expected["prompt_sha256"]:
        raise SystemExit(f"Prompt hash mismatch: {name}")
    return ids, routes


def capture_pytorch(model_dir: Path, reference_dir: Path, output: Path, case_name: str | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    verify_weights(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    layers = [layer for layer in model.model.layers if getattr(layer, "is_moe", False)]
    if len(layers) != 19:
        raise SystemExit("Expected 19 MoE layers")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pytorch-env.json").write_text(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "weight_sha256": WEIGHT_SHA256,
    }, indent=2) + "\n")
    for name in selected_cases(reference_dir, case_name):
        ids, expected_routes = fixture(reference_dir, name)
        hidden: dict[int, np.ndarray] = {}
        handles = []
        for index, layer in enumerate(layers):
            def capture(_module, inputs, index=index):
                hidden[index] = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[-1].float().cpu().numpy().copy()
            handles.append(layer.mlp.gate.register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                model(input_ids=torch.from_numpy(ids.astype(np.int64))[None], use_cache=False)
                routes = np.stack([
                    layer.mlp.gate.last_topi.reshape(-1, 8)[-1].cpu().numpy()
                    for layer in layers
                ])
                selections = np.stack([
                    (torch.sigmoid(layer.mlp.gate.last_logits.reshape(-1, 64)[-1].float())
                     + layer.mlp.gate.expert_bias.float()).cpu().numpy()
                    for layer in layers
                ])
        finally:
            for handle in handles:
                handle.remove()
        matches_frozen_routes = np.array_equal(routes, expected_routes)
        np.savez_compressed(
            output / f"{name}.pytorch-routing.npz", routes=routes,
            selection=selections, hidden=np.stack([hidden[i] for i in range(19)]),
            matches_frozen_routes=np.array(matches_frozen_routes),
        )
        print(f"PyTorch captured {name}; M1 reference routes match: {matches_frozen_routes}", flush=True)


def capture_mlx(model_dir: Path, reference_dir: Path, output: Path, case_name: str | None) -> None:
    import mlx.core as mx
    from mlx_hobbylm.weights import load_local

    verify_weights(model_dir)
    model, cfg = load_local(model_dir)
    if cfg.max_position_embeddings != 4096 or cfg.top_k != 8:
        raise SystemExit("Expected 4K top-8 candidate")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mlx-env.json").write_text(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "mlx": mx.__version__, "weight_sha256": WEIGHT_SHA256,
    }, indent=2) + "\n")
    for name in selected_cases(reference_dir, case_name):
        ids, _ = fixture(reference_dir, name)
        x = model.embed(mx.array(ids[None, :], dtype=mx.int32))
        if cfg.scale_embeddings:
            x = x * (cfg.d_model**0.5)
        selections, routes, hidden = [], [], []
        for block in model.blocks:
            mid = x + block.attn(block.attn_norm(x))
            f = block.ffn_norm(mid)
            if hasattr(block.ffn, "expert_bias"):
                gate_input = f.astype(mx.float32)
                logits = block.ffn.gate(gate_input)
                selection = mx.sigmoid(logits) + block.ffn.expert_bias
                o = block.ffn(f)
                mx.eval(selection, block.ffn.last_topi)
                selections.append(np.asarray(selection)[0, -1].copy())
                routes.append(np.asarray(block.ffn.last_topi)[0, -1].copy())
                hidden.append(np.asarray(gate_input)[0, -1].copy())
            else:
                o = block.ffn(f)
            x = mid + o
        if len(routes) != 19:
            raise SystemExit("Expected 19 MoE layers")
        np.savez_compressed(
            output / f"{name}.mlx-routing.npz", routes=np.stack(routes),
            selection=np.stack(selections), hidden=np.stack(hidden),
        )
        print(f"MLX captured {name}", flush=True)


def boundary(selection: np.ndarray) -> dict:
    order = np.argsort(-selection.astype(np.float64))
    eighth, ninth = int(order[7]), int(order[8])
    margin = float(selection[eighth].astype(np.float64) - selection[ninth].astype(np.float64))
    ulp = float(np.spacing(np.float32(selection[eighth])))
    return {
        "rank8_expert": eighth, "rank9_expert": ninth,
        "margin": margin, "margin_in_fp32_ulps": margin / ulp if ulp > 0 else None,
        "fp32_ulp_at_boundary": ulp,
    }


def analyze(reference_dir: Path, output: Path, case_name: str | None) -> None:
    cases = []
    for name in selected_cases(reference_dir, case_name):
        with np.load(output / f"{name}.pytorch-routing.npz") as data:
            pt = {key: data[key].copy() for key in data.files}
        with np.load(output / f"{name}.mlx-routing.npz") as data:
            mlx = {key: data[key].copy() for key in data.files}
        frozen = json.loads((reference_dir / f"{name}.mlx.json").read_text())
        mismatch = [i for i in range(19) if set(pt["routes"][i]) != set(mlx["routes"][i])]
        layers = []
        for i in mismatch:
            p, m = boundary(pt["selection"][i]), boundary(mlx["selection"][i])
            if p["margin"] == 0 and m["margin"] == 0:
                label = "exact_fp32_ties_on_both_backends"
            elif p["margin"] == 0 or m["margin"] == 0:
                label = "exact_tie_on_one_backend_only"
            else:
                label = "numerical_reordering_not_an_exact_tie"
            pt_only = sorted(set(map(int, pt["routes"][i])) - set(map(int, mlx["routes"][i])))
            mlx_only = sorted(set(map(int, mlx["routes"][i])) - set(map(int, pt["routes"][i])))
            layers.append({
                "moe_index": i, "classification": label,
                "pytorch_boundary": p, "mlx_boundary": m,
                "pytorch_only_experts": pt_only, "mlx_only_experts": mlx_only,
                "pytorch_scores_for_flipped": {str(e): float(pt["selection"][i, e]) for e in pt_only + mlx_only},
                "mlx_scores_for_flipped": {str(e): float(mlx["selection"][i, e]) for e in pt_only + mlx_only},
                "router_input_max_abs_diff": float(np.max(np.abs(pt["hidden"][i] - mlx["hidden"][i]))),
            })
        cases.append({
            "case": name, "frozen_mismatched_moe_indices": frozen["mismatched_moe_indices"],
            "diagnostic_mismatched_moe_indices": mismatch,
            "pytorch_routes_match_frozen_m1_reference": bool(pt["matches_frozen_routes"]),
            "matches_frozen_m1_mismatch_pattern": mismatch == frozen["mismatched_moe_indices"],
            "layers": layers,
        })
    report = {
        "status": "research_only_no_tolerance_or_parity_relabeling",
        "note": "Classifications describe the captured top-8 FP32 boundary on identical prompt token sequences; they do not prove downstream behavioral acceptability.",
        "pytorch_environment": json.loads((output / "pytorch-env.json").read_text()),
        "mlx_environment": json.loads((output / "mlx-env.json").read_text()),
        "cases": cases,
    }
    (output / "routing-boundary-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "frozen_patterns_reproduced": sum(c["matches_frozen_m1_mismatch_pattern"] for c in cases), "layer_classifications": [layer["classification"] for case in cases for layer in case["layers"]]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pytorch", "mlx", "analyze"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="One frozen failed parity case for a smoke test")
    args = parser.parse_args()
    if args.stage == "pytorch":
        capture_pytorch(args.model, args.reference_dir, args.output, args.case)
    elif args.stage == "mlx":
        capture_mlx(args.model, args.reference_dir, args.output, args.case)
    else:
        analyze(args.reference_dir, args.output, args.case)


if __name__ == "__main__":
    main()
