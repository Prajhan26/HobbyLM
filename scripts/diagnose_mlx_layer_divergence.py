"""Research-only layer-by-layer PyTorch/MLX divergence diagnostic.

Targets the frozen ``archive-512`` parity failure (PyTorch next token 383, MLX
921, one MoE expert set differs). It never changes model code, weights, the
FP32 router, or any parity criterion. It only records intermediates.

Stages (each writes to --work, default ``artifacts/archive512-divergence``):

  pytorch  [--dtype float32|float64]  capture every intermediate in PyTorch.
           float64 is a *diagnostic oracle only*: same weights upcast exactly,
           with the router also evaluated in float64 by a script-local forward.
  mlx      free-run capture, per-op teacher-forced runs on PyTorch's own
           inputs, and a forced-routing run that feeds PyTorch's expert indices.
  analyze  compare everything and write a JSON summary (tracked location).

Run order is in docs/MLX_4K_CHECKPOINT_AUDIT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mlx_4k_candidate import prompt_ids

REPO = "harims95/hobbylm-1b-broad-sft-3450-hf"
REVISION = "f4b8557098c01fd42195f32feee3a3275dfbfbe3"
PROMPT_SHA256 = "d3c1d267d2b5ad6d7c4406dc7b6f479d1ac3af374a524dd62801f409ce3730bb"
WORK = Path("artifacts/archive512-divergence")
SUMMARY = Path("eval/mlx_archive512_divergence/summary.json")
OPS = ["attn_norm", "attn_out", "mid", "ffn_in", "router_logits", "ffn_out", "out"]


def model_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, revision=REVISION))


def frozen_prompt() -> np.ndarray:
    ids = prompt_ids(512, "archive")
    digest = hashlib.sha256(ids.tobytes()).hexdigest()
    if digest != PROMPT_SHA256:
        raise SystemExit(f"archive-512 prompt changed: {digest}")
    return ids


# ----------------------------------------------------------------- PyTorch
def capture_pytorch(dtype: str, work: Path) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    ids = frozen_prompt()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir()), trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    cfg = model.config
    assert cfg.gating == "sigmoid" and cfg.balancing == "aux_free"
    assert not cfg.norm_topk_prob and cfg.routed_scaling_factor == 1.0
    if dtype == "float64":
        model = model.double()
        for layer in model.model.layers:
            if layer.is_moe:
                gate = layer.mlp.gate

                def forward64(hidden_states, routing_mask=None, gate=gate):
                    logits = F.linear(hidden_states.double(), gate.weight.double())
                    scores = torch.sigmoid(logits)
                    selection = scores + gate.expert_bias.double()
                    indices = torch.topk(selection, gate.top_k, dim=-1).indices
                    gate.last_topi = indices
                    gate.last_logits = logits
                    weights = torch.gather(scores, -1, indices)
                    return logits, weights, indices, hidden_states.new_zeros(())

                gate.forward = forward64
    cap: dict[str, np.ndarray] = {}

    def store(name, tensor):
        array = tensor.detach()[0].cpu().numpy()
        cap[name] = array.astype(np.float64 if dtype == "float64" else np.float32)

    inner = model.model
    inner.embed_tokens.register_forward_hook(lambda m, a, o: store("embed", o))
    inner.norm.register_forward_hook(lambda m, a, o: store("final_norm", o))
    for index, layer in enumerate(inner.layers):
        tag = f"{index:02d}"
        layer.register_forward_hook(lambda m, a, o, t=tag: store(f"{t}__out", o[0]))
        layer.input_layernorm.register_forward_hook(lambda m, a, o, t=tag: store(f"{t}__attn_norm", o))
        layer.self_attn.register_forward_hook(lambda m, a, o, t=tag: store(f"{t}__attn_out", o[0]))
        layer.post_attention_layernorm.register_forward_hook(lambda m, a, o, t=tag: store(f"{t}__ffn_in", o))
        layer.mlp.register_forward_hook(
            lambda m, a, o, t=tag: store(f"{t}__ffn_out", o[0] if isinstance(o, tuple) else o)
        )
        layer.register_forward_pre_hook(lambda m, a, t=tag: store(f"{t}__in", a[0]))
    with torch.inference_mode():
        logits = model(input_ids=torch.from_numpy(ids.astype(np.int64))[None], use_cache=False).logits[0]
    cap["logits_last"] = logits[-1].cpu().numpy().astype(np.float64 if dtype == "float64" else np.float32)
    for index, layer in enumerate(inner.layers):
        tag = f"{index:02d}"
        cap[f"{tag}__mid"] = cap[f"{tag}__in"] + cap[f"{tag}__attn_out"]
        if layer.is_moe:
            cap[f"{tag}__router_logits"] = layer.mlp.gate.last_logits.cpu().numpy()
            cap[f"{tag}__topi"] = layer.mlp.gate.last_topi.cpu().numpy().astype(np.int32)
            cap[f"{tag}__bias"] = layer.mlp.gate.expert_bias.cpu().numpy().astype(np.float32)
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / f"pytorch-{dtype}.npz", **cap)
    print(json.dumps({"stage": "pytorch", "dtype": dtype, "next_token": int(np.argmax(cap["logits_last"][:50257]))}))


# --------------------------------------------------------------------- MLX
def capture_mlx(work: Path) -> None:
    import mlx.core as mx
    from mlx_hobbylm.weights import load_local

    ids = frozen_prompt()
    model, cfg = load_local(model_dir())
    assert cfg.gating == "sigmoid" and not cfg.norm_topk_prob and cfg.routed_scaling_factor == 1.0
    x_ids = mx.array(ids[None, :], dtype=mx.int32)
    pt = np.load(work / "pytorch-float32.npz")
    out: dict[str, np.ndarray] = {}

    def np_(a):
        mx.eval(a)
        return np.asarray(a.astype(mx.float32))[0]

    def forced_moe(ffn, f, indices):
        scores = mx.sigmoid(ffn.gate(f.astype(mx.float32)))
        weights = mx.take_along_axis(scores, indices, axis=-1)
        routed = ffn.experts(f, indices)
        result = (routed * weights.astype(routed.dtype)[..., None]).sum(axis=-2)
        return result + ffn.shared(f) if cfg.n_shared else result

    # 1. Free run replicated op by op; must equal the library forward exactly.
    reference_logits = model(x_ids)[0, -1].astype(mx.float32)
    mx.eval(reference_logits)
    x = model.embed(x_ids)
    out["embed"] = np_(x)
    for index, block in enumerate(model.blocks):
        tag = f"{index:02d}"
        out[f"free__{tag}__in"] = np_(x)
        n = block.attn_norm(x)
        a = block.attn(n)
        mid = x + a
        f = block.ffn_norm(mid)
        o = block.ffn(f)
        x = mid + o
        out[f"free__{tag}__attn_norm"], out[f"free__{tag}__attn_out"] = np_(n), np_(a)
        out[f"free__{tag}__mid"], out[f"free__{tag}__ffn_in"] = np_(mid), np_(f)
        out[f"free__{tag}__ffn_out"], out[f"free__{tag}__out"] = np_(o), np_(x)
        if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None:
            out[f"free__{tag}__router_logits"] = np_(block.ffn.gate(f.astype(mx.float32)))
            mx.eval(block.ffn.last_topi)
            out[f"free__{tag}__topi"] = np.asarray(block.ffn.last_topi)[0].astype(np.int32)
    final = model.final_norm(x)
    logits = model.embed.as_linear(final) if cfg.tie_embeddings else model.lm_head(final)
    logits_last = logits[0, -1].astype(mx.float32)
    mx.eval(logits_last)
    out["free__logits_last"] = np.asarray(logits_last)
    out["replication_max_abs_vs_library"] = np.abs(out["free__logits_last"] - np.asarray(reference_logits)).max()

    # 2. Per-op teacher forcing: every op receives PyTorch's own float32 input.
    for index, block in enumerate(model.blocks):
        tag = f"{index:02d}"
        g = lambda name: mx.array(pt[f"{tag}__{name}"][None])  # noqa: E731
        out[f"tf__{tag}__attn_norm"] = np_(block.attn_norm(g("in")))
        out[f"tf__{tag}__attn_out"] = np_(block.attn(g("attn_norm")))
        out[f"tf__{tag}__ffn_in"] = np_(block.ffn_norm(g("mid")))
        o = block.ffn(g("ffn_in"))
        out[f"tf__{tag}__ffn_out"] = np_(o)
        out[f"tf__{tag}__out"] = np_(g("mid") + o)
        if hasattr(block.ffn, "last_topi") and block.ffn.last_topi is not None:
            out[f"tf__{tag}__router_logits"] = np_(block.ffn.gate(g("ffn_in").astype(mx.float32)))
            mx.eval(block.ffn.last_topi)
            out[f"tf__{tag}__topi"] = np.asarray(block.ffn.last_topi)[0].astype(np.int32)

    # 3. Forced routing: feed PyTorch's expert indices into the MLX model.
    def run_forced(layers: set[int]) -> np.ndarray:
        x = model.embed(x_ids)
        for index, block in enumerate(model.blocks):
            mid = x + block.attn(block.attn_norm(x))
            f = block.ffn_norm(mid)
            tag = f"{index:02d}"
            if index in layers and f"{tag}__topi" in pt:
                o = forced_moe(block.ffn, f, mx.array(pt[f"{tag}__topi"][None].astype(np.int32)))
            else:
                o = block.ffn(f)
            x = mid + o
        final = model.final_norm(x)
        lg = model.embed.as_linear(final) if cfg.tie_embeddings else model.lm_head(final)
        lg = lg[0, -1].astype(mx.float32)
        mx.eval(lg)
        return np.asarray(lg)

    # Self-test: forcing MLX's own indices must reproduce the library MoE output.
    f18 = model.blocks[18].ffn_norm(mx.array(pt["18__mid"][None]))
    own = mx.array(out["tf__18__topi"][None].astype(np.int32))
    library = np_(model.blocks[18].ffn(f18))
    out["forced_selftest_max_abs"] = np.abs(np_(forced_moe(model.blocks[18].ffn, f18, own)) - library).max()
    moe_layers = {i for i, b in enumerate(model.blocks) if hasattr(b.ffn, "last_topi")}
    out["forced_all_logits"] = run_forced(moe_layers)
    out["forced_layer18_logits"] = run_forced({18})
    np.savez(work / "mlx-float32.npz", **out)
    print(json.dumps({
        "stage": "mlx",
        "replication_max_abs_vs_library": float(out["replication_max_abs_vs_library"]),
        "free_next_token": int(np.argmax(out["free__logits_last"][:50257])),
    }))


# ----------------------------------------------------------------- analysis
def rel(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = a - b
    return {
        "max_abs": float(np.abs(d).max()),
        "rel_rms": float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-30)),
        "last_pos_rel_rms": float(np.linalg.norm(d[-1]) / max(np.linalg.norm(b[-1]), 1e-30)),
    }


def first_over(rows: list[dict], key: str, threshold: float):
    for row in rows:
        if row[key] > threshold:
            return {"layer": row["layer"], "op": row["op"], key: row[key]}
    return None


def selection(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = (1.0 / (1.0 + np.exp(-logits.astype(np.float32)))).astype(np.float32)
    return scores + bias.astype(np.float32)


def margins(sel: np.ndarray) -> np.ndarray:
    ordered = -np.sort(-sel, axis=-1)
    return (ordered[..., 7] - ordered[..., 8]).astype(np.float64)


def set_diff_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([set(map(int, x)) != set(map(int, y)) for x, y in zip(a, b)])


def analyze(work: Path, summary_path: Path) -> None:
    pt = np.load(work / "pytorch-float32.npz")
    mx_ = np.load(work / "mlx-float32.npz")
    p64 = np.load(work / "pytorch-float64.npz") if (work / "pytorch-float64.npz").exists() else None
    layers = sorted({int(k[:2]) for k in pt.files if k[:2].isdigit()})
    moe = [l for l in layers if f"{l:02d}__topi" in pt.files]
    summary: dict = {
        "status": "research_only_not_release_validation",
        "model": {"repo": REPO, "revision": REVISION},
        "prompt": {"case": "archive-512", "sha256": PROMPT_SHA256},
        "platform": platform.platform(),
        "mlx_replication_max_abs_vs_library_forward": float(mx_["replication_max_abs_vs_library"]),
    }

    def tok(logits):
        return int(np.argmax(logits[:50257]))

    final = {"pytorch_fp32": pt["logits_last"], "mlx_fp32": mx_["free__logits_last"]}
    if p64 is not None:
        final["pytorch_fp64_oracle"] = p64["logits_last"]
    final["mlx_forced_pytorch_routes_all_layers"] = mx_["forced_all_logits"]
    final["mlx_forced_pytorch_routes_layer18_only"] = mx_["forced_layer18_logits"]
    summary["final_logits"] = {
        name: {
            "next_token": tok(v),
            "logit_383": float(v[383]), "logit_921": float(v[921]),
            "gap_383_minus_921": float(v[383] - v[921]),
            "max_abs_vs_pytorch_fp32": float(np.abs(v - pt["logits_last"]).max()),
            "mean_abs_vs_pytorch_fp32": float(np.abs(v - pt["logits_last"]).mean()),
        } for name, v in final.items()
    }

    def series(a_get, b_get, names):
        rows = []
        for l in layers:
            for op in names:
                key = f"{l:02d}__{op}"
                a, b = a_get(l, op), b_get(l, op)
                if a is None or b is None:
                    continue
                rows.append({"layer": l, "op": op, **rel(a, b)})
        return rows

    def get_pt(l, op):
        k = f"{l:02d}__{op}"
        return pt[k] if k in pt.files else None

    def get_mx(prefix):
        return lambda l, op: mx_[f"{prefix}__{l:02d}__{op}"] if f"{prefix}__{l:02d}__{op}" in mx_.files else None

    def get_64(l, op):
        k = f"{l:02d}__{op}"
        return p64[k] if p64 is not None and k in p64.files else None

    comparisons = {
        "free_run_mlx_vs_pytorch_fp32": series(get_mx("free"), get_pt, OPS),
        "teacher_forced_per_op_mlx_vs_pytorch_fp32": series(get_mx("tf"), get_pt, [o for o in OPS if o not in ("mid",)]),
    }
    if p64 is not None:
        comparisons["free_run_pytorch_fp32_vs_fp64"] = series(get_pt, get_64, OPS)
        comparisons["free_run_mlx_vs_pytorch_fp64"] = series(get_mx("free"), get_64, OPS)
    summary["divergence"] = {}
    for name, rows in comparisons.items():
        summary["divergence"][name] = {
            "first_rel_rms_over": {str(t): first_over(rows, "rel_rms", t) for t in (1e-7, 1e-6, 1e-5, 1e-4)},
            "rows": rows,
        }

    # Router analysis over all MoE layers and positions.
    router: dict = {"layers": []}
    near = {1e-5: 0, 1e-4: 0, 1e-3: 0}
    total = 0
    for l in moe:
        tag = f"{l:02d}"
        bias = pt[f"{tag}__bias"]
        sel_pt = selection(pt[f"{tag}__router_logits"], bias)
        sel_mx = selection(mx_[f"free__{tag}__router_logits"], bias)
        m_pt, m_mx = margins(sel_pt), margins(sel_mx)
        free_flip = set_diff_rows(pt[f"{tag}__topi"], mx_[f"free__{tag}__topi"])
        tf_flip = set_diff_rows(pt[f"{tag}__topi"], mx_[f"tf__{tag}__topi"])
        exact_ties = int((m_pt == 0).sum())
        total += m_pt.size
        for t in near:
            near[t] += int((m_pt < t).sum())
        # Classify teacher-forced flips: identical router input, so any set
        # difference is selection-op behaviour, not upstream numerical drift.
        classes = {"exact_fp32_tie_at_boundary": 0, "not_a_tie": 0,
                   "pytorch_kept_lower_index": 0, "pytorch_kept_higher_index": 0,
                   "mlx_kept_lower_index": 0, "mlx_kept_higher_index": 0}
        for i in np.where(tf_flip)[0]:
            ordered = np.sort(sel_pt[i])[::-1]
            boundary_value = ordered[7]
            if ordered[7] != ordered[8]:
                classes["not_a_tie"] += 1
                continue
            classes["exact_fp32_tie_at_boundary"] += 1
            tied = np.where(sel_pt[i] == boundary_value)[0]
            kept_pt = sorted(set(map(int, pt[f"{tag}__topi"][i])) & set(map(int, tied)))
            kept_mx = sorted(set(map(int, mx_[f"tf__{tag}__topi"][i])) & set(map(int, tied)))
            lower = sorted(map(int, tied))[: len(kept_pt)]
            higher = sorted(map(int, tied))[len(tied) - len(kept_pt):]
            classes["pytorch_kept_lower_index"] += int(kept_pt == lower)
            classes["pytorch_kept_higher_index"] += int(kept_pt == higher)
            classes["mlx_kept_lower_index"] += int(kept_mx == lower)
            classes["mlx_kept_higher_index"] += int(kept_mx == higher)
        entry = {
            "moe_index": l - 1, "transformer_layer": l,
            "teacher_forced_flip_classification": classes,
            "positions_with_set_difference_free_run": [int(i) for i in np.where(free_flip)[0]],
            "positions_with_set_difference_teacher_forced_router_input": [int(i) for i in np.where(tf_flip)[0]],
            "exact_fp32_selection_ties_pytorch": exact_ties,
            "router_logit_max_abs_free_run": float(np.abs(pt[f"{tag}__router_logits"] - mx_[f"free__{tag}__router_logits"]).max()),
            "router_logit_max_abs_teacher_forced": float(np.abs(pt[f"{tag}__router_logits"] - mx_[f"tf__{tag}__router_logits"]).max()),
            "flip_margins_pytorch_fp32": {int(i): float(m_pt[i]) for i in np.where(free_flip)[0]},
            "flip_margins_mlx_fp32": {int(i): float(m_mx[i]) for i in np.where(free_flip)[0]},
        }
        router["layers"].append(entry)
    router["selection_margin_fraction_below"] = {str(k): v / total for k, v in near.items()}
    router["token_layer_pairs"] = total
    router["fp32_ulp_at_selection_scale"] = float(np.spacing(np.float32(99.6)))
    summary["router"] = router

    # Drill-down: last position of transformer layer 18 (MoE index 17).
    L, pos = 18, 511
    tag = f"{L:02d}"
    bias = pt[f"{tag}__bias"].astype(np.float32)
    drill = {"moe_index": L - 1, "transformer_layer": L, "position": pos, "bias_min_max": [float(bias.min()), float(bias.max())]}
    views = {"pytorch_fp32": (pt[f"{tag}__router_logits"][pos], pt[f"{tag}__topi"][pos]),
             "mlx_fp32_free_run": (mx_[f"free__{tag}__router_logits"][pos], mx_[f"free__{tag}__topi"][pos]),
             "mlx_fp32_pytorch_input": (mx_[f"tf__{tag}__router_logits"][pos], mx_[f"tf__{tag}__topi"][pos])}
    if p64 is not None:
        views["pytorch_fp64_oracle"] = (p64[f"{tag}__router_logits"][pos], p64[f"{tag}__topi"][pos])
    boundary = sorted(set(map(int, pt[f"{tag}__topi"][pos])) ^ set(map(int, mx_[f"free__{tag}__topi"][pos])))
    ranks = np.argsort(-selection(pt[f"{tag}__router_logits"][pos], bias))[6:12]
    watch = sorted(set(map(int, ranks)) | set(boundary))
    drill["watched_experts"] = watch
    for name, (logits, topi) in views.items():
        if name == "pytorch_fp64_oracle":
            score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
            sel = score + bias.astype(np.float64)
            order = np.argsort(-sel)
            gap = float(sel[order[7]] - sel[order[8]])
        else:
            sel = selection(logits, bias)
            order = np.argsort(-sel, kind="stable")
            gap = float(sel[order[7]] - sel[order[8]])
        drill[name] = {
            "experts": sorted(map(int, topi)),
            "router_logits": {str(e): float(logits[e]) for e in watch},
            "selection_scores": {str(e): float(sel[e]) for e in watch},
            "rank8_expert": int(order[7]), "rank9_expert": int(order[8]),
            "rank8_minus_rank9_selection_margin": gap,
        }
    summary["layer18_last_position"] = drill
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pytorch", "mlx", "analyze"])
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    if args.stage == "pytorch":
        capture_pytorch(args.dtype, args.work)
    elif args.stage == "mlx":
        capture_mlx(args.work)
    else:
        analyze(args.work, args.summary)


if __name__ == "__main__":
    main()
