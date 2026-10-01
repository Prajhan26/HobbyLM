"""Prepare an unpublished local 4K-configured candidate from pinned HF weights.

This changes only a copied config. The original release and its cached weights
are never modified. A 4K config is not evidence of 4K parity or useful recall.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL = "harims95/hobbylm-1b-broad-sft-3450-hf"
REVISION = "f4b8557098c01fd42195f32feee3a3275dfbfbe3"
DEFAULT_OUTPUT = Path("artifacts/hobbylm-1b-sft-3450-4k-candidate")
CODE_FILES = ("configuration_hobbylm.py", "modeling_hobbylm.py")


def prepare(output: Path) -> dict:
    source = Path(snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=["config.json", "generation_config.json", "model.safetensors", "*.py"],
    ))
    original = json.loads((source / "config.json").read_text())
    if original.get("max_position_embeddings") != 1024:
        raise ValueError("Pinned source no longer has the expected 1024 context")
    if original.get("torch_dtype") != "float32":
        raise ValueError("Pinned source is not declared FP32")
    for filename in (*CODE_FILES, "model.safetensors"):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)

    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing:
        raise FileExistsError(f"Candidate directory is not empty: {output}")

    candidate = dict(original)
    candidate["max_position_embeddings"] = 4096
    (output / "config.json").write_text(json.dumps(candidate, indent=2) + "\n")
    for filename in CODE_FILES:
        shutil.copy2(source / filename, output / filename)
    if (source / "generation_config.json").exists():
        shutil.copy2(source / "generation_config.json", output / "generation_config.json")
    (output / "model.safetensors").symlink_to((source / "model.safetensors").resolve())

    manifest = {
        "status": "research_candidate_only_not_published_or_certified",
        "source_model": MODEL,
        "source_revision": REVISION,
        "source_config_max_position_embeddings": 1024,
        "candidate_config_max_position_embeddings": 4096,
        "weight_handling": "symlink_to_unchanged_pinned_fp32_safetensors",
        "raw_training_checkpoint_weight_identity": "not_yet_verified",
        "known_4096_token_expert_set_parity": "fail_18_of_19_moe_layers_on_one_probe",
        "source_snapshot": str(source),
    }
    (output / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
