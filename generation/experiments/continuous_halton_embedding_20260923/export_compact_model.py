"""Export the inference-only 321M continuous-Halton generation bundle.

The source checkpoints are never modified.  The only model parameters omitted
are four TiTok wrapper interfaces that ``forward_1d`` never reads.  Optimizer,
scheduler, scaler, and training-only state are not copied into the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bert2d.paths import sha256
from h20.assets import official_model
from experiments.continuous_halton_embedding_20260923.continuous_halton import ContinuousHalton
from experiments.continuous_halton_embedding_20260923.embedding_adapter import EmbeddingAdapter


FORMAT = "mot_continuous_halton_compact_model_v1"
UNUSED_ONE_D_MODULES = (
    "embedding_2d",
    "pos_embedding_2d",
    "budget_embedding",
    "output_2d",
)
EXPECTED = {
    "one_d": 177_202_432,
    "halton": 142_224_641,
    "adapter": 673_024,
    "scorer": 509_313,
}
EXPECTED_REMOVED = 25_382_656
EXPECTED_TOTAL = 320_609_410


def _tensor_dict(state: object, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError(f"{label} is not a non-empty tensor state dict")
    return state


def _equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        a.shape == right[key].shape
        and a.dtype == right[key].dtype
        and torch.equal(a, right[key])
        for key, a in left.items()
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--halton-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--halton-upstream-root",
        type=Path,
        help=(
            "local checkout of valeoai/Halton-MaskGIT; overrides the source "
            "checkpoint's machine-specific upstream_root only while building "
            "the module"
        ),
    )
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_name(args.output.name + ".incomplete")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if temporary.exists() or manifest_path.exists():
        raise FileExistsError(temporary if temporary.exists() else manifest_path)

    one_d, one_d_load_audit = official_model(args.assets_root)
    before = sum(parameter.numel() for parameter in one_d.parameters())
    removed = 0
    for name in UNUSED_ONE_D_MODULES:
        module = getattr(one_d, name, None)
        if module is None:
            raise RuntimeError(f"missing unused 1D interface: {name}")
        removed += sum(parameter.numel() for parameter in module.parameters())
        delattr(one_d, name)
    one_d_count = sum(parameter.numel() for parameter in one_d.parameters())
    if (before, removed, one_d_count) != (202_585_088, EXPECTED_REMOVED, EXPECTED["one_d"]):
        raise RuntimeError("unexpected TiTok 1D pruning ledger")

    halton_source = torch.load(
        args.halton_checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    halton_config = halton_source["config"]
    halton_upstream_root = (
        args.halton_upstream_root
        if args.halton_upstream_root is not None
        else Path(halton_config["upstream_root"])
    )
    halton = ContinuousHalton(halton_upstream_root)
    halton.load_state_dict(halton_source["model"], strict=True)
    if int(halton_source["step"]) != 20_000:
        raise RuntimeError("compact export requires the formal Halton step-20000 checkpoint")

    adapter_source = torch.load(
        args.adapter_checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    adapter = EmbeddingAdapter(**adapter_source["model_config"])
    adapter.load_state_dict(adapter_source["model"], strict=True)
    if int(adapter_source["step"]) != 20_000:
        raise RuntimeError("compact export requires the formal adapter step-20000 checkpoint")

    scorer_source = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=True)
    scorer_model = _tensor_dict(scorer_source["model"], "scorer model")

    counts = {
        "one_d": one_d_count,
        "halton": sum(parameter.numel() for parameter in halton.parameters()),
        "adapter": sum(parameter.numel() for parameter in adapter.parameters()),
        "scorer": sum(value.numel() for value in scorer_model.values()),
    }
    if counts != EXPECTED or sum(counts.values()) != EXPECTED_TOTAL:
        raise RuntimeError(f"unexpected compact parameter ledger: {counts}")

    models = {
        "one_d": _tensor_dict(one_d.state_dict(), "one_d model"),
        "halton": _tensor_dict(halton.state_dict(), "halton model"),
        "adapter": _tensor_dict(adapter.state_dict(), "adapter model"),
        "scorer": scorer_model,
    }
    sources = {
        "one_d_asset": str((args.assets_root / "weights/generator_titok_l32.bin").resolve()),
        "one_d_asset_sha256": sha256(args.assets_root / "weights/generator_titok_l32.bin"),
        "halton_checkpoint": str(args.halton_checkpoint.resolve()),
        "halton_checkpoint_sha256": sha256(args.halton_checkpoint),
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256(args.adapter_checkpoint),
        "scorer_checkpoint": str(args.scorer_checkpoint.resolve()),
        "scorer_checkpoint_sha256": sha256(args.scorer_checkpoint),
    }
    output = {
        "format": FORMAT,
        "parameter_counts": {**counts, "total": EXPECTED_TOTAL},
        "removed_one_d_modules": list(UNUSED_ONE_D_MODULES),
        "removed_parameters": removed,
        "one_d_load_audit": one_d_load_audit,
        "sources": sources,
        "configs": {
            "halton": halton_config,
            "adapter": adapter_source["train_config"],
            "adapter_model": adapter_source["model_config"],
            "scorer_args": scorer_source["args"],
        },
        "models": models,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, temporary)
    restored = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    if (
        restored.get("format") != FORMAT
        or restored.get("parameter_counts") != output["parameter_counts"]
        or restored.get("sources") != sources
        or set(restored.get("models", {})) != set(models)
        or any(not _equal_state(models[name], restored["models"][name]) for name in models)
    ):
        raise RuntimeError("compact checkpoint round-trip changed model tensors or metadata")
    temporary.rename(args.output)

    manifest = {
        "status": "complete",
        "format": FORMAT,
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": _file_sha256(args.output),
        "parameter_counts": output["parameter_counts"],
        "removed_one_d_modules": output["removed_one_d_modules"],
        "removed_parameters": removed,
        "sources": sources,
        "round_trip_exact": True,
        "contains_optimizer": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
