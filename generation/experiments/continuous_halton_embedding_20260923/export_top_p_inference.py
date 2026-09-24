"""Export the exact top-p EMA components used by TopPFrozenAssets.

This does not alter the source training checkpoint. Tensor values and dtypes are
preserved; optimizer, discriminator, and unused EMA modules are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


SOURCE_SHA256 = "61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819"
PREFIXES = ("latent_decoder.", "llamagen_vq.")
EXPECTED_STEP = 247482
EXPECTED_TENSORS = 661
EXPECTED_NUMEL = 377_198_731


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_sha = sha256(args.source)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"unexpected source SHA-256: {source_sha}")
    source = torch.load(args.source, map_location="cpu", mmap=True, weights_only=True)
    if int(source.get("step", -1)) != EXPECTED_STEP:
        raise ValueError("wrong source step")
    ema = source["model_ema"]
    selected = {key: value.detach().cpu() for key, value in ema.items()
                if key.startswith(PREFIXES)}
    if len(selected) != EXPECTED_TENSORS or sum(v.numel() for v in selected.values()) != EXPECTED_NUMEL:
        raise ValueError("unexpected inference tensor ledger")
    output = dict(
        format="top_p_ema_inference_v1",
        source_sha256=source_sha,
        step=EXPECTED_STEP,
        model_ema=selected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(output, temporary)
    restored = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    restored_ema = restored["model_ema"]
    if (restored["source_sha256"] != source_sha or restored["step"] != EXPECTED_STEP
            or restored_ema.keys() != selected.keys()
            or any(not torch.equal(value, restored_ema[key]) for key, value in selected.items())):
        raise RuntimeError("export round-trip changed inference tensors")
    temporary.rename(args.output)
    print(f"output={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={sha256(args.output)}")
    print(f"tensors={len(selected)} parameters={EXPECTED_NUMEL}")


if __name__ == "__main__":
    main()
