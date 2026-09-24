"""CPU-only equivalence check for the original and inference-only top-p EMA assets."""

from __future__ import annotations

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from bert2d.paths import ASSETS
from experiments.feature_to_token_20260916.assets_top_p import (
    TOP_P_INFERENCE_SHA256,
    TOP_P_SHA256,
    TopPFrozenAssets,
)


def same_state(left: torch.nn.Module, right: torch.nn.Module, label: str) -> tuple[int, int]:
    a, b = left.state_dict(), right.state_dict()
    if a.keys() != b.keys():
        raise RuntimeError(f"{label} keys differ")
    for key, value in a.items():
        if not torch.equal(value, b[key]):
            raise RuntimeError(f"{label} tensor differs: {key}")
    return len(a), sum(value.numel() for value in a.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--inference", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    original = TopPFrozenAssets(ASSETS, args.source, device="cpu", chunk=4)
    compact = TopPFrozenAssets(ASSETS, args.inference, device="cpu", chunk=4)
    original.assert_frozen()
    compact.assert_frozen()
    if (original.audit["top_p_artifact_sha256"] != TOP_P_SHA256
            or compact.audit["top_p_artifact_sha256"] != TOP_P_INFERENCE_SHA256
            or compact.audit["top_p_checkpoint_sha256"] != TOP_P_SHA256
            or original.audit["loaded_tensors"] != compact.audit["loaded_tensors"]):
        raise RuntimeError("top-p asset provenance differs")
    print("native", same_state(original.native, compact.native, "native"), flush=True)
    print("shell", same_state(original.shell, compact.shell, "shell"), flush=True)
    print("strict-loaded source and inference assets are bitwise equal", flush=True)


if __name__ == "__main__":
    main()
