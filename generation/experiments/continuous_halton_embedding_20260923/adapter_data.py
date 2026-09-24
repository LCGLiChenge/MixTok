"""Data and checkpoint helpers for the continuous embedding adapter."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from bert2d.paths import atomic_json, sha256
from experiments.continuous_halton_embedding_20260923.assets_top_p import TOP_P_SHA256, verify_source_meta

PROXY_FORMAT = "top_p_1d_base_rgb_reencoded_proxy_v1"
NUM_SOURCES = 1_281_167
NUM_AUG = 2
VOCABULARY = 16_384

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_split(seed: int, eval_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    cutoff = int(round(eval_fraction * 10_000))
    require(1 <= cutoff <= 9_999, "eval fraction must map to [1,9999]/10000")
    source = np.arange(NUM_SOURCES, dtype=np.uint64)
    hashed = source * np.uint64(6364136223846793005) + np.uint64(seed)
    is_eval = (hashed % np.uint64(10_000)) < np.uint64(cutoff)
    return np.flatnonzero(~is_eval).astype(np.int64), np.flatnonzero(is_eval).astype(np.int64)


def flat_rows(source: np.ndarray) -> np.ndarray:
    result = np.empty(len(source) * NUM_AUG, dtype=np.int64)
    result[0::2] = source * NUM_AUG
    result[1::2] = source * NUM_AUG + 1
    return result


def training_order(rows: np.ndarray, count: int, seed: int,
                   repeat: bool) -> np.ndarray:
    """Build deterministic per-pass shuffles without replacement within a pass."""
    require(count > 0, "positive training sample count required")
    if not repeat:
        require(count <= len(rows), "pilot requires no-replacement training rows")
    rng = np.random.default_rng(seed)
    result = np.empty(count, dtype=np.int64)
    offset = 0
    while offset < count:
        permutation = rng.permutation(rows)
        take = min(len(permutation), count - offset)
        result[offset:offset + take] = permutation[:take]
        offset += take
    return result


def load_data_contract(codes_root: Path, proxy_root: Path, top_p_sha256: str):
    meta_path = codes_root / "meta.json"
    meta = json.loads(meta_path.read_text())
    verify_source_meta(meta, codes_root)
    require(meta.get("completed") is True, "packed cache is incomplete")
    require(int(meta.get("num_samples", -1)) == NUM_SOURCES, "wrong source count")
    require(int(meta.get("num_aug", -1)) == NUM_AUG, "two ADM views required")
    require(int(meta.get("mot_step", -1)) == 247482, "top-p step 247482 required")
    require(meta.get("mot_state_key") == "model_ema", "top-p EMA cache required")
    z1 = np.load(codes_root / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
    written = np.load(codes_root / "written.npy", mmap_mode="r", allow_pickle=False)
    require(z1.shape == (NUM_SOURCES, NUM_AUG, 32) and z1.dtype == np.uint16,
            "bad TiTok cache layout")
    require(written.shape == (NUM_SOURCES,) and written.dtype == np.uint8
            and bool((written == 1).all()), "packed TiTok cache is incomplete")

    proxy_config_path = proxy_root / "config.json"
    proxy_summary_path = proxy_root / "summary.json"
    proxy_config = json.loads(proxy_config_path.read_text())
    proxy_summary = json.loads(proxy_summary_path.read_text())
    require(proxy_config.get("format") == PROXY_FORMAT, "wrong proxy semantics")
    require(proxy_config.get("flat_layout") == "flat_id=source_index*num_aug+augmentation",
            "wrong proxy indexing")
    require(proxy_summary.get("status") == "complete"
            and proxy_summary.get("complete_coverage") is True,
            "proxy cache is incomplete")
    require(proxy_summary.get("proxy_codes_sha256"), "proxy cache has no final SHA256")
    require(proxy_summary.get("top_p_checkpoint_sha256") == top_p_sha256,
            "proxy targets use a different top-p tokenizer")
    require(proxy_config.get("source_titok_sha256") == sha256(codes_root / "titok_codes.npy"),
            "proxy targets use different 1D rows")
    proxy = np.load(proxy_root / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
    require(proxy.shape == (NUM_SOURCES * NUM_AUG, 256) and proxy.dtype == np.uint16,
            "bad proxy cache layout")
    require(0 <= int(proxy_summary["minimum_id"])
            <= int(proxy_summary["maximum_id"]) < VOCABULARY,
            "proxy token range is invalid")
    audit = {
        "packed_meta_sha256": sha256(meta_path),
        "proxy_config_sha256": sha256(proxy_config_path),
        "proxy_summary_sha256": sha256(proxy_summary_path),
        "proxy_codes_sha256": proxy_summary["proxy_codes_sha256"],
    }
    return z1, proxy, audit


def make_features(ids: torch.Tensor, quantizer, decoder) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(ids.device.type, enabled=False):
        quantized = quantizer.get_codebook_entry(ids.reshape(-1))
        quantized = quantized.reshape(len(ids), 1, 32, -1).permute(0, 3, 1, 2).contiguous()
        features = decoder(quantized.float())
    require(features.shape == (len(ids), 256, 16, 16), "unexpected f_1d shape")
    return features.detach()


def save_checkpoint(out: Path, core, optimizer, step: int, config: dict, smoke: bool) -> dict:
    state = {"format": config["format"], "model_config": core.config,
             "model": {key: value.detach().cpu() for key, value in core.state_dict().items()},
             "step": step, "train_config": config, "optimizer": optimizer.state_dict()}
    temporary = out / "latest.pt.tmp"
    torch.save(state, temporary)
    loaded = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
    require(loaded["format"] == config["format"] and loaded["step"] == step, "checkpoint header mismatch")
    require(set(loaded["model"]) == set(state["model"]), "checkpoint keys mismatch")
    for name, value in state["model"].items():
        require(torch.equal(value, loaded["model"][name]), f"checkpoint tensor mismatch: {name}")
    receipt = {"step": step, "sha256": sha256(temporary),
               "bytes": temporary.stat().st_size, "round_trip_exact": True}
    if smoke:
        temporary.unlink()
        receipt["smoke_checkpoint_removed"] = True
    else:
        temporary.replace(out / "latest.pt")
        atomic_json(out / "latest.json", receipt)
    return receipt
