"""Cache teacher token IDs for the continuous embedding adapter.

The cache is deterministic and resumable.  Multiple ranks own disjoint flat
sample IDs and write into one pre-created uint16 NPY array.  No Router, 2D GT
tokens, or trainable converter is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")

from bert2d.paths import ASSETS, atomic_json, sha256
from h20.assets import specifications
from experiments.continuous_halton_embedding_20260923.assets_top_p import TopPFrozenAssets, TOP_P_SHA256, verified_top_p_artifact, verify_source_meta


FORMAT = "top_p_1d_base_rgb_reencoded_proxy_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def registered_spec(relative: str) -> dict:
    matches = [item for item in specifications() if item["local_path"] == relative]
    require(len(matches) == 1, f"missing unique asset specification: {relative}")
    return matches[0]


def verify_file(root: Path, relative: str) -> str:
    spec = registered_spec(relative)
    path = root / relative
    require(path.is_file(), f"missing asset: {path}")
    require(path.stat().st_size == spec["size"], f"asset size mismatch: {path}")
    actual = sha256(path)
    require(actual == spec["sha256"], f"asset sha256 mismatch: {path}")
    return actual


def load_exact_models(root: Path, top_p_ckpt: Path, device: torch.device):
    """Load only the modules required by the exact RGB round trip."""
    frozen = TopPFrozenAssets(root, top_p_ckpt, device, chunk=1)
    modules = nn.ModuleDict(dict(quantizer=frozen.native.quantize,
                                 latent_decoder=frozen.shell.latent_decoder,
                                 vq=frozen.shell.llamagen_vq))
    modules.eval().requires_grad_(False)
    require(
        all(not module.training for module in modules.modules())
        and all(not parameter.requires_grad for parameter in modules.parameters()),
        "cache models must remain frozen",
    )
    audit = {
        "top_p_step": frozen.audit["top_p_step"],
        "top_p_state": "model_ema",
        "native_tokenizer_sha256": verify_file(root, "weights/tokenizer_titok_l32.bin"),
        "top_p_checkpoint_sha256": TOP_P_SHA256,
        "codebook_size": 16384,
        "precision": "fp32_no_tf32",
        "rgb_transform": "clamp((decoder(f_1d)+1)/2,0,1); encoder_input=2*rgb-1",
        "original_llamagen_checkpoint_used": False,
    }
    return modules, audit


@torch.inference_mode()
def encode_batch(models: nn.ModuleDict, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    require(ids.dtype == torch.long and ids.ndim == 2 and ids.shape[1] == 32, "z1 shape")
    require(bool(((ids >= 0) & (ids < 4096)).all()), "z1 ID outside [0,4095]")
    quantized = models["quantizer"].get_codebook_entry(ids.reshape(-1))
    quantized = quantized.reshape(len(ids), 1, 32, -1).permute(0, 3, 1, 2).contiguous()
    feature = models["latent_decoder"](quantized.float())
    rgb = ((models["vq"].decoder(feature.float()) + 1.0) * 0.5).clamp(0, 1)
    require(rgb.dtype == torch.float32 and rgb.shape[1:] == (3, 256, 256), "RGB contract")
    require(bool(torch.isfinite(rgb).all()), "non-finite RGB")
    codes = models["vq"].encode(rgb * 2.0 - 1.0)[2][2].reshape(len(ids), 256).long()
    require(bool(((codes >= 0) & (codes < 16384)).all()), "teacher ID outside [0,16383]")
    return codes, rgb


def source_contract(codes_root: Path, top_p_ckpt: Path):
    meta_path = codes_root / "meta.json"
    meta = json.loads(meta_path.read_text())
    verify_source_meta(meta)
    require(meta.get("completed") is True and int(meta.get("format_version", -1)) == 1, "bad cache meta")
    require(int(meta.get("num_samples", -1)) == 1_281_167, "full ImageNet train required")
    require(int(meta.get("num_aug", -1)) == 2 and meta.get("augmentation") == "adm", "two ADM views required")
    require(int(meta.get("mot_step", -1)) == 247482 and meta.get("mot_state_key") == "model_ema", "top-p EMA cache required")
    if "mot_source_sha256" not in meta:
        require(Path(meta.get("mot_ckpt", "")).resolve() == top_p_ckpt.resolve(),
                "legacy source cache path differs; re-extract to record checkpoint hash")
    verified_top_p_artifact(top_p_ckpt)
    z1 = np.load(codes_root / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
    written = np.load(codes_root / "written.npy", mmap_mode="r", allow_pickle=False)
    require(z1.shape == (1_281_167, 2, 32) and z1.dtype == np.uint16, "bad TiTok cache layout")
    # The original packed extractor marks a source row only after writing both
    # ADM views, hence this bitmap is [N] uint8 rather than [N,2] bool.
    require(
        written.shape == (1_281_167,)
        and written.dtype == np.uint8
        and bool((written == 1).all()),
        "source cache incomplete",
    )
    return z1, meta, sha256(meta_path), sha256(codes_root / "titok_codes.npy")


def expected_config(args, z1, meta_sha: str, z1_sha: str, total: int) -> dict:
    return {
        "format": FORMAT,
        "shape": [total, 256],
        "dtype": "uint16",
        "flat_layout": "flat_id=source_index*num_aug+augmentation",
        "source_shape": list(z1.shape),
        "source_root": str(args.source_root.resolve()),
        "source_meta_sha256": meta_sha,
        "source_titok_sha256": z1_sha,
        "top_p_checkpoint_sha256": TOP_P_SHA256,
        "top_p_artifact_sha256": verified_top_p_artifact(args.top_p_ckpt),
        "top_p_checkpoint": str(args.top_p_ckpt.resolve()),
        "total_available": int(z1.shape[0] * z1.shape[1]),
        "world_size": args.world_size,
        "limit": args.limit,
    }


def prepare(args) -> None:
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT.resolve()), "large cache must be outside repository")
    output.mkdir(parents=True, exist_ok=True)
    verified_top_p_artifact(args.top_p_ckpt)
    z1, _, meta_sha, z1_sha = source_contract(args.source_root, args.top_p_ckpt)
    available = int(z1.shape[0] * z1.shape[1])
    total = available if args.limit == 0 else min(args.limit, available)
    config = expected_config(args, z1, meta_sha, z1_sha, total)
    config_path = output / "config.json"
    data_path = output / "proxy_codes.npy"
    if config_path.exists():
        require(json.loads(config_path.read_text()) == config, "resume config mismatch")
        values = np.load(data_path, mmap_mode="r", allow_pickle=False)
        require(values.shape == (total, 256) and values.dtype == np.uint16, "resume array mismatch")
    else:
        require(not data_path.exists(), "orphan proxy array requires inspection")
        values = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.uint16, shape=(total, 256))
        values.flush()
        atomic_json(config_path, config)
    for rank in range(args.world_size):
        count = len(range(rank, total, args.world_size))
        path = output / f"written_rank{rank}.npy"
        if path.exists():
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            require(value.shape == (count,) and value.dtype == np.bool_, "written resume mismatch")
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=np.bool_, shape=(count,))
            value[:] = False
            value.flush()
    atomic_json(output / "prepare.json", {**config, "status": "prepared", "bytes": data_path.stat().st_size})
    print(json.dumps({"status": "prepared", "total": total, "output": str(output)}), flush=True)


def extract(args) -> None:
    require(0 <= args.rank < args.world_size, "invalid rank")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    free, total_memory = torch.cuda.mem_get_info(device)
    require(free >= args.min_free_gib * 1024**3, "insufficient free GPU memory at admission")

    output = args.output.resolve()
    config = json.loads((output / "config.json").read_text())
    require(config["format"] == FORMAT and config["world_size"] == args.world_size, "config mismatch")
    total = int(config["shape"][0])
    z1, _, meta_sha, z1_sha = source_contract(args.source_root, args.top_p_ckpt)
    require(meta_sha == config["source_meta_sha256"] and z1_sha == config["source_titok_sha256"], "source drift")
    values = np.load(output / "proxy_codes.npy", mmap_mode="r+", allow_pickle=False)
    written = np.load(output / f"written_rank{args.rank}.npy", mmap_mode="r+", allow_pickle=False)
    flat_ids = np.arange(args.rank, total, args.world_size, dtype=np.int64)
    require(len(flat_ids) == len(written), "rank partition mismatch")
    pending = np.flatnonzero(~np.asarray(written))
    status_path = output / f"status_rank{args.rank}.json"
    started = time.monotonic()
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    atomic_json(status_path, {"status": "loading", "rank": args.rank, "pid": os.getpid(), "remaining": len(pending)})
    models, model_audit = load_exact_models(args.assets_root, args.top_p_ckpt, device)
    first_checked = False
    completed_before = int(written.sum())
    ticks = []
    for offset in range(0, len(pending), args.batch):
        if stop:
            break
        local = pending[offset:offset + args.batch]
        ids = flat_ids[local]
        rows, aug = np.divmod(ids, z1.shape[1])
        tokens = torch.from_numpy(np.array(z1[rows, aug], dtype=np.int64)).to(device)
        tick = time.monotonic()
        codes, rgb = encode_batch(models, tokens)
        if not first_checked:
            again, rgb_again = encode_batch(models, tokens)
            require(torch.equal(codes, again) and torch.equal(rgb, rgb_again), "first-batch deterministic replay differs")
            first_checked = True
        values[ids] = codes.cpu().numpy().astype(np.uint16, copy=False)
        written[local] = True
        ticks.append(time.monotonic() - tick)
        done_now = offset + len(local)
        if done_now % (args.batch * args.flush_every) == 0 or done_now == len(pending):
            values.flush()
            written.flush()
            completed = completed_before + done_now
            elapsed = time.monotonic() - started
            rate = done_now / max(elapsed, 1e-9)
            record = {
                "status": "running",
                "rank": args.rank,
                "pid": os.getpid(),
                "completed": completed,
                "rank_total": len(written),
                "items_per_second": rate,
                "eta_seconds": (len(pending) - done_now) / max(rate, 1e-9),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                "admission_free_gib": free / 1024**3,
                "gpu_total_gib": total_memory / 1024**3,
                "batch_seconds_mean_recent": float(np.mean(ticks[-args.flush_every:])),
                "model_audit": model_audit,
            }
            atomic_json(status_path, record)
            print(json.dumps(record), flush=True)
        del tokens, codes, rgb
    values.flush()
    written.flush()
    final_status = "stopped" if stop else "complete"
    atomic_json(status_path, {
        "status": final_status,
        "rank": args.rank,
        "pid": os.getpid(),
        "completed": int(written.sum()),
        "rank_total": len(written),
        "seconds": time.monotonic() - started,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "deterministic_replay_exact": first_checked,
        "model_audit": model_audit,
    })
    print(json.dumps({"status": final_status, "rank": args.rank, "completed": int(written.sum())}), flush=True)


def finalize(args) -> None:
    output = args.output.resolve()
    config = json.loads((output / "config.json").read_text())
    total = int(config["shape"][0])
    for rank in range(args.world_size):
        written = np.load(output / f"written_rank{rank}.npy", mmap_mode="r", allow_pickle=False)
        require(bool(written.all()), f"rank {rank} is incomplete")
    values = np.load(output / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
    require(values.shape == (total, 256) and values.dtype == np.uint16, "final layout differs")
    low, high = 16384, -1
    for offset in range(0, total, 8192):
        chunk = np.asarray(values[offset:offset + 8192])
        low = min(low, int(chunk.min()))
        high = max(high, int(chunk.max()))
    require(0 <= low <= high < 16384, "final teacher IDs invalid")
    summary = {
        **config,
        "status": "complete",
        "minimum_id": low,
        "maximum_id": high,
        "proxy_codes_sha256": sha256(output / "proxy_codes.npy"),
        "config_sha256": sha256(output / "config.json"),
        "bytes": (output / "proxy_codes.npy").stat().st_size,
        "complete_coverage": True,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def repartition(args) -> None:
    """Losslessly migrate stopped rank bitmaps without touching proxy data."""
    output = args.output.resolve()
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    old_world = int(config["world_size"])
    new_world = int(args.world_size)
    require(old_world > 0 and new_world > 0 and old_world != new_world, "world size must change")
    total = int(config["shape"][0])
    values = np.load(output / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
    require(values.shape == (total, 256) and values.dtype == np.uint16, "proxy array changed")

    global_written = np.zeros(total, dtype=np.bool_)
    old_counts = []
    for rank in range(old_world):
        status_path = output / f"status_rank{rank}.json"
        status = json.loads(status_path.read_text())
        pid = int(status["pid"])
        alive = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        require(not alive, f"old extraction rank {rank} is still alive")
        written_path = output / f"written_rank{rank}.npy"
        written = np.load(written_path, mmap_mode="r", allow_pickle=False)
        ids = np.arange(rank, total, old_world, dtype=np.int64)
        require(written.shape == (len(ids),) and written.dtype == np.bool_, "old bitmap layout differs")
        global_written[ids] = np.asarray(written)
        old_counts.append(int(written.sum()))

        backup_written = output / f"written_rank{rank}_world{old_world}.npy"
        backup_status = output / f"status_rank{rank}_world{old_world}.json"
        require(not backup_written.exists() and not backup_status.exists(), "repartition backup already exists")
        written_path.replace(backup_written)
        shutil.copy2(status_path, backup_status)

    completed = int(global_written.sum())
    new_counts = []
    for rank in range(new_world):
        ids = np.arange(rank, total, new_world, dtype=np.int64)
        path = output / f"written_rank{rank}.npy"
        require(not path.exists() or rank < old_world, "unexpected destination bitmap")
        temporary = output / f"written_rank{rank}_world{new_world}.new.npy"
        require(not temporary.exists(), "stale repartition temporary exists")
        written = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.bool_, shape=(len(ids),))
        written[:] = global_written[ids]
        written.flush()
        new_counts.append(int(written.sum()))
        temporary.replace(path)

    require(sum(old_counts) == completed == sum(new_counts), "completed-count conservation failed")
    backup_config = output / f"config_world{old_world}.json"
    require(not backup_config.exists(), "config backup already exists")
    shutil.copy2(config_path, backup_config)
    history = list(config.get("repartition_history", []))
    history.append({
        "from_world": old_world,
        "to_world": new_world,
        "completed": completed,
        "old_counts": old_counts,
        "new_counts": new_counts,
        "proxy_codes_sha256_not_recomputed": True,
    })
    config["world_size"] = new_world
    config["repartition_history"] = history
    atomic_json(config_path, config)
    receipt = {
        "status": "complete",
        "from_world": old_world,
        "to_world": new_world,
        "completed": completed,
        "old_counts": old_counts,
        "new_counts": new_counts,
        "count_conserved": True,
        "proxy_data_untouched": True,
        "config_sha256": sha256(config_path),
    }
    atomic_json(output / f"repartition_world{old_world}_to_world{new_world}.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "extract", "finalize", "repartition"))
    parser.add_argument("--assets-root", type=Path, default=ASSETS)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--top-p-ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="First N flat items; zero means all")
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    require(args.world_size > 0 and args.batch > 0 and args.flush_every > 0, "positive sizes required")
    require(0 < args.memory_fraction <= 1, "invalid memory fraction")
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "extract":
        extract(args)
    elif args.mode == "finalize":
        finalize(args)
    else:
        repartition(args)
