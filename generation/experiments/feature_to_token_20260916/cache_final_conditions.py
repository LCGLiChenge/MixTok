"""Cache the final top-p mapper proxy and scorer route for every ADM view."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from bert2d.paths import atomic_json, sha256
from experiments.feature_to_token_20260916.assets_top_p import TopPFrozenAssets
from experiments.feature_to_token_20260916.converter_codebook import CodebookFeatureTokenConverter
from experiments.feature_to_token_20260916.top_p_scorer_original import (
    OneDTopPGridScorer, top_p_mask,
)
from experiments.feature_to_token_20260916.train_proxy_converter import make_features


FORMAT = "top_p_mapper_route_cache_v1"
TOP_P_SHA256 = "61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819"
NUM_SOURCES = 1_281_167
NUM_ROWS = NUM_SOURCES * 2


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_codes(args):
    meta = json.loads((args.source_root / "meta.json").read_text())
    require(meta.get("completed") is True and int(meta.get("num_samples", -1)) == NUM_SOURCES,
            "top-p paired source cache is incomplete")
    require(int(meta.get("mot_step", -1)) == 247482 and meta.get("mot_state_key") == "model_ema",
            "source cache is not the specified top-p EMA")
    require(Path(meta.get("mot_ckpt", "")).resolve() == args.top_p_ckpt.resolve(),
            "source cache checkpoint path differs")
    require((args.source_root / "manifest.sha256").is_file(), "source hash manifest is missing")
    codes = np.load(args.source_root / "titok_codes.npy", mmap_mode="r", allow_pickle=False)
    require(codes.shape == (NUM_SOURCES, 2, 32) and codes.dtype == np.uint16,
            "top-p source 1D code shape/dtype differs")
    return codes


def config_for(args, world):
    codes = source_codes(args)
    require(args.p == 0.75 and args.temperature == 2.0,
            "use the accepted top-p scorer p=0.75, temperature=2.0")
    require(sha256(args.top_p_ckpt) == TOP_P_SHA256, "top-p tokenizer hash differs")
    scorer_hash, mapper_hash = sha256(args.scorer_ckpt), sha256(args.mapper_ckpt)
    mapper = torch.load(args.mapper_ckpt, map_location="cpu", mmap=True, weights_only=True)
    require(mapper.get("step") == 20000 and mapper.get("format") ==
            "feature_proxy_token_converter_codebook_v1", "mapper is not the fresh 20k codebook run")
    require(mapper["train_config"].get("top_p_checkpoint_sha256") == TOP_P_SHA256,
            "mapper trained with different tokenizer")
    del mapper
    scorer = torch.load(args.scorer_ckpt, map_location="cpu", weights_only=True)
    require(scorer.get("step") == 20000, "scorer is not the accepted 20k run")
    del scorer
    return {
        "format": FORMAT, "status": "prepared", "shape": [NUM_ROWS, 256],
        "dtype": "uint16", "route_dtype": "uint16",
        "flat_layout": "flat_id=source_index*num_aug+augmentation",
        "p": args.p, "temperature": args.temperature, "world_size": world,
        "source_root": str(args.source_root.resolve()),
        "source_meta_sha256": sha256(args.source_root / "meta.json"),
        "source_manifest_sha256": sha256(args.source_root / "manifest.sha256"),
        "source_titok_sha256": sha256(args.source_root / "titok_codes.npy"),
        "source_shape": list(codes.shape),
        "top_p_checkpoint": str(args.top_p_ckpt.resolve()),
        "top_p_checkpoint_sha256": TOP_P_SHA256,
        "scorer_checkpoint": str(args.scorer_ckpt.resolve()),
        "scorer_sha256": scorer_hash,
        "scorer_source_sha256": sha256(Path(__file__).with_name("top_p_scorer_original.py")),
        "mapper_checkpoint": str(args.mapper_ckpt.resolve()),
        "mapper_sha256": mapper_hash,
        "cache_source_sha256": sha256(Path(__file__)),
        "frozen_assets_source_sha256": sha256(Path(__file__).with_name("assets_top_p.py")),
        "precision": "f1d/scorer/mapper FP32 noTF32; no RGB input",
    }


def prepare(args):
    config = config_for(args, args.world_size)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, shape in (("proxy_codes.npy", (NUM_ROWS, 256)),
                        ("route_indices.npy", (NUM_ROWS, 256)),
                        ("route_k.npy", (NUM_ROWS,))):
        array = np.lib.format.open_memmap(args.output / name, mode="w+",
                                          dtype=np.uint16, shape=shape)
        array.flush()
    for rank in range(args.world_size):
        count = len(range(rank, NUM_ROWS, args.world_size))
        written = np.lib.format.open_memmap(args.output / f"written_rank{rank}.npy",
                                             mode="w+", dtype=np.bool_, shape=(count,))
        written[:] = False
        written.flush()
    atomic_json(args.output / "config.json", config)
    print(json.dumps({"status": "prepared", "rows": NUM_ROWS,
                      "output": str(args.output)}), flush=True)


def load_models(args, config, device):
    frozen = TopPFrozenAssets(args.assets_root, args.top_p_ckpt, device,
                              chunk=args.batch)
    scorer_state = torch.load(args.scorer_ckpt, map_location="cpu", weights_only=True)
    scorer_args = scorer_state["args"]
    scorer = OneDTopPGridScorer(hidden=int(scorer_args["hidden"]),
                                depth=int(scorer_args["depth"])).to(device)
    scorer.load_state_dict(scorer_state["model"], strict=True)
    scorer.eval().requires_grad_(False)
    mapper_state = torch.load(args.mapper_ckpt, map_location="cpu", mmap=True,
                              weights_only=True)
    mapper = CodebookFeatureTokenConverter(**mapper_state["model_config"]).to(device)
    with torch.no_grad():
        native_book = frozen.shell.llamagen_vq.quantize.get_codebook_entry(
            torch.arange(16384, device=device))
        mapper.initialize_codebook(native_book.float(),
                                   frozen.shell.llamagen_vq.post_quant_conv)
    mapper.load_state_dict(mapper_state["model"], strict=True)
    mapper = mapper.eval().requires_grad_(False)
    if (config["scorer_sha256"] != sha256(args.scorer_ckpt)
            or config["mapper_sha256"] != sha256(args.mapper_ckpt)):
        raise RuntimeError("frozen scorer or mapper checkpoint changed after prepare")
    return frozen, scorer, mapper


@torch.inference_mode()
def extract(args):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((args.output / "config.json").read_text())
    require(config["format"] == FORMAT and config["world_size"] == args.world_size,
            "cache config differs")
    require(config["p"] == args.p and config["temperature"] == args.temperature,
            "top-p selection differs")
    source = source_codes(args)
    proxy = np.load(args.output / "proxy_codes.npy", mmap_mode="r+", allow_pickle=False)
    route = np.load(args.output / "route_indices.npy", mmap_mode="r+", allow_pickle=False)
    route_k = np.load(args.output / "route_k.npy", mmap_mode="r+", allow_pickle=False)
    written = np.load(args.output / f"written_rank{args.rank}.npy", mmap_mode="r+",
                      allow_pickle=False)
    flat_ids = np.arange(args.rank, NUM_ROWS, args.world_size, dtype=np.int64)
    require(written.shape == (len(flat_ids),), "rank partition changed")
    pending = np.flatnonzero(~np.asarray(written))
    frozen, scorer, mapper = load_models(args, config, device)
    started = time.monotonic()
    done = 0
    for begin in range(0, len(pending), args.batch):
        local = pending[begin:begin + args.batch]
        rows = flat_ids[local]
        image, view = np.divmod(rows, 2)
        ids = torch.from_numpy(np.array(source[image, view], dtype=np.int64)).to(device)
        f1d = make_features(ids, frozen.native.quantize,
                            frozen.shell.latent_decoder)
        proxy_ids = mapper.tokens(f1d.float()).reshape(len(rows), 256)
        selected, k_tensor = top_p_mask(scorer(f1d.float()), args.p,
                                        args.temperature)
        require(bool(((proxy_ids >= 0) & (proxy_ids < 16384)).all()), "bad proxy ID")
        proxy[rows] = proxy_ids.cpu().numpy().astype(np.uint16, copy=False)
        for j, row in enumerate(rows):
            chosen = torch.nonzero(selected[j], as_tuple=False).flatten().cpu().numpy()
            k = int(k_tensor[j])
            require(k == len(chosen) and 1 <= k <= 256, "invalid top-p K")
            route[row, :k] = chosen.astype(np.uint16, copy=False)
            route_k[row] = k
        written[local] = True
        done += len(local)
        if done % (args.batch * args.flush_every) == 0 or done == len(pending):
            proxy.flush(); route.flush(); route_k.flush(); written.flush()
            record = {"status": "running", "rank": args.rank, "pid": os.getpid(),
                      "completed": int(written.sum()), "rank_total": len(written),
                      "elapsed_seconds": time.monotonic() - started,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3}
            atomic_json(args.output / f"status_rank{args.rank}.json", record)
            print(json.dumps(record), flush=True)
    proxy.flush(); route.flush(); route_k.flush(); written.flush()
    frozen.assert_frozen()
    atomic_json(args.output / f"status_rank{args.rank}.json",
                {"status": "complete", "rank": args.rank, "pid": os.getpid(),
                 "completed": int(written.sum()), "rank_total": len(written),
                 "elapsed_seconds": time.monotonic() - started,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3})


def finalize(args):
    config = json.loads((args.output / "config.json").read_text())
    require(config["format"] == FORMAT and config["world_size"] == args.world_size,
            "cache config differs")
    for rank in range(args.world_size):
        written = np.load(args.output / f"written_rank{rank}.npy", mmap_mode="r",
                          allow_pickle=False)
        require(bool(written.all()), f"rank {rank} did not finish every row")
    proxy = np.load(args.output / "proxy_codes.npy", mmap_mode="r", allow_pickle=False)
    route = np.load(args.output / "route_indices.npy", mmap_mode="r", allow_pickle=False)
    route_k = np.load(args.output / "route_k.npy", mmap_mode="r", allow_pickle=False)
    require(proxy.shape == (NUM_ROWS, 256) and route.shape == (NUM_ROWS, 256)
            and route_k.shape == (NUM_ROWS,), "final cache dimensions differ")
    histogram = np.zeros(257, dtype=np.int64)
    low, high = 16384, -1
    for begin in range(0, NUM_ROWS, 8192):
        end = min(begin + 8192, NUM_ROWS)
        p = np.asarray(proxy[begin:end]); r = np.asarray(route[begin:end]); k = np.asarray(route_k[begin:end])
        require(bool(np.all((k >= 1) & (k <= 256))), "route K outside [1,256]")
        require(bool(np.all(p < 16384)), "proxy ID outside vocabulary")
        valid = np.arange(256)[None, :] < k[:, None]
        require(bool(np.all(r[valid] < 256)), "selected route index outside grid")
        pairs = np.arange(255)[None, :] < (k[:, None] - 1)
        require(not bool(np.any((np.diff(r.astype(np.int32), axis=1) <= 0) & pairs)),
                "selected route indices are not strictly sorted")
        histogram += np.bincount(k.astype(np.int64), minlength=257)
        low = min(low, int(p.min())); high = max(high, int(p.max()))
    require(int(histogram.sum()) == NUM_ROWS, "K histogram row count differs")
    k = np.asarray(route_k)
    summary = {"format": FORMAT, "status": "complete", "complete_coverage": True,
               "rows": NUM_ROWS, "mean_k": float(k.mean()),
               "p10_k": float(np.quantile(k, 0.1)), "p50_k": float(np.quantile(k, 0.5)),
               "p90_k": float(np.quantile(k, 0.9)), "p99_k": float(np.quantile(k, 0.99)),
               "min_k": int(k.min()), "max_k": int(k.max()),
               "k_histogram": histogram.tolist(), "minimum_proxy_id": low,
               "maximum_proxy_id": high,
               "scorer_sha256": config["scorer_sha256"],
               "mapper_sha256": config["mapper_sha256"],
               "top_p_checkpoint_sha256": config["top_p_checkpoint_sha256"],
               "proxy_codes_sha256": sha256(args.output / "proxy_codes.npy"),
               "route_indices_sha256": sha256(args.output / "route_indices.npy"),
               "route_k_sha256": sha256(args.output / "route_k.npy"),
               "config_sha256": sha256(args.output / "config.json")}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"status": "complete", "rows": NUM_ROWS,
                      "mean_k": summary["mean_k"]}), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "extract", "finalize"))
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--top-p-ckpt", type=Path, required=True)
    p.add_argument("--scorer-ckpt", type=Path, required=True)
    p.add_argument("--mapper-ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument("--p", type=float, default=0.75)
    p.add_argument("--temperature", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    require(args.world_size > 0 and 0 <= args.rank < args.world_size
            and args.batch > 0 and args.flush_every > 0, "invalid rank or batch")
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "extract":
        extract(args)
    else:
        finalize(args)
