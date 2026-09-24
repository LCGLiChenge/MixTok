"""Launch multi-GPU sharded free-generation FID for BERT sparse 2D-only checkpoints.

Each shard is an independent single-GPU invocation of
``bert_sparse2d_free_fid``.  Shards save pool features for disjoint global
sample-id ranges; this wrapper verifies complete coverage, merges the pools, and
computes one global FID/IS with the same ADM evaluator used by the single-GPU
script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

import numpy as np

from .paths import ASSETS, RESULT_ROOT, ROOT, atomic_json, output_path
from .paths import EVALUATOR, REFERENCE, GRAPH, digest
from .eval import array_digest, split_ids


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_gpus(value: str) -> list[str]:
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    require(gpus, "at least one GPU is required")
    require(len(set(gpus)) == len(gpus), "duplicate GPU ids are not allowed")
    require(len(gpus) <= 8, "too many local shards requested")
    return gpus


def load_evaluator_module():
    spec = importlib.util.spec_from_file_location("mix_style_free_adm_eval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH = str(GRAPH)
    return module


def shard_command(args, shard_dir: Path, num_shards: int, shard_index: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "bert2d.eval",
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(shard_dir),
        "--state",
        args.state,
        "--n",
        str(args.n),
        "--batch",
        str(args.batch),
        "--feature-batch",
        str(args.feature_batch),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--feature-chunk",
        str(args.feature_chunk),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    cmd.extend(["--expected-checkpoint-sha256", args._checkpoint_hash,
                "--expected-checkpoint-step", str(args._checkpoint_step)])
    if args.assets_root_override is not None:
        cmd.extend(["--assets-root-override", str(args.assets_root_override)])
    if args.smoke:
        cmd.append("--smoke")
    if args.allow_50k:
        cmd.append("--allow-50k")
    if args.variants:
        cmd.append("--variants")
        cmd.extend(args.variants)
    return cmd


def launch_shards(args, output: Path, gpus: list[str]) -> None:
    shard_root = output / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    procs = []
    selector = selectors.DefaultSelector()
    try:
        for shard_index, gpu in enumerate(gpus):
            shard_dir = shard_root / f"shard_{shard_index:02d}"
            require(not shard_dir.exists(), f"fresh shard output required: {shard_dir}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("USE_TF", "0")
            env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            cmd = shard_command(args, shard_dir, len(gpus), shard_index)
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            procs.append((shard_index, gpu, proc, shard_dir, cmd))
            selector.register(proc.stdout, selectors.EVENT_READ, (shard_index, gpu))
            print(json.dumps({"stage": "launch", "shard": shard_index, "gpu": gpu, "pid": proc.pid, "output": str(shard_dir)}), flush=True)

        alive = {proc.pid for _, _, proc, _, _ in procs}
        last_heartbeat = time.monotonic()
        failed = None
        while alive:
            for key, _ in selector.select(timeout=5.0):
                line = key.fileobj.readline()
                if line:
                    shard_index, gpu = key.data
                    print(json.dumps({"stage": "shard_log", "shard": shard_index, "gpu": gpu, "line": line.rstrip()}), flush=True)
            for shard_index, gpu, proc, shard_dir, cmd in procs:
                if proc.pid not in alive:
                    continue
                code = proc.poll()
                if code is None:
                    continue
                alive.remove(proc.pid)
                if proc.stdout is not None:
                    try:
                        selector.unregister(proc.stdout)
                    except Exception:
                        pass
                    for line in proc.stdout:
                        print(json.dumps({"stage": "shard_log", "shard": shard_index, "gpu": gpu, "line": line.rstrip()}), flush=True)
                print(json.dumps({"stage": "exit", "shard": shard_index, "gpu": gpu, "pid": proc.pid, "returncode": code}), flush=True)
                if code != 0 and failed is None:
                    failed = (shard_index, gpu, code, cmd)
            if failed is not None:
                for _, _, proc, _, _ in procs:
                    if proc.poll() is None:
                        proc.terminate()
                raise RuntimeError(f"shard {failed[0]} on GPU {failed[1]} failed with code {failed[2]}")
            now = time.monotonic()
            if now - last_heartbeat > 60:
                running = [dict(shard=i, gpu=g, pid=p.pid) for i, g, p, _, _ in procs if p.poll() is None]
                print(json.dumps({"stage": "heartbeat", "running": running}), flush=True)
                last_heartbeat = now
    finally:
        selector.close()
        for _, _, proc, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        survivors=[]
        for _, _, proc, _, _ in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                survivors.append(proc.pid)
            if proc.stdout is not None:
                proc.stdout.close()
        if survivors:
            raise RuntimeError(f"eval children did not terminate: {survivors}")


def merge_shards(args, output: Path, gpus: list[str]) -> dict:
    # Limit the merge-time TensorFlow graph to one visible GPU.  Generation and
    # ADM feature extraction have already happened inside per-shard workers;
    # merge only needs the softmax graph for Inception Score when n is large.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    module = load_evaluator_module()
    tf = module.tf
    tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8, inter_op_parallelism_threads=2)
    config.gpu_options.allow_growth = True

    shard_root = output / "shards"
    summaries = []
    endpoints = None
    coverage = np.zeros(args.n, dtype=bool)
    for shard_index in range(len(gpus)):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        shard_manifest = json.loads((shard_dir / "manifest.json").read_text())
        require(shard_manifest["checkpoint_audit"]["sha256"] == args._checkpoint_hash, "checkpoint changed across shards; evaluate a stable latest.pt")
        summary = json.loads((shard_dir / "summary.json").read_text())
        require(summary["status"] == "complete", f"shard {shard_index} incomplete")
        require(summary["n"] == args.n and summary["num_shards"] == len(gpus), f"shard {shard_index} metadata mismatch")
        require(summary["shard_index"] == shard_index, f"shard {shard_index} index mismatch")
        ids = split_ids(args.n, len(gpus), shard_index)
        require(summary["local_n"] == len(ids), f"shard {shard_index} local_n mismatch")
        require(summary["shard_ids_sha256"] == array_digest(ids), f"shard {shard_index} id digest mismatch")
        coverage[ids] = True
        if endpoints is None:
            endpoints = list(summary["endpoints"])
        require(endpoints == list(summary["endpoints"]), f"shard {shard_index} endpoint mismatch")
        summaries.append(summary)
    require(bool(coverage.all()), "sharded eval did not cover every global sample id")
    endpoints = endpoints or []

    features = {}
    for name in endpoints:
        combined = np.lib.format.open_memmap(output / (name + "_pool.npy"), mode="w+", dtype=np.float32, shape=(args.n, 2048))
        for shard_index in range(len(gpus)):
            ids = split_ids(args.n, len(gpus), shard_index)
            shard_pool = np.load(shard_root / f"shard_{shard_index:02d}" / (name + "_pool.npy"), mmap_mode="r")
            require(shard_pool.shape == (len(ids), 2048), f"shard {shard_index} {name} pool shape mismatch")
            require(np.isfinite(shard_pool).all(), f"shard {shard_index} {name} pool contains nonfinite values")
            combined[ids] = shard_pool
        combined.flush()
        features[name] = combined

    metrics = {}
    score_fid = args.n > 2048 and not args.smoke
    with tf.Session(config=config) as session, np.load(REFERENCE, allow_pickle=False) as reference:
        evaluator = module.Evaluator(session, batch_size=args.feature_batch)
        ref = module.FIDStatistics(reference["mu"], reference["sigma"])
        for name in endpoints:
            stats = evaluator.compute_statistics(features[name])
            row = dict(feature_mean_l2_to_reference=float(np.linalg.norm(stats.mu - reference["mu"])))
            if score_fid:
                row["fid"] = float(stats.frechet_distance(ref))
                row["inception_score"] = float(evaluator.compute_inception_score(features[name], split_size=max(1, min(5000, args.n))))
            metrics[name] = row
            np.savez_compressed(output / (name + "_statistics.npz"), mu=stats.mu, sigma=stats.sigma)
            atomic_json(output / "partial_metrics.json", metrics)

    label_counts = np.bincount(np.arange(args.n) % 1000, minlength=1000)
    summary = dict(
        status="complete",
        format="bert_sparse2d_sharded_free_fid_v1",
        n=args.n,
        num_shards=len(gpus),
        gpus=gpus,
        smoke=args.smoke,
        state=args.state,
        seed=args.seed,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=args._checkpoint_hash,
        variants=args.variants,
        endpoints=endpoints,
        metrics=metrics,
        complete_coverage=bool(coverage.all()),
        complete_free_generation_fid=score_fid,
        fid_skipped_reason=None if score_fid else "smoke or n<=2048",
        full_class_balance=bool(np.all(label_counts == (args.n // 1000))) if args.n % 1000 == 0 else False,
        direct_replace=True,
        selected_sparse_2d_only=True,
        unified_checkpoint_1d_and_2d=False,
        shard_summaries=summaries,
        runtime_seconds=sum(float(row.get("runtime_seconds", 0.0)) for row in summaries),
        wall_seconds=float(time.monotonic() - args._started),
        mean_k=float(np.mean([row["mean_k"] for row in summaries])),
        feature_hashes={name + "_pool": digest(output / (name + "_pool.npy")) for name in endpoints},
        evaluator=str(EVALUATOR),
        reference=str(REFERENCE),
        graph=str(GRAPH),
        source_hashes={
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
    )
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", dict(status="complete", completed=args.n, n=args.n, num_shards=len(gpus), seconds=summary["wall_seconds"]))
    print(json.dumps({"status": "complete", "output": str(output), "metrics": metrics}, sort_keys=True), flush=True)
    return summary


def main(args: argparse.Namespace) -> None:
    global REFERENCE, GRAPH
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    asset_root = Path(args.assets_root_override or ASSETS).resolve()
    REFERENCE = asset_root / "fid/VIRTUAL_imagenet256_labeled.npz"
    GRAPH = asset_root / "fid/classify_image_graph_def.pb"
    from .assets import FID, check
    args._started = time.monotonic()
    output = output_path(args.output)
    require(output.is_relative_to(RESULT_ROOT.resolve()) and output != RESULT_ROOT.resolve(), "output must be under approved result root")
    require(not output.exists(), "fresh output required")
    max_n = 50000 if args.allow_50k else 5000
    require(1 <= args.n <= max_n, "bounded local5k/explicit 50k only")
    if args.n > 5000:
        require(args.allow_50k and args.n == 50000, "50k evaluation requires --allow-50k and --n 50000")
    if args.smoke:
        require(args.n <= 64, "smoke capped at64")
    else:
        require(args.n >= 4096, "FID screen requires >=4096 samples; use --smoke for smaller tests")
    require(1 <= args.batch <= 32 and 1 <= args.feature_batch <= 32, "batch limits violated")
    gpus = parse_gpus(args.gpus)
    args.variants = args.variants or ["halton_fixed_margin4"]
    output.mkdir(parents=True)
    meta = json.loads((args.checkpoint / "latest.json").read_text())
    args._checkpoint_hash = digest(args.checkpoint / "latest.pt")
    args._checkpoint_step = int(meta["step"])
    require(meta["sha256"] == args._checkpoint_hash, "latest metadata/hash mismatch")
    if args.expected_checkpoint_sha256 is not None:
        require(args._checkpoint_hash == args.expected_checkpoint_sha256, "requested checkpoint was replaced")
    if args.expected_checkpoint_step is not None:
        require(args._checkpoint_step == args.expected_checkpoint_step, "requested checkpoint step changed")
    atomic_json(output / "manifest.json", dict(
        format="bert_sparse2d_sharded_free_fid_v1",
        n=args.n,
        num_shards=len(gpus),
        gpus=gpus,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=args._checkpoint_hash,
        state=args.state,
        seed=args.seed,
        batch=args.batch,
        feature_batch=args.feature_batch,
        feature_chunk=args.feature_chunk,
        assets_root_override=str(args.assets_root_override.resolve()) if args.assets_root_override else None,
        variants=args.variants,
        allow_50k=args.allow_50k,
        smoke=args.smoke,
        launched_at=time.strftime("%Y-%m-%d %H:%M:%S %z"),
    ))
    launch_shards(args, output, gpus)
    # Shards already verified these assets; verify again before merge, outside the ACK wait.
    for spec in FID: check(asset_root / spec["local_path"], spec)
    merge_shards(args, output, gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("raw",), default="raw")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-checkpoint-step", type=int)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--gpus", required=True, help="comma-separated GPU ids, one shard per GPU")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--feature-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--feature-chunk", type=int, default=4)
    parser.add_argument("--assets-root-override", type=Path, help="override checkpoint config assets_root for portable evaluation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--allow-50k", action="store_true")
    main(parser.parse_args())
