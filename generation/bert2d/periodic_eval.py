"""Serialized epoch-boundary FID; no checkpoint copies, one W&B run."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from .paths import ROOT, atomic_json, sha256

VARIANT = "halton_fixed_margin4"


def read_json(path):
    return json.loads(Path(path).read_text())


def define_axes(run):
    # W&B's internal history counter differs from optimizer steps after evals.
    run.define_metric("step")
    run.define_metric("epoch")
    run.define_metric("*", step_metric="step")
    run.define_metric("eval/epoch")
    run.define_metric("eval/*", step_metric="eval/epoch")


def metric_row(summary, step, epoch):
    if not (summary["status"] == "complete" and summary["n"] == 5000
            and summary["complete_coverage"] and summary["complete_free_generation_fid"]
            and summary["direct_replace"] and summary["state"] == "raw"
            and summary["variants"] == [VARIANT]):
        raise ValueError("not a complete 5k full-replacement FID evaluation")
    metrics = summary["metrics"]
    base, full = float(metrics["base"]["fid"]), float(metrics[VARIANT]["fid"])
    if not (math.isfinite(base) and math.isfinite(full)):
        raise ValueError("nonfinite FID")
    row = {"step": int(step), "epoch": int(epoch), "eval/num_samples": 5000,
           "eval/fid5k_base": base, "eval/fid5k_full": full,
           "eval/fid5k_full_minus_base": full - base}
    for name, endpoint in (("base", "base"), ("full", VARIANT)):
        if "inception_score" in metrics[endpoint]:
            value = float(metrics[endpoint]["inception_score"])
            if not math.isfinite(value):
                raise ValueError("nonfinite Inception Score")
            row["eval/is5k_" + name] = value
    return row


def upload(output, evaluation, mode):
    record = read_json(evaluation / "evaluation.json")
    summary = read_json(evaluation / "summary.json")
    if summary["checkpoint_sha256"] != record["checkpoint_sha256"]:
        raise ValueError("evaluation checkpoint identity mismatch")
    row = metric_row(summary, record["step"], record["epoch"])
    row["eval/epoch"] = row["epoch"]
    row["eval/checkpoint_step"] = row["step"]
    if mode != "disabled":
        import wandb
        identity = read_json(output / "wandb_run.json")
        with wandb.init(project=identity["project"], id=identity["id"],
                        resume="must" if mode == "online" else "allow",
                        dir=str(output), mode=mode) as run:
            define_axes(run)
            run.log(row)  # Do not reuse an already committed explicit W&B step.
    atomic_json(evaluation / "wandb_logged.json", {"mode": mode, "metrics": row,
                "checkpoint_sha256": record["checkpoint_sha256"]})


def run_schedule(args, train_command, env, gpus, runner=None, evaluator=None):
    """Wait for every child to exit before the next train/eval phase starts."""
    runner = runner or (lambda cmd: subprocess.run(cmd, cwd=ROOT, env=env, check=True))
    evaluator = evaluator or evaluate_boundary
    out = args.output.resolve()
    while True:
        meta = read_json(out / "latest.json") if (out / "latest.pt").exists() else None
        step = int(meta["step"]) if meta else 0
        per_epoch = int(meta["config"]["updates_per_epoch"]) if meta else math.ceil(2562334 / args.global_batch)
        target = args.epochs * per_epoch
        if step > target:
            raise ValueError("checkpoint is beyond the requested total epochs")
        if step and step % per_epoch == 0 and (step // per_epoch) % args.eval_every == 0:
            evaluator(args, meta, env, gpus, runner)
        if step == target:
            atomic_json(out / "pipeline_status.json", {"status": "complete", "step": step,
                        "epochs": args.epochs, "eval_every_epochs": args.eval_every})
            return
        boundary = min(args.epochs, (step // (per_epoch * args.eval_every) + 1) * args.eval_every)
        cmd = list(train_command) + ["--stop-after-epoch", str(boundary)]
        if meta:
            cmd += ["--resume", str(out)]
        if out.exists():
            atomic_json(out / "pipeline_status.json", {"status": "training", "step": step, "next_boundary_epoch": boundary})
        runner(cmd)
        report = read_json(out / "summary.json")
        if report["status"] == "interrupted":
            raise RuntimeError("training was interrupted; not restarting it automatically")
        if report["step"] != boundary * per_epoch:
            raise RuntimeError("training did not reach its requested evaluation boundary")


def evaluate_boundary(args, meta, env, gpus, runner):
    out = args.output.resolve()
    epoch = int(meta["step"]) // int(meta["config"]["updates_per_epoch"])
    identity = dict(epoch=epoch, step=meta["step"], checkpoint_sha256=meta["sha256"],
                    n=5000, seed=args.eval_seed, gpus=gpus, batch=args.eval_batch,
                    feature_batch=args.eval_batch, variants=[VARIANT], state="raw")
    if sha256(out / "latest.pt") != meta["sha256"]:
        raise ValueError("latest checkpoint changed before periodic evaluation")
    root = out / "evaluations"
    root.mkdir(exist_ok=True)
    evaluation = None
    for previous in sorted(root.glob(f"epoch{epoch:03d}_attempt*")):
        if not (previous / "evaluation.json").exists():
            continue
        if read_json(previous / "evaluation.json") == identity and (previous / "summary.json").exists():
            summary = read_json(previous / "summary.json")
            metric_row(summary, meta["step"], epoch)
            if summary["checkpoint_sha256"] != meta["sha256"]:
                raise ValueError("stored FID checkpoint mismatch")
            evaluation = previous
            break
    if evaluation is None:
        attempt = 1
        while (root / f"epoch{epoch:03d}_attempt{attempt:02d}").exists():
            attempt += 1
        evaluation = root / f"epoch{epoch:03d}_attempt{attempt:02d}"
        # eval_sharded requires a fresh output path, so store pending identity outside it.
        atomic_json(out / "pipeline_status.json", dict(status="evaluating", **identity))
        cmd = [sys.executable, "-m", "bert2d.eval_sharded", "--checkpoint", str(out),
               "--output", str(evaluation), "--assets-root-override", str(args.assets_root),
               "--gpus", ",".join(gpus), "--n", "5000", "--batch", str(args.eval_batch),
               "--feature-batch", str(args.eval_batch), "--seed", str(args.eval_seed),
               "--variants", VARIANT]
        try:
            runner(cmd)
        finally:
            # Keep partial attempts and their provenance; never overwrite failed evidence.
            if evaluation.exists():
                atomic_json(evaluation / "evaluation.json", identity)
    if sha256(out / "latest.pt") != meta["sha256"]:
        raise ValueError("latest checkpoint changed during periodic evaluation")
    receipt = evaluation / "wandb_logged.json"
    if not receipt.exists() or read_json(receipt)["mode"] != args.wandb_mode:
        # A separate process avoids retaining W&B/torch/TF state across phases.
        runner([sys.executable, "-m", "bert2d.periodic_eval", "--upload", str(evaluation),
                "--output", str(out), "--wandb-mode", args.wandb_mode])
    atomic_json(out / "pipeline_status.json", dict(status="evaluated", evaluation=str(evaluation), **identity))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    a = p.parse_args()
    upload(a.output, a.upload, a.wandb_mode)
