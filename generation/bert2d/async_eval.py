"""One background evaluator, no checkpoint snapshots, trainer-only W&B writer."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .paths import ROOT, atomic_json
from .periodic_eval import VARIANT, metric_row, read_json


def process_token(pid):
    """Linux process start time prevents confusing a recycled PID with our job."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def all_loaded(evaluation, identity):
    for index in range(len(identity["gpus"])):
        path = evaluation / "shards" / f"shard_{index:02d}" / "checkpoint_loaded.json"
        if not path.exists():
            return False
        row = read_json(path)
        if (row.get("sha256") != identity["checkpoint_sha256"] or
                row.get("step") != identity["step"] or row.get("state") != "raw"):
            raise ValueError("async shard loaded the wrong checkpoint identity")
    return True


class AsyncEvaluator:
    def __init__(self, output, assets_root, gpus, run, *, batch=8, seed=20260914,
                 mode="online", load_timeout=600, timeout=1800, popen=None):
        self.output, self.assets_root = Path(output), Path(assets_root)
        self.gpus, self.run, self.batch, self.seed, self.mode = list(gpus), run, batch, seed, mode
        self.load_timeout, self.timeout = load_timeout, timeout
        self.popen = popen or subprocess.Popen
        self.active = None
        self.state_path = self.output / "async_eval_status.json"
        if self.state_path.exists():
            state = read_json(self.state_path)
            if (state.get("status") in ("loading", "running") and state.get("process_token")
                    and process_token(state.get("pid")) == state["process_token"]):
                raise RuntimeError("a previous evaluator is still alive; wait for it before restarting")
        # Recover completed results after a trainer restart without reopening W&B.
        for path in sorted((self.output / "evaluations").glob("epoch*_attempt*/evaluation.json")):
            if (path.parent / "summary.json").exists():
                self.publish(path.parent, read_json(path))

    def publish(self, evaluation, identity):
        summary = read_json(evaluation / "summary.json")
        if summary.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
            raise ValueError("async result checkpoint hash mismatch")
        row = metric_row(summary, identity["step"], identity["epoch"])
        receipt = evaluation / "wandb_logged.json"
        if receipt.exists():
            old = read_json(receipt)
            if old.get("mode") == self.mode and old.get("checkpoint_sha256") == identity["checkpoint_sha256"]:
                return
        # Never rewind the training step/epoch axes when a delayed result arrives.
        row["eval/epoch"] = row.pop("epoch")
        row["eval/checkpoint_step"] = row.pop("step")
        if self.mode != "disabled":
            if self.run is None:
                raise RuntimeError("async W&B logging requires the trainer's active run")
            self.run.log(row)
        atomic_json(receipt, dict(mode=self.mode, checkpoint_sha256=identity["checkpoint_sha256"], metrics=row))
        print(json.dumps(dict(stage="async_eval_result", **row)), flush=True)

    def start(self, meta):
        self.finish()  # Backpressure: no overlapping evaluations or skipped epochs.
        identity = dict(epoch=int(meta["step"]) // int(meta["config"]["updates_per_epoch"]),
                        step=meta["step"], checkpoint_sha256=meta["sha256"], n=5000,
                        seed=self.seed, gpus=self.gpus, batch=self.batch,
                        feature_batch=self.batch, variants=[VARIANT], state="raw")
        root = self.output / "evaluations"
        root.mkdir(exist_ok=True)
        for previous in sorted(root.glob(f"epoch{identity['epoch']:03d}_attempt*")):
            if ((previous / "evaluation.json").exists() and (previous / "summary.json").exists()
                    and read_json(previous / "evaluation.json") == identity):
                self.publish(previous, identity)
                return
        attempt = 1
        while (root / f"epoch{identity['epoch']:03d}_attempt{attempt:02d}").exists():
            attempt += 1
        evaluation = root / f"epoch{identity['epoch']:03d}_attempt{attempt:02d}"
        command = [sys.executable, "-m", "bert2d.eval_sharded", "--checkpoint", str(self.output),
                   "--output", str(evaluation), "--assets-root-override", str(self.assets_root),
                   "--gpus", ",".join(self.gpus), "--n", "5000", "--batch", str(self.batch),
                   "--feature-batch", str(self.batch), "--seed", str(self.seed),
                   "--expected-checkpoint-sha256", identity["checkpoint_sha256"],
                   "--expected-checkpoint-step", str(identity["step"]), "--variants", VARIANT]
        env = dict(os.environ, USE_TF="0")
        proc = self.popen(command, cwd=ROOT, env=env, start_new_session=True)
        self.active = dict(proc=proc, evaluation=evaluation, identity=identity, started=time.monotonic())
        atomic_json(self.state_path, dict(status="loading", pid=proc.pid, process_token=process_token(proc.pid),
                    evaluation=str(evaluation), **identity))
        # Trainer is blocked only until EVERY shard owns an independent model.
        try:
            while True:
                if evaluation.exists():
                    atomic_json(evaluation / "evaluation.json", identity)
                self.poll()
                if self.active is None:
                    return
                if all_loaded(evaluation, identity):
                    atomic_json(self.state_path, dict(status="running", pid=proc.pid,
                                process_token=process_token(proc.pid), evaluation=str(evaluation), **identity))
                    return
                if time.monotonic() - self.active["started"] > self.load_timeout:
                    raise TimeoutError("async evaluator checkpoint-load timeout")
                time.sleep(.25)
        except BaseException:
            self.cancel()
            raise

    def poll(self):
        if self.active is None:
            return
        active = self.active
        code = active["proc"].poll()
        if code is None:
            if time.monotonic() - active["started"] > self.timeout:
                self.cancel()
                raise TimeoutError("async evaluator runtime timeout")
            return
        if code != 0:
            self.cancel()
            raise RuntimeError(f"async evaluator failed with exit code {code}; see {active['evaluation']}")
        self.publish(active["evaluation"], active["identity"])
        atomic_json(self.state_path, dict(status="complete", evaluation=str(active["evaluation"]), **active["identity"]))
        self.active = None

    def finish(self):
        while self.active is not None:
            self.poll()
            if self.active is not None:
                time.sleep(.25)

    def cancel(self):
        if self.active is None:
            return
        active = self.active
        # Only the process group created by this controller; never other jobs.
        try:
            os.killpg(active["proc"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            active["proc"].wait(timeout=15)
        except subprocess.TimeoutExpired:
            # No SIGKILL. Leave an explicit record; block restart while still alive.
            atomic_json(self.state_path, dict(status="running", cancellation_failed=True,
                        pid=active["proc"].pid, process_token=process_token(active["proc"].pid)))
            raise RuntimeError("evaluator did not terminate; inspect the recorded owned PID")
        atomic_json(self.state_path, dict(status="failed", evaluation=str(active["evaluation"]), **active["identity"]))
        self.active = None
