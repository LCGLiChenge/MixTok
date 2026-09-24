"""CPU tests for cadence, recovery, latest-only and W&B metric axes."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .paths import atomic_json, sha256
from .periodic_eval import VARIANT, metric_row, run_schedule, upload


def summary(digest):
    return dict(status="complete", n=5000, complete_coverage=True,
                complete_free_generation_fid=True, direct_replace=True,
                state="raw", variants=[VARIANT], checkpoint_sha256=digest,
                metrics={"base": {"fid": 10.5}, VARIANT: {"fid": 9.5}})


class PeriodicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "run"
        self.args = SimpleNamespace(output=self.out, assets_root=Path(self.tmp.name),
            epochs=40, eval_every=2, global_batch=2562334, eval_seed=20260914,
            eval_batch=8, wandb_mode="disabled")
        self.trained, self.evaluated = [], []

    def fake_runner(self, cmd):
        def value(flag): return cmd[cmd.index(flag) + 1]
        if "--stop-after-epoch" in cmd:
            epoch = int(value("--stop-after-epoch"))
            self.trained.append(epoch)
            self.out.mkdir(exist_ok=True)
            (self.out / "latest.pt").write_bytes(str(epoch).encode())
            atomic_json(self.out / "latest.json", dict(step=epoch,
                sha256=sha256(self.out / "latest.pt"), config={"updates_per_epoch": 1}))
            atomic_json(self.out / "summary.json", {"status": "awaiting_eval", "step": epoch})
        elif "bert2d.eval_sharded" in cmd:
            self.evaluated.append(int(json.loads((self.out / "latest.json").read_text())["step"]))
            evaluation = Path(value("--output"))
            evaluation.mkdir()
            atomic_json(evaluation / "summary.json", summary(sha256(self.out / "latest.pt")))
        else:
            upload(self.out, Path(value("--upload")), "disabled")

    def test_40_epochs_evaluates_20_times_and_keeps_only_latest(self):
        run_schedule(self.args, ["train"], {}, ["0", "1"], runner=self.fake_runner)
        self.assertEqual(self.trained, list(range(2, 41, 2)))
        self.assertEqual(self.evaluated, self.trained)
        self.assertEqual(list(self.out.rglob("*.pt")), [self.out / "latest.pt"])
        run_schedule(self.args, ["train"], {}, ["0", "1"], runner=self.fake_runner)
        self.assertEqual(len(self.evaluated), 20)  # Reuse completed matching FID.

    def test_bounded_test_stops_after_one_eval_without_changing_full_target(self):
        from .launch import schedule_arguments
        self.args.test_epochs=1
        self.args.eval_every=1
        limited=schedule_arguments(self.args)
        self.assertEqual(self.args.epochs,40)
        self.assertEqual(limited.epochs,1)
        commands=[]
        def runner(cmd):
            commands.append(cmd)
            self.fake_runner(cmd)
        run_schedule(limited,["train","--epochs","40"],{},["0"],runner=runner)
        self.assertEqual(self.trained,[1])
        self.assertEqual(self.evaluated,[1])
        self.assertEqual(commands[0][commands[0].index("--epochs")+1],"40")
        run_schedule(limited,["train","--epochs","40"],{},["0"],runner=runner)
        self.assertEqual(self.trained,[1])
        self.args.test_epochs=41
        with self.assertRaises(ValueError):schedule_arguments(self.args)

    def test_failed_eval_blocks_next_training_and_retries_fresh_path(self):
        def fail(cmd):
            if "bert2d.eval_sharded" in cmd:
                Path(cmd[cmd.index("--output") + 1]).mkdir()
                raise RuntimeError("simulated eval failure")
            self.fake_runner(cmd)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run_schedule(self.args, ["train"], {}, ["0"], runner=fail)
        self.assertEqual(self.trained, [2])
        run_schedule(self.args, ["train"], {}, ["0"], runner=self.fake_runner)
        self.assertEqual(self.trained, list(range(2, 41, 2)))
        self.assertTrue((self.out / "evaluations/epoch002_attempt02/summary.json").exists())

    def test_metrics_reject_partial_or_nan(self):
        row = summary("x")
        self.assertEqual(metric_row(row, 10, 2)["eval/fid5k_full_minus_base"], -1.)
        row["complete_coverage"] = False
        with self.assertRaises(ValueError): metric_row(row, 10, 2)
        row = summary("x")
        row["metrics"][VARIANT]["fid"] = float("nan")
        with self.assertRaises(ValueError): metric_row(row, 10, 2)

    def test_upload_same_wandb_run_without_duplicate_explicit_step(self):
        self.out.mkdir()
        evaluation = self.out / "eval"
        evaluation.mkdir()
        atomic_json(self.out / "wandb_run.json", {"id": "test-run", "project": "test-project"})
        atomic_json(evaluation / "evaluation.json", {"checkpoint_sha256": "x", "step": 11440, "epoch": 2})
        atomic_json(evaluation / "summary.json", summary("x"))
        sdk = MagicMock()
        run = sdk.init.return_value.__enter__.return_value
        with patch.dict("sys.modules", {"wandb": sdk}):
            upload(self.out, evaluation, "online")
        self.assertEqual(sdk.init.call_args.kwargs["id"], "test-run")
        self.assertEqual(sdk.init.call_args.kwargs["resume"], "must")
        self.assertEqual(run.log.call_args.kwargs, {})
        self.assertEqual(run.log.call_args.args[0]["epoch"], 2)
        run.define_metric.assert_any_call("eval/*", step_metric="eval/epoch")


if __name__ == "__main__":
    unittest.main()
