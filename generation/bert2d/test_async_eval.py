"""CPU lifecycle/race regression tests; no GPU, weights, or network required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from .async_eval import AsyncEvaluator, all_loaded
from .paths import atomic_json
from .periodic_eval import VARIANT


class Process:
    pid = 99999999
    code = None
    def poll(self): return self.code
    def wait(self, timeout=None): return self.code


class AsyncTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=MagicMock();self.process=Process()
        self.meta=dict(step=5720,sha256='old_hash',config=dict(updates_per_epoch=5720))
        self.launches=[]
        def launch(cmd,**kwargs):
            self.launches.append((cmd,kwargs))
            evaluation=Path(cmd[cmd.index('--output')+1])
            for index in range(2):
                atomic_json(evaluation/'shards'/f'shard_{index:02d}'/'checkpoint_loaded.json',
                            dict(step=int(cmd[cmd.index('--expected-checkpoint-step')+1]),
                                 sha256=cmd[cmd.index('--expected-checkpoint-sha256')+1],state='raw'))
            self.evaluation=evaluation
            return self.process
        self.manager=AsyncEvaluator(self.root,self.root/'assets',['1','3'],self.run,popen=launch)

    def complete(self, digest='old_hash'):
        atomic_json(self.evaluation/'summary.json',dict(status='complete',n=5000,
            complete_coverage=True,complete_free_generation_fid=True,direct_replace=True,
            state='raw',variants=[VARIANT],checkpoint_sha256=digest,
            metrics={'base':{'fid':10.7},VARIANT:{'fid':30.8}}))
        self.process.code=0

    def test_returns_after_load_while_evaluation_still_running(self):
        self.manager.start(self.meta)
        self.assertIsNotNone(self.manager.active)
        self.assertIsNone(self.process.poll())
        self.run.log.assert_not_called()
        self.assertTrue(self.launches[0][1]['start_new_session'])
        self.assertIn('--expected-checkpoint-sha256',self.launches[0][0])
        self.assertEqual(list(self.root.rglob('*.pt')),[])

    def test_later_latest_replacement_does_not_change_result_identity(self):
        self.manager.start(self.meta)
        atomic_json(self.root/'latest.json',dict(step=11440,sha256='new_hash'))
        self.complete();self.manager.poll()
        row=self.run.log.call_args.args[0]
        self.assertEqual(row['eval/epoch'],1)
        self.assertEqual(row['eval/checkpoint_step'],5720)
        self.assertNotIn('step',row);self.assertNotIn('epoch',row)
        self.assertIsNone(self.manager.active)
        self.manager.start(self.meta)
        self.assertEqual(len(self.launches),1)
        self.assertEqual(self.run.log.call_count,1)

    def test_waits_for_every_shard_and_rejects_wrong_hash(self):
        self.manager.start(self.meta)
        identity=self.manager.active['identity']
        path=self.evaluation/'shards/shard_01/checkpoint_loaded.json'
        path.unlink()
        self.assertFalse(all_loaded(self.evaluation,identity))
        atomic_json(path,dict(step=5720,sha256='wrong',state='raw'))
        with self.assertRaisesRegex(ValueError,'wrong checkpoint'):
            all_loaded(self.evaluation,identity)

    def test_failed_worker_stops_and_never_logs_a_result(self):
        self.manager.start(self.meta);self.process.code=2
        with patch('bert2d.async_eval.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'exit code 2'):self.manager.poll()
            kill.assert_called_once()
        self.run.log.assert_not_called()
        self.assertIsNone(self.manager.active)

    def test_finished_but_wrong_summary_hash_is_rejected(self):
        self.manager.start(self.meta);self.complete('wrong')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.manager.poll()
        self.run.log.assert_not_called()

    def test_second_epoch_waits_for_previous_before_launch(self):
        self.manager.start(self.meta)
        self.complete();self.manager.finish()
        self.assertEqual(self.run.log.call_count,1)
        self.assertIsNone(self.manager.active)

    def test_restart_republishes_unlogged_completed_result(self):
        self.manager.start(self.meta);self.complete()
        # Simulate previous trainer's death after the evaluator completed.
        atomic_json(self.root/'async_eval_status.json',dict(status='complete'))
        recovered=AsyncEvaluator(self.root,self.root/'assets',['1','3'],self.run)
        self.assertIsNone(recovered.active)
        self.run.log.assert_called_once()

    def test_timeout_cancels_only_owned_process_group(self):
        self.manager.start(self.meta)
        self.manager.active['started']-=2000
        with patch('bert2d.async_eval.os.killpg') as kill:
            with self.assertRaises(TimeoutError):self.manager.poll()
            self.assertEqual(kill.call_args.args[0],self.process.pid)

    def test_every_epoch_through_40_logs_once_without_snapshots(self):
        for epoch in range(1,41):
            self.process.code=None
            meta=dict(self.meta,step=5720*epoch)
            self.manager.start(meta)
            self.complete();self.manager.poll()
        self.assertEqual(len(self.launches),40)
        self.assertEqual([c.args[0]['eval/epoch'] for c in self.run.log.call_args_list],list(range(1,41)))
        self.assertEqual(list(self.root.rglob('*.pt')),[])

    def test_load_timeout_cancels_instead_of_resuming(self):
        self.manager.load_timeout=-1
        with patch('bert2d.async_eval.all_loaded',return_value=False), patch('bert2d.async_eval.os.killpg') as kill:
            with self.assertRaisesRegex(TimeoutError,'checkpoint-load'):
                self.manager.start(self.meta)
            kill.assert_called_once()
        self.assertIsNone(self.manager.active)
        self.run.log.assert_not_called()

    def test_upload_error_keeps_result_without_success_receipt(self):
        self.manager.start(self.meta);self.complete()
        self.run.log.side_effect=RuntimeError('upload failed')
        with self.assertRaisesRegex(RuntimeError,'upload failed'):self.manager.poll()
        self.assertFalse((self.evaluation/'wandb_logged.json').exists())
        self.assertIsNotNone(self.manager.active)
        with patch('bert2d.async_eval.os.killpg'):self.manager.cancel()

    def test_does_not_start_a_second_live_evaluator_on_restart(self):
        atomic_json(self.root/'async_eval_status.json',dict(status='running',pid=123,process_token='token'))
        with patch('bert2d.async_eval.process_token',return_value='token'):
            with self.assertRaisesRegex(RuntimeError,'still alive'):
                AsyncEvaluator(self.root,self.root/'assets',['1','3'],self.run)


if __name__=='__main__':unittest.main()
