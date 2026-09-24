"""FIFO evaluation of immutable hard-linked snapshots; training never waits for FID."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from bert2d.paths import atomic_json


class AsyncTopPEval:
    def __init__(self,out,gpu,n=5000,cfg_w=2.75,mapper_checkpoint=None,
                 scorer_checkpoint=None,top_p_checkpoint=None,stage2_steps=8):
        self.out=Path(out);self.gpu=str(gpu);self.n=n;self.cfg_w=float(cfg_w)
        self.mapper_checkpoint=mapper_checkpoint
        self.scorer_checkpoint=scorer_checkpoint
        self.top_p_checkpoint=top_p_checkpoint
        self.stage2_steps=int(stage2_steps)
        self.pending=[];self.active=None
        self.next_admission_check=0.

    def available_gpu(self):
        """Defer evaluation only, not training, if its extra headroom is absent."""
        if time.monotonic()<self.next_admission_check: return None
        self.next_admission_check=time.monotonic()+30
        allowed=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if self.gpu=='auto' else [self.gpu]
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                          '--format=csv,noheader,nounits'],text=True,timeout=5)
            cards=[]
            for line in text.splitlines():
                index,uuid,free=(part.strip() for part in line.split(','))
                if index in allowed or uuid in allowed: cards.append((int(free),uuid))
            best=max(cards,default=(0,None))
            chosen=best[1] if best[0]>=20*1024 else None
            atomic_json(self.out/'eval_admission.json',dict(status='ready' if chosen else 'waiting_for_memory',
                extra_free_mib_required=20*1024,best_free_mib=best[0],gpu=chosen,pending=len(self.pending)))
            return chosen
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            atomic_json(self.out/'eval_admission.json',dict(status='waiting_for_gpu_query',error=repr(exc)))
            return None

    def enqueue(self,step,epoch):
        snapshot=self.out/f'eval_pending_step{step}.pt'
        os.link(self.out/'latest.pt',snapshot)
        identity=json.loads((self.out/'latest.json').read_text())
        job=dict(step=step,epoch=epoch,checkpoint=str(snapshot),output=str(self.out/f'eval_step{step}'),
                 checkpoint_sha256=identity['sha256'])
        atomic_json(self.out/f'eval_request_step{step}.json',job)
        self.pending.append(job)

    def poll(self):
        results=[]
        if self.active is not None:
            process,job,log=self.active
            code=process.poll()
            if code is None: return results
            log.close()
            summary=Path(job['output'])/'summary.json'
            result=json.loads(summary.read_text()) if code==0 and summary.exists() else dict(status='failed',exit_code=code)
            if result.get('status')=='complete' and result.get('checkpoint_sha256')!=job['checkpoint_sha256']:
                result=dict(status='failed',error='checkpoint identity mismatch')
            result.update(step=job['step'],epoch=job['epoch'])
            # Snapshot no longer in use on process exit. Retain metrics and exact hash,
            # not growing collections of model/optimizer snapshots, even on failure.
            meta=json.loads((self.out/f'eval_request_step{job["step"]}.json').read_text())
            result['request']=meta
            atomic_json(self.out/f'eval_receipt_step{job["step"]}.json',result)
            Path(job['checkpoint']).unlink()
            results.append(result);self.active=None
        if self.active is None and self.pending:
            gpu=self.available_gpu()
            if gpu is None: return results
            job=self.pending.pop(0)
            job['physical_gpu']=gpu
            atomic_json(self.out/f'eval_request_step{job["step"]}.json',job)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'):
                env.pop(key,None)
            env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
            command=[sys.executable,str(Path(__file__).with_name('eval_top_p_worker.py')),
                     '--checkpoint',job['checkpoint'],'--output',job['output'],'--n',str(self.n),
                     '--cfg-w',str(self.cfg_w),'--stage2-steps',str(self.stage2_steps),
                     '--workers',os.environ.get('MOTAR_EVAL_WORKERS','8')]
            if self.mapper_checkpoint is not None:
                command.extend(('--mapper-checkpoint',str(self.mapper_checkpoint)))
            if self.scorer_checkpoint is not None:
                command.extend(('--scorer-checkpoint',str(self.scorer_checkpoint)))
            if self.top_p_checkpoint is not None:
                command.extend(('--top-p-checkpoint',str(self.top_p_checkpoint)))
            log=(self.out/f'eval_step{job["step"]}.log').open('w')
            try: process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:
                log.close();self.pending.insert(0,job);raise
            self.active=(process,job,log)
        return results

    @property
    def busy(self): return self.active is not None or bool(self.pending)
