"""Sequential 8-H20 adapter -> Halton -> fresh CFG sweep -> 50k gFID."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUNTIME=Path(__file__).resolve().parents[2]
ROOT=RUNTIME.parent
sys.path.insert(0,str(RUNTIME))
from bert2d.paths import atomic_json

ASSETS=Path(os.environ.get('MOTAR_ASSETS', ROOT/'assets/generation')).resolve()
RESULTS=Path(os.environ.get('MOTAR_RESULTS', ROOT/'results/generation')).resolve()
TOPP=Path(os.environ.get('TOP_P_CKPT', ROOT/'assets/reconstruction/epoch_0006_step_00247482.pt')).resolve()
SCORER=Path(os.environ.get('TOP_P_SCORER_CKPT', ASSETS/'top_p_scorer/latest.pt')).resolve()
UPSTREAM=Path(os.environ.get('HALTON_UPSTREAM_ROOT', ROOT/'Halton-MaskGIT/source')).resolve()
OFFICIAL=Path(os.environ.get('HALTON_OFFICIAL_CKPT', ASSETS/'ImageNet_256_base.pth')).resolve()
GT2D=Path(os.environ.get('TOP_P_GT2D_CACHE', ASSETS/'gt2d_top_p_train')).resolve()
TEACHER=Path(os.environ.get('TOP_P_TEACHER_PROXY_CACHE', ASSETS/'teacher_proxy_top_p_train')).resolve()
ROUTES=Path(os.environ.get('TOP_P_ROUTE_CACHE', ASSETS/'top_p_continuous_route_cache')).resolve()
PACKAGE='experiments.continuous_halton_embedding_20260923'


def env():
    e=os.environ.copy()
    e.update(MOTAR_ASSETS=str(ASSETS), MOTAR_RESULTS=str(RESULTS),
        PYTHONPATH=str(RUNTIME)+(os.pathsep+e['PYTHONPATH'] if e.get('PYTHONPATH') else ''))
    for key,value in (('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7'),
                      ('OMP_NUM_THREADS','4'),('OPENBLAS_NUM_THREADS','4'),
                      ('MOTAR_EVAL_WORKERS','8'),('MOTAR_EVAL_ALLOWED_GPUS','0,1,2,3,4,5,6,7')):
        e.setdefault(key,value)
    return e


def adapter_command(output,smoke=False):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        '-m',PACKAGE+'.train_adapter','--source-root',str(GT2D),
        '--teacher-root',str(TEACHER),'--top-p-ckpt',str(TOPP),
        '--pretrained',str(OFFICIAL),'--output',str(output),'--assets-root',str(ASSETS)]
    if smoke:
        command+=['--smoke','--stop-after','3','--batch','1','--eval-batch','1',
            '--eval-sources','8','--min-free-gib','4','--memory-fraction','.055']
    return command


def halton_command(output,adapter,smoke=False,skip_fid=False):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        '-m',PACKAGE+'.train_halton','--upstream-root',str(UPSTREAM),'--pretrained',str(OFFICIAL),
        '--source-root',str(GT2D),'--cache-root',str(ROUTES),
        '--top-p-ckpt',str(TOPP),'--adapter-ckpt',str(adapter),'--scorer-ckpt',str(SCORER),
        '--output',str(output),'--eval-stage2-steps','8','--eval-cfg-w','3.375']
    if smoke:
        command+=['--smoke','--updates','3','--micro','1','--min-free-gib','4',
                  '--memory-fraction','.06','--adapter-step','3']
        if skip_fid:
            command+=['--skip-smoke-fid']
    else:
        command+=['--stop-after','20000','--micro','32']
    return command


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    environment=env()
    os.environ.update(environment)
    adapter=RESULTS/'adapter20k_seed20260918'
    halton=RESULTS/'halton20k_seed0'
    sweep=RESULTS/'cfg_sweep_50k'
    phases=[('adapter',adapter_command(adapter),adapter/'summary.json'),
        ('halton',halton_command(halton,adapter/'latest.pt'),halton/'complete.json'),
        ('cfg_sweep_50k',[sys.executable,'-m',PACKAGE+'.sweep_cfg',
            '--checkpoint',str(halton/'latest.pt'),'--adapter-checkpoint',str(adapter/'latest.pt'),
            '--scorer-checkpoint',str(SCORER),'--top-p-checkpoint',str(TOPP),
            '--output',str(sweep)],sweep/'summary.json')]
    started=time.time()
    for phase,command,receipt in phases:
        if receipt.exists():
            result=json.loads(receipt.read_text())
            if result.get('status')!='complete' or (phase!='cfg_sweep_50k' and result.get('step')!=20000):
                raise RuntimeError('existing phase receipt is not complete')
            continue
        child=subprocess.Popen(command,cwd=RUNTIME,env=environment)
        atomic_json(RESULTS/'pipeline_status.json',dict(status='running',phase=phase,
            pid=os.getpid(),child_pid=child.pid,command=command,started=started))
        code=child.wait()
        if code!=0:
            raise RuntimeError(f'{phase} exited with code {code}')
        result=json.loads(receipt.read_text())
        if result.get('status')!='complete' or (phase!='cfg_sweep_50k' and result.get('step')!=20000):
            raise RuntimeError(f'{phase} completion mismatch')
    atomic_json(RESULTS/'pipeline_status.json',dict(status='complete',pid=os.getpid(),started=started,
        seconds=time.time()-started,final=json.loads((sweep/'summary.json').read_text())))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RESULTS/'pipeline_failure.json',dict(error=repr(error),pid=os.getpid()))
        raise
