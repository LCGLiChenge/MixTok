"""Read-only process/progress checks; writes only monitoring receipts, never kills."""
import json
import time
from pathlib import Path
from bert2d.paths import atomic_json
from experiments.continuous_halton_embedding_20260923.run_pipeline import RESULTS

def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None

def live_command(pid):
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0]
        cmd = Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        return stat != 'Z' and 'continuous_halton_embedding_20260923' in cmd
    except (OSError, ValueError, TypeError):
        return False

def snapshot(root=RESULTS):
    state = read_json(root/'pipeline_status.json') or {}
    phase = state.get('phase', 'unknown')
    record = dict(time=time.time(), pipeline_status=state.get('status'),
                  phase=phase, parent_live=live_command(state.get('pid')),
                  child_live=live_command(state.get('child_pid')))
    failure = read_json(root/'pipeline_failure.json')
    record['failure'] = failure
    if phase == 'adapter':
        folder = root/'adapter20k_seed20260918'
        ranks = [read_json(folder/f'status_rank{i}.json') for i in range(8)]
        ranks = [r for r in ranks if r]
        record['steps'] = [r.get('step') for r in ranks]
        record['rank_processes_live'] = [live_command(r.get('pid')) for r in ranks]
        evals = sorted(folder.glob('eval_step*.json'), key=lambda p:int(p.stem[9:]))
        record['latest_eval'] = read_json(evals[-1]) if evals else None
        record['checkpoint'] = read_json(folder/'latest.json')
        status = folder/'status_rank0.json'
    elif phase == 'halton':
        folder = root/'halton20k_seed0'
        ranks = [read_json(folder/f'rank{i}.json') for i in range(8)]
        record['steps'] = [r.get('step') for r in ranks if r]
        receipts = sorted(folder.glob('eval_receipt_step*.json'),
                          key=lambda p:int(p.stem[17:]))
        record['latest_eval'] = read_json(receipts[-1]) if receipts else None
        status = folder/'rank0.json'
    else:
        folder = root/'cfg_sweep_50k'
        record['sweep_progress'] = read_json(folder/'sweep_progress.json')
        record['final_50k_status'] = read_json(folder/'final_50k/status.json')
        status = folder/'sweep_progress.json'
    record['progress_age_seconds'] = time.time()-status.stat().st_mtime if status.exists() else None
    record['advisories'] = []
    if failure:
        record['advisories'].append('pipeline failure receipt exists; inspect before any restart')
    if state.get('status') != 'complete' and not record['parent_live']:
        record['advisories'].append('pipeline parent missing or identity differs')
    age = record['progress_age_seconds']
    if phase in ('adapter','halton') and age is not None and age > 900:
        record['advisories'].append('no training status update for 15min; may be eval or a stall')
    # Completion is only reported as observed, not used to mark the research goal done.
    record['completion_observed'] = state.get('status') == 'complete'
    return record

def main():
    last_journal = 0
    previous_phase = None
    while True:
        try:
            row = snapshot()
        except (OSError, ValueError, KeyError) as error:
            row = dict(time=time.time(), observation_error=repr(error))
        atomic_json(RESULTS/'monitor_status.json',row)
        event = (row.get('phase') != previous_phase or row.get('advisories')
                 or row.get('observation_error') or row.get('completion_observed')
                 or time.monotonic()-last_journal >= 3600)
        if event:
            with (RESULTS/'monitor_events.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            last_journal = time.monotonic()
        previous_phase = row.get('phase')
        if row.get('completion_observed'):
            return
        time.sleep(60)

if __name__ == '__main__':
    main()
