"""Remove only explicit, completed test artifacts after retaining identities."""
import json
from pathlib import Path
from bert2d.paths import atomic_json, sha256
from experiments.continuous_halton_embedding_20260923.run_pipeline import ROOT

def main():
    for relative, expected in (
        ('results/adapter_smoke/summary.json', 'complete'),
        ('results/halton_smoke_retry1/complete.json', 'complete'),
        ('results/eval_smoke8/summary.json', 'complete'),
    ):
        assert json.loads((ROOT/relative).read_text())['status'] == expected
    relatives = [
        'results/adapter_smoke/latest.pt',
        'results/halton_smoke_retry1/latest.pt',
        *[f'results/eval_smoke8/first_shard{i}.npz' for i in range(4)],
    ]
    paths = [ROOT/relative for relative in relatives]
    assert all(path.is_file() and not path.is_symlink() for path in paths)
    records = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p)) for p in paths]
    receipt = ROOT/'smoke_cleanup.json'
    assert not receipt.exists()
    atomic_json(receipt, dict(status='planned', files=records, reason='completed smoke; retain metrics and source'))
    for path in paths:
        path.unlink()
    assert all(not path.exists() for path in paths)
    atomic_json(receipt, dict(status='complete', files=records, bytes_removed=sum(r['bytes'] for r in records)))
    print(json.dumps(dict(status='complete', files=len(records), bytes_removed=sum(r['bytes'] for r in records))))

if __name__ == '__main__':
    main()
