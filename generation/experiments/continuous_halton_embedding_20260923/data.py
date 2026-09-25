"""Route-only cache contract for continuous embedding generation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from bert2d.paths import sha256
from experiments.continuous_halton_embedding_20260923.bucket_sampler import RankDynamicBucketSampler
from experiments.continuous_halton_embedding_20260923.assets_top_p import TOP_P_SHA256, verify_source_meta

FORMAT = 'top_p_continuous_route_cache_v1'
NUM_SOURCES, NUM_AUG = 1281167,2


def verify_source_file(config, root, key, filename):
    """Bind training inputs to the route configuration or a legacy source manifest."""
    expected = config.get(key)
    if expected is None:
        manifest = root / 'manifest.sha256'
        if not manifest.is_file():
            raise RuntimeError(f'{filename} has no recorded source hash')
        entries = {}
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            entries[name.lstrip('*')] = digest
        expected = entries.get(filename)
        if expected is None:
            raise RuntimeError(f'{filename} is missing from source manifest')
    if sha256(root / filename) != expected:
        raise RuntimeError(f'{filename} hash differs from route source')


class ContinuousRouteDataset(Dataset):
    def __init__(self, source_root, gt_root, cache_root):
        source_root,gt_root,cache_root = map(Path,(source_root,gt_root,cache_root))
        config = json.loads((cache_root/'config.json').read_text())
        summary = json.loads((cache_root/'summary.json').read_text())
        meta = json.loads((gt_root/'meta.json').read_text())
        verify_source_meta(meta, gt_root)
        if (config['format'] != FORMAT or summary['status'] != 'complete'
            or config['top_p_checkpoint_sha256'] != TOP_P_SHA256
            or summary['config_sha256'] != sha256(cache_root/'config.json')
            or not meta.get('completed') or meta.get('mot_step') != 247482
            or meta.get('mot_state_key') != 'model_ema'):
            raise RuntimeError('route/source provenance mismatch')
        if sha256(gt_root/'meta.json') != config['source_meta_sha256']:
            raise RuntimeError('GT source metadata differs from route cache')
        if sha256(source_root/'titok_codes.npy') != config['source_titok_sha256']:
            raise RuntimeError('1D cache identity mismatch')
        verify_source_file(config, gt_root, 'source_gt2d_sha256', 'llamagen_codes.npy')
        verify_source_file(config, source_root, 'source_labels_sha256', 'labels.npy')
        verify_source_file(config, gt_root, 'source_labels_sha256', 'labels.npy')
        for name, expected in (('route_indices.npy', 'route_indices_sha256'),
                               ('route_k.npy', 'route_k_sha256')):
            if sha256(cache_root / name) != summary.get(expected):
                raise RuntimeError(f'{name} hash differs from finalized route cache')
        self.z1d = np.load(source_root/'titok_codes.npy',mmap_mode='r')
        self.gt2d = np.load(gt_root/'llamagen_codes.npy',mmap_mode='r')
        self.labels = np.load(source_root/'labels.npy',mmap_mode='r')
        self.indices = np.load(cache_root/'route_indices.npy',mmap_mode='r')
        self.route_k = np.load(cache_root/'route_k.npy',mmap_mode='r')
        if (self.z1d.shape != (NUM_SOURCES,NUM_AUG,32) or self.gt2d.shape != (NUM_SOURCES,NUM_AUG,256)
            or self.indices.shape != (NUM_SOURCES*NUM_AUG,256) or self.route_k.shape != (NUM_SOURCES*NUM_AUG,)
            or self.labels.shape != (NUM_SOURCES,)):
            raise RuntimeError('cache shape mismatch')
        if not all(a.dtype==np.uint16 for a in (self.z1d,self.gt2d,self.indices,self.route_k)):
            raise RuntimeError('cache dtype mismatch')
        if not np.array_equal(self.labels,np.load(gt_root/'labels.npy',mmap_mode='r')):
            raise RuntimeError('label alignment differs')
        self.k_values = np.asarray(self.route_k).copy()
        if not np.all((self.k_values>=1)&(self.k_values<=256)):
            raise RuntimeError('invalid K')

    def __len__(self):
        return NUM_SOURCES*NUM_AUG

    def __getitem__(self,flat):
        source,view=divmod(int(flat),NUM_AUG)
        k=int(self.route_k[flat])
        indices=np.array(self.indices[flat,:k],dtype=np.int64,copy=True)
        if (np.any(indices<0) or np.any(indices>=256) or np.any(indices[1:]<=indices[:-1])):
            raise RuntimeError('invalid route coordinates')
        return dict(z1d=torch.from_numpy(np.array(self.z1d[source,view],dtype=np.int64)),
            z2d=torch.from_numpy(np.array(self.gt2d[source,view,indices],dtype=np.int64)),
            route_indices=torch.from_numpy(indices),k=torch.tensor(k),
            label=torch.tensor(int(self.labels[source])),source_index=torch.tensor(source),augmentation=torch.tensor(view))


def collate_continuous(rows):
    if any(int(row['k']) != int(rows[0]['k']) for row in rows):
        raise RuntimeError('mixed K in batch')
    result={key:torch.stack([r[key] for r in rows]) for key in rows[0]}
    result['route_valid']=torch.ones_like(result['z2d'],dtype=torch.bool)
    return result
