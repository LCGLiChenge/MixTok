"""Small corruption checks for the GT files used by continuous Halton training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bert2d.paths import sha256
from experiments.continuous_halton_embedding_20260923 import data


class RouteSourceIntegrityTest(unittest.TestCase):
    def make_fixture(self, root: Path):
        source = root / 'source'
        cache = root / 'route'
        source.mkdir()
        cache.mkdir()
        (source / 'meta.json').write_text(json.dumps({
            'completed': True, 'mot_step': 247482, 'mot_state_key': 'model_ema',
        }))
        np.save(source / 'titok_codes.npy', np.zeros((2, 2, 32), dtype=np.uint16))
        np.save(source / 'llamagen_codes.npy', np.zeros((2, 2, 256), dtype=np.uint16))
        np.save(source / 'labels.npy', np.array([3, 7], dtype=np.uint16))
        np.save(cache / 'route_indices.npy', np.zeros((4, 256), dtype=np.uint16))
        np.save(cache / 'route_k.npy', np.ones(4, dtype=np.uint16))
        config = {
            'format': data.FORMAT,
            'top_p_checkpoint_sha256': data.TOP_P_SHA256,
            'source_meta_sha256': sha256(source / 'meta.json'),
            'source_titok_sha256': sha256(source / 'titok_codes.npy'),
            'source_gt2d_sha256': sha256(source / 'llamagen_codes.npy'),
            'source_labels_sha256': sha256(source / 'labels.npy'),
        }
        self.write_route_config(cache, config)
        return source, cache, config

    def write_route_config(self, cache: Path, config: dict):
        (cache / 'config.json').write_text(json.dumps(config))
        (cache / 'summary.json').write_text(json.dumps({
            'status': 'complete',
            'config_sha256': sha256(cache / 'config.json'),
            'route_indices_sha256': sha256(cache / 'route_indices.npy'),
            'route_k_sha256': sha256(cache / 'route_k.npy'),
        }))

    def load(self, source: Path, cache: Path):
        with patch.object(data, 'NUM_SOURCES', 2):
            return data.ContinuousRouteDataset(source, source, cache)

    def test_current_config_rejects_changed_gt_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cache, _ = self.make_fixture(Path(tmp))
            dataset = self.load(source, cache)
            self.assertEqual(len(dataset.k_values), 4)
            self.assertEqual(int(dataset[0]['label']), 3)
            gt = np.load(source / 'llamagen_codes.npy', mmap_mode='r+')
            gt[0, 0, 0] = 1
            gt.flush()
            with self.assertRaisesRegex(RuntimeError, 'llamagen_codes.npy hash differs'):
                self.load(source, cache)
            gt[0, 0, 0] = 0
            gt.flush()
            labels = np.load(source / 'labels.npy', mmap_mode='r+')
            labels[0] = 9
            labels.flush()
            with self.assertRaisesRegex(RuntimeError, 'labels.npy hash differs'):
                self.load(source, cache)
            labels[0] = 3
            labels.flush()
            (source / 'meta.json').write_text(json.dumps({
                'completed': True, 'mot_step': 247482,
                'mot_state_key': 'model_ema', 'changed': True,
            }))
            with self.assertRaisesRegex(RuntimeError, 'metadata differs'):
                self.load(source, cache)

    def test_legacy_config_uses_source_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cache, config = self.make_fixture(Path(tmp))
            (source / 'manifest.sha256').write_text(
                f"{config['source_gt2d_sha256']}  llamagen_codes.npy\n"
                f"{config['source_labels_sha256']}  labels.npy\n"
            )
            del config['source_gt2d_sha256']
            del config['source_labels_sha256']
            self.write_route_config(cache, config)
            self.assertEqual(len(self.load(source, cache).k_values), 4)
            gt = np.load(source / 'llamagen_codes.npy', mmap_mode='r+')
            gt[0, 0, 0] = 1
            gt.flush()
            with self.assertRaisesRegex(RuntimeError, 'llamagen_codes.npy hash differs'):
                self.load(source, cache)


if __name__ == '__main__':
    unittest.main()
