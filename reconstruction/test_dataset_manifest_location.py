"""The ImageNet index must not require write access to the dataset tree."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import build_cached_imagefolder


class DatasetManifestLocationTest(unittest.TestCase):
    def test_default_manifest_uses_output_with_read_only_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'imagenet'
            output = root / 'run'
            for name in ('class_a', 'class_b'):
                folder = data / name
                folder.mkdir(parents=True)
                Image.new('RGB', (32, 32), 'red').save(folder / 'example.png')
            args = SimpleNamespace(data_path=str(data), dataset_manifest='',
                                   output_dir=str(output), rebuild_dataset_manifest=False,
                                   image_size=32, random_crop=False, random_flip=False)
            data.chmod(0o555)
            try:
                dataset = build_cached_imagefolder(args, distributed=False, rank=0)
                self.assertEqual(len(dataset), 2)
                self.assertTrue((output / 'dataset_manifest.pt').is_file())
                self.assertEqual(list(data.glob('*.pt')), [])
                self.assertEqual(len(build_cached_imagefolder(args, False, 0)), 2)
            finally:
                data.chmod(0o755)


if __name__ == '__main__':
    unittest.main()
