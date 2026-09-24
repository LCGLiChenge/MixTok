#!/usr/bin/env python3
"""Download public pretrained weights needed by the current MoT top-p pipeline.

This does not download our private/trained checkpoint:
  - weights/top_p_resume.pt

Use --test to download into a temporary directory and delete the test files after
basic validation.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path, expected_md5: str | None = None, retries: int = 8) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    last_exc = None
    for attempt in range(1, retries + 1):
        print(f'download {url}\n  -> {dest} (attempt {attempt}/{retries})', flush=True)
        tmp.unlink(missing_ok=True)
        try:
            urllib.request.urlretrieve(url, tmp)
            if expected_md5 is not None:
                got = md5_file(tmp)
                if got != expected_md5:
                    raise RuntimeError(f'md5 mismatch for {dest}: got {got}, expected {expected_md5}')
            tmp.replace(dest)
            return
        except (urllib.error.URLError, urllib.error.ContentTooShortError, http.client.IncompleteRead, TimeoutError, RuntimeError) as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f'failed to download {url} after {retries} attempts: {last_exc}')



def download_zip_extract(url: str, dest_dir: Path, strip_top_level: bool = True, retries: int = 8) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mot_zip_') as tmp_name:
        tmp_dir = Path(tmp_name)
        archive = tmp_dir / 'archive.zip'
        download_url(url, archive, retries=retries)
        extract_root = tmp_dir / 'extract'
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_root)
        children = [p for p in extract_root.iterdir()]
        src = children[0] if strip_top_level and len(children) == 1 and children[0].is_dir() else extract_root
        tmp_dest = dest_dir.with_name(dest_dir.name + '.tmp')
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest)
        shutil.copytree(src, tmp_dest)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        tmp_dest.replace(dest_dir)

def download_hf(repo_id: str, filename: str, dest: Path, endpoint: str | None = None) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError('huggingface_hub is required for HF downloads: pip install huggingface_hub') from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    local_dir = str(dest.parent)
    print(f'download hf://{repo_id}/{filename}\n  -> {dest}', flush=True)
    old_endpoint = os.environ.get('HF_ENDPOINT')
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
    try:
        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir))
    finally:
        if endpoint:
            if old_endpoint is None:
                os.environ.pop('HF_ENDPOINT', None)
            else:
                os.environ['HF_ENDPOINT'] = old_endpoint
    if downloaded.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.copy2(downloaded, dest)


def validate_torch_load(path: Path) -> None:
    import torch
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if obj is None:
        raise RuntimeError(f'torch.load returned None for {path}')


def validate_file(path: Path, min_bytes: int, torch_load: bool = False, expected_md5: str | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f'{path} too small: {size} bytes < {min_bytes}')
    if expected_md5 is not None:
        got = md5_file(path)
        if got != expected_md5:
            raise RuntimeError(f'md5 mismatch for {path}: got {got}, expected {expected_md5}')
    if torch_load:
        validate_torch_load(path)
    print(f'ok {path} ({size / (1024**2):.1f} MiB)', flush=True)


def public_weight_specs(project_root: Path, torch_cache_root: Path, hf_endpoint: str | None, retries: int):
    return {
        'titok_l32': {
            'path': project_root / '1d-tokenizer' / 'tokenizer_titok_l32.bin',
            'download': lambda p: download_hf('fun-research/TiTok', 'tokenizer_titok_l32.bin', p, hf_endpoint),
            'min_bytes': 1024 * 1024 * 1000,
            'torch_load': False,
        },
        'llamagen_vq_ds16_c2i': {
            'path': project_root / 'LlamaGen' / 'pretrained_models' / 'vq_ds16_c2i.pt',
            'download': lambda p: download_hf('FoundationVision/LlamaGen', 'vq_ds16_c2i.pt', p, hf_endpoint),
            'min_bytes': 1024 * 1024 * 200,
            'torch_load': True,
        },
        'dinov2_repo': {
            'path': torch_cache_root / 'hub' / 'facebookresearch_dinov2_main' / 'hubconf.py',
            'download': lambda p: download_zip_extract('https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip', p.parent, retries=retries),
            'min_bytes': 1024,
            'torch_load': False,
        },
        'dinov2_vits14': {
            'path': torch_cache_root / 'hub' / 'checkpoints' / 'dinov2_vits14_pretrain.pth',
            'download': lambda p: download_url('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', p, retries=retries),
            'min_bytes': 1024 * 1024 * 50,
            'torch_load': True,
        },
        'lpips_vgg': {
            'path': project_root / 'LlamaGen' / 'tokenizer' / 'tokenizer_image' / 'cache' / 'vgg.pth',
            'download': lambda p: download_url('https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1', p, expected_md5='d507d7349b931f0638a25a48a722f98a', retries=retries),
            'min_bytes': 1024,
            'torch_load': True,
            'md5': 'd507d7349b931f0638a25a48a722f98a',
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Download public pretrained weights for MoT top-p training.')
    parser.add_argument('--project-root', type=Path, default=Path(".."), help='Root containing 1d-tokenizer and LlamaGen.')
    parser.add_argument('--torch-cache-root', type=Path, default=Path("../.cache/torch"), help='Torch cache root containing hub/checkpoints.')
    parser.add_argument('--test', action='store_true', help='Download into a temp dir, validate, then delete it.')
    parser.add_argument('--test-dir', type=Path, default=None, help='Optional temp dir for --test.')
    parser.add_argument('--only', nargs='*', default=None, help='Subset to download: titok_l32 llamagen_vq_ds16_c2i dinov2_repo dinov2_vits14 lpips_vgg')
    parser.add_argument('--hf-endpoint', default=os.environ.get('HF_ENDPOINT'), help='Optional HF endpoint, e.g. https://hf-mirror.com')
    parser.add_argument('--overwrite', action='store_true', help='Re-download existing files.')
    parser.add_argument('--retries', type=int, default=8, help='Retry count for direct URL downloads.')
    parser.add_argument('--keep-test-files', action='store_true', help='With --test, do not delete downloaded test files.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test:
        root = args.test_dir or Path(tempfile.mkdtemp(prefix='mot_public_weights_test_'))
        project_root = root
        torch_cache_root = root / '.cache' / 'torch'
        root_created = True
    else:
        root = None
        project_root = args.project_root
        torch_cache_root = args.torch_cache_root
        root_created = False
    specs = public_weight_specs(project_root, torch_cache_root, args.hf_endpoint, args.retries)
    names = args.only or list(specs)
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f'unknown weight names: {unknown}; choices={sorted(specs)}')
    print(f'project_root={project_root}', flush=True)
    print(f'torch_cache_root={torch_cache_root}', flush=True)
    try:
        for name in names:
            spec = specs[name]
            path = spec['path']
            if path.exists() and not args.overwrite:
                print(f'skip existing {name}: {path}', flush=True)
            else:
                spec['download'](path)
            validate_file(path, spec['min_bytes'], torch_load=spec.get('torch_load', False), expected_md5=spec.get('md5'))
        print('all requested public weights are ready', flush=True)
        return 0
    finally:
        if args.test and root_created and not args.keep_test_files:
            print(f'cleanup test dir {root}', flush=True)
            shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
