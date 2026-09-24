"""Frozen paired feature-to-token audit. No tokenizer or training weights changed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
import numpy as np
import torch
from bert2d.paths import ASSETS, atomic_json, sha256
from h20_joint.assets import FrozenAssets

PACKED = Path(os.environ.get('MOT_PACKED_VAL', ROOT.parent / 'assets/generation/codes/val'))
WEIGHT = Path(os.environ.get('MOT_INCEPTION_WEIGHT', ROOT.parent / 'assets/generation/fid/weights-inception-2015-12-05-6726825d.pth'))
ARMS = ('base', 'nearest', 'oracle_mix', 'nearest_oracle_mix', 'affine_span', 'gt2d')


def nearest(features, book, chunk=256):
    """Exact squared Euclidean nearest neighbour in decoder-input space."""
    flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    ids = []
    for row in flat.split(chunk):
        distance = row.square().sum(1, keepdim=True) + book.square().sum(1)[None] - 2 * row @ book.T
        ids.append(distance.argmin(1))
    ids = torch.cat(ids).reshape(len(features), -1)
    values = book[ids].transpose(1, 2).reshape_as(features)
    return ids, values


def affine_project(features, weight, bias):
    # All projected 8D codewords lie in bias + column-space(weight).
    u, s, _ = torch.linalg.svd(weight.double(), full_matrices=False)
    rank = int((s > s.max() * 1e-10).sum())
    basis = u[:, :rank].float()
    flat = features.flatten(2).transpose(1, 2) - bias
    projected = (flat @ basis) @ basis.T + bias
    return projected.transpose(1, 2).reshape_as(features), rank


def self_test():
    torch.manual_seed(12)
    book = torch.randn(31, 8)
    features = torch.randn(2, 8, 3, 3)
    ids, values = nearest(features, book, 4)
    expected = (features.permute(0, 2, 3, 1).reshape(-1, 8)[:, None] - book).square().sum(-1).argmin(-1)
    assert torch.equal(ids.flatten(), expected)
    torch.testing.assert_close(values.permute(0, 2, 3, 1).reshape(-1, 8), book[expected])
    weight = torch.randn(8, 3)
    bias = torch.randn(8)
    projection, rank = affine_project(features, weight, bias)
    assert rank == 3
    residual = (features - projection).permute(0, 2, 3, 1).reshape(-1, 8)
    torch.testing.assert_close(residual @ weight, torch.zeros(18, 3), atol=2e-5, rtol=0)
    print('nearest/affine CPU tests passed', flush=True)


def fid(features, real, device):
    # Symmetric PSD covariance sandwich, float64, no scipy complex sqrtm.
    x = torch.as_tensor(features, device=device, dtype=torch.float64)
    y = torch.as_tensor(real, device=device, dtype=torch.float64)
    mx, my = x.mean(0), y.mean(0)
    x, y = x - mx, y - my
    cx, cy = x.T @ x / (len(x)-1), y.T @ y / (len(y)-1)
    eigen, vec = torch.linalg.eigh(cy)
    sqrt = (vec * eigen.clamp_min(0).sqrt()[None]) @ vec.T
    middle = sqrt @ cx @ sqrt
    cross = torch.linalg.eigvalsh((middle + middle.T) * .5).clamp_min(0).sqrt().sum()
    return float(((mx-my).square().sum() + torch.trace(cx) + torch.trace(cy) - 2*cross).clamp_min(0))


def run(args):
    from PIL import Image
    from torchvision.datasets import ImageFolder
    from torchvision.utils import save_image
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.functional.image import structural_similarity_index_measure
    from scripts.extract_mot_titok_llamagen_packed import PairedTransform
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    local = int(os.getenv('LOCAL_RANK', '0'))
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.85, device)
    if torch.cuda.mem_get_info(device)[0] < 20 * 1024**3:
        raise RuntimeError('requires 20 GiB free at admission')
    if not 2 <= args.n <= 1000:
        raise ValueError('2..1000 stratified validation images')
    out = args.output.resolve()
    allowed = Path(os.environ.get('MOT_PROBE_RESULTS', ROOT.parent / 'results/generation/probes')).resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('output must be a named child of MOT_PROBE_RESULTS')
    out.mkdir(parents=True, exist_ok=True)
    status = out / f'status_rank{rank}.json'
    if status.exists():
        raise FileExistsError(status)
    start = time.monotonic()
    atomic_json(status, dict(status='loading', pid=os.getpid(), rank=rank))
    meta = json.loads((PACKED/'meta.json').read_text())
    assert meta['completed'] and meta['num_samples'] == 50000 and meta['mot_step'] == 199440
    assert meta['mot_state_key'] == 'model_ema' and meta['augmentation'] == 'none'
    dataset = ImageFolder(meta['data_path'])
    labels = np.load(PACKED/'labels.npy', mmap_mode='r')
    assert np.array_equal(dataset.targets, labels)
    assert np.load(PACKED/'written.npy', mmap_mode='r').all()
    rng = np.random.default_rng(20260916)
    classes = rng.permutation(1000)[:args.n]
    chosen = np.array([rng.choice(np.flatnonzero(labels == c)) for c in classes])
    indices = chosen[rank::world]
    z1cache = np.load(PACKED/'titok_codes.npy', mmap_mode='r')
    z2cache = np.load(PACKED/'llamagen_codes.npy', mmap_mode='r')
    transform = PairedTransform(256)
    frozen = FrozenAssets(ASSETS, 'cpu', chunk=args.batch)
    # Native RGB TiTok decoder is not needed; do not put it on GPU.
    frozen.shell.to(device)
    frozen.adapter.to(device)
    frozen.device = device
    frozen.assert_frozen()
    vq = frozen.shell.llamagen_vq
    projection = vq.post_quant_conv
    assert projection.kernel_size == (1, 1) and projection.stride == (1, 1)
    assert projection.padding == (0, 0) and projection.weight.shape == (256, 8, 1, 1)
    metric = FrechetInceptionDistance(feature=2048, normalize=False,
            feature_extractor_weights_path=str(WEIGHT)).to(device).eval().requires_grad_(False)
    features = {k: [] for k in ('real',) + ARMS}
    records = []
    manifest = dict(protocol='feature_to_token_paired_val_v1', seed=20260916, n=args.n,
        subset=chosen.tolist(), rank=rank, world=world, batch=args.batch, arms=ARMS,
        frozen_audit=frozen.audit, packed_meta=meta, packed_meta_sha256=sha256(PACKED/'meta.json'),
        script_sha256=sha256(Path(__file__)), inception_sha256=sha256(WEIGHT),
        precision='fp32_no_tf32', device=torch.cuda.get_device_name(device),
        source='GT image-encoded 1D; no MaskGIT generation; E117 routes from 1D only',
        fid_reference='identical selected real val crops, torch-fidelity Inception; NOT ADM generation FID',
        nearest='all 16384 normalized native codewords after post_quant_conv; exact Euclidean',
        affine_span='diagnostic continuous projection only, NOT a discrete token method')
    atomic_json(out/f'manifest_rank{rank}.json', manifest)
    with torch.inference_mode():
        embedding = vq.quantize.get_codebook_entry(torch.arange(16384, device=device))
        book = projection(embedding.T[None, :, None]).squeeze(0).squeeze(1).T.contiguous()
        assert book.shape == (16384, 256)
        # On-manifold identity check, avoiding zero vectors/ties by using spread-out IDs.
        test_ids = torch.arange(0, 16384, 64, device=device)
        recovered, recovered_features = nearest(book[test_ids].T.reshape(1, 256, 16, 16), book)
        torch.testing.assert_close(recovered_features, book[test_ids].T.reshape(1, 256, 16, 16), atol=1e-5, rtol=1e-5)
        for offset in range(0, len(indices), args.batch):
            if time.monotonic() - start > 1800:
                raise TimeoutError('30-minute bounded probe expired')
            rows = indices[offset:offset+args.batch]
            real = torch.stack([transform(Image.open(dataset.samples[int(i)][0]).convert('RGB'))[0] for i in rows]).to(device)
            z1 = torch.from_numpy(np.array(z1cache[rows, 0], dtype=np.int64)).to(device)
            z2 = torch.from_numpy(np.array(z2cache[rows, 0], dtype=np.int64)).to(device)
            bundle = frozen.bundle(z1)
            base, idx, valid = bundle['base'], bundle['index'], bundle['valid']
            selected = torch.zeros(len(rows), 256, device=device, dtype=torch.bool)
            bi = torch.arange(len(rows), device=device)[:, None].expand_as(idx)[valid]
            selected[bi, idx[valid]] = True
            ids, quantized = nearest(base, book)
            gt = book[z2].transpose(1, 2).reshape_as(base)
            affine, subspace_rank = affine_project(base, projection.weight[:, :, 0, 0], projection.bias)
            sparse_gt = z2.gather(1, idx.clamp_min(0))
            oracle = frozen.mixed(base, sparse_gt, idx, valid)
            quantized_oracle = frozen.mixed(quantized, sparse_gt, idx, valid)
            mask = selected.reshape(-1, 1, 16, 16)
            torch.testing.assert_close(oracle, torch.where(mask, gt, base), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(quantized_oracle, torch.where(mask, gt, quantized), atol=1e-6, rtol=1e-6)
            error = (base-quantized).square().mean(1).flatten(1)
            residual = (base-affine).square().mean(1).flatten(1)
            # Continuous affine projection lower-bounds every codeword's squared distance.
            assert bool((residual <= error + 1e-5).all())
            batch_records = []
            for j, row in enumerate(rows):
                rec = dict(index=int(row), label=int(labels[row]), k=int(selected[j].sum()), subspace_rank=subspace_rank,
                    feature_power=float(base[j].square().mean()), feature_mse=float(error[j].mean()),
                    affine_residual_mse=float(residual[j].mean()),
                    feature_mse_unselected=float(error[j, ~selected[j]].mean()),
                    feature_mse_selected=float(error[j, selected[j]].mean()),
                    token_gt_match_unselected=float((ids[j, ~selected[j]] == z2[j, ~selected[j]]).float().mean()),
                    token_gt_match_selected=float((ids[j, selected[j]] == z2[j, selected[j]]).float().mean()))
                batch_records.append(rec)
            base_image = frozen.render(base)
            previews = [real]
            for name, latent in dict(real=None, base=base, nearest=quantized, oracle_mix=oracle,
                    nearest_oracle_mix=quantized_oracle, affine_span=affine, gt2d=gt).items():
                pixels = real if name == 'real' else base_image if name == 'base' else frozen.render(latent)
                assert bool(torch.isfinite(pixels).all())
                encoded = (pixels * 255).round().to(torch.uint8)
                features[name].append(metric.inception(encoded).cpu().numpy())
                if name != 'real':
                    psnr = -10 * (pixels-real).square().flatten(1).mean(1).clamp_min(1e-12).log10()
                    base_psnr = -10 * (pixels-base_image).square().flatten(1).mean(1).clamp_min(1e-12).log10()
                    ssim = structural_similarity_index_measure(pixels, real, data_range=1., reduction='none')
                    for j, rec in enumerate(batch_records):
                        rec[name] = dict(psnr=float(psnr[j]), ssim=float(ssim[j]), psnr_vs_base=float(base_psnr[j]))
                    if offset == 0: previews.append(pixels)
            if offset == 0:
                # Each ROW is one image; columns: real followed by ARMS.
                grid = torch.stack(previews, 1).flatten(0, 1)
                save_image(grid, out/f'preview_rank{rank}.png', nrow=len(previews))
            records.extend(batch_records)
            atomic_json(status, dict(status='running', pid=os.getpid(), completed=len(records), total=len(indices),
                seconds=time.monotonic()-start, peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
            atomic_json(out/f'records_rank{rank}.json', records)
            print(json.dumps(dict(rank=rank, completed=len(records), total=len(indices), seconds=time.monotonic()-start)), flush=True)
    np.savez(out/f'features_rank{rank}.npz', indices=indices, **{k: np.concatenate(v) for k,v in features.items()})
    atomic_json(status, dict(status='complete', pid=os.getpid(), completed=len(records), seconds=time.monotonic()-start,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    if rank != 0: return
    while any(not (out/f'status_rank{r}.json').exists() or json.loads((out/f'status_rank{r}.json').read_text())['status'] != 'complete' for r in range(world)):
        if time.monotonic()-start > 1800: raise TimeoutError('rank merge timed out')
        time.sleep(2)
    all_records = sum([json.loads((out/f'records_rank{r}.json').read_text()) for r in range(world)], [])
    arrays = [np.load(out/f'features_rank{r}.npz') for r in range(world)]
    assert sorted(np.concatenate([a['indices'] for a in arrays]).tolist()) == sorted(chosen.tolist())
    real_features = np.concatenate([a['real'] for a in arrays])
    results = {}
    for arm in ARMS:
        results[arm] = {key: float(np.mean([r[arm][key] for r in all_records])) for key in ('psnr', 'ssim', 'psnr_vs_base')}
        results[arm]['paired_subset_rfid'] = fid(np.concatenate([a[arm] for a in arrays]), real_features, device)
    scalar = {k: float(np.mean([r[k] for r in all_records])) for k in all_records[0] if k not in ('index','label')+ARMS}
    atomic_json(out/'summary.json', dict(status='complete', n=args.n, results=results, feature_metrics=scalar,
        seconds=time.monotonic()-start, interpretation='paired reconstruction audit; small-sample rFID, not generation FID',
        manifest='manifest_rank0.json'))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test: self_test()
    else:
        if args.output is None: p.error('--output required')
        try: run(args)
        except BaseException as exc:
            if args.output.exists():
                atomic_json(args.output/f'failure_rank{os.getenv("RANK", "0")}.json', dict(error=repr(exc), pid=os.getpid()))
            raise
