# MixTok Generation: Continuous Embedding MaskGIT

This directory contains the **continuous embedding** generation path. The official TiTok MaskGIT samples 32 global 1D IDs. The frozen top-p tokenizer expands them into a 16 × 16 feature grid, and a scorer selects a variable number of refinement locations. A trained adapter maps the 1D feature grid to 256 continuous, 768-dimensional Halton input embeddings. Halton MaskGIT generates discrete 2D IDs only at the selected locations; the shared decoder keeps the 1D-derived features everywhere else.

![MixTok generation with continuous embedding context, top-p spatial selection, sparse 2D MaskGIT, and a shared decoder.](../docs/maskgit.png)

## Requirements

- Python 3.10, eight H20 GPUs, PyTorch 2.10.0, torchvision 0.25.0, and the packages in [`requirements-h20.txt`](requirements-h20.txt) and [`constraints-h20.txt`](constraints-h20.txt). Training uses W&B online login.
- Official Halton-MaskGIT source and `ImageNet_256_base.pth` from [`llvictorll/Halton-MaskGIT`](https://huggingface.co/llvictorll/Halton-MaskGIT). The required checkpoint SHA-256 is `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f`.
- Official TiTok-L32 weights and the evaluated top-p reconstruction checkpoint at step 247482, loaded from `model_ema`. The top-p checkpoint SHA-256 is `61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819`. Training uses the full checkpoint; [`export_top_p_inference.py`](experiments/continuous_halton_embedding_20260923/export_top_p_inference.py) can make a smaller inference-only copy.
- The accepted 1D-feature top-p scorer checkpoint and the paired ImageNet top-p GT code cache, including both cached augmentations. The cache contains `titok_codes.npy`, `llamagen_codes.npy`, `labels.npy`, and `meta.json`.
- A completed teacher-token cache for adapter targets and a completed route-only cache for sparse training. The teacher cache's `proxy_codes.npy` filename is part of its existing data format; the generator uses continuous embeddings. The route cache contains `config.json`, `summary.json`, `route_indices.npy`, and `route_k.npy`.
- ADM FID reference assets for evaluation. [`configs/h20_assets.json`](configs/h20_assets.json) records local asset paths, sizes, and hashes.

Checkpoints, ImageNet data, generated caches, and ADM reference files are external assets. Supply them under `../assets/` or set the paths below to their actual locations. The checked-in source uses relative paths and checks the expected asset identities.

## Train and evaluate

Run from `generation/` after supplying the assets. The launchers require new output directories and run for 20,000 optimizer updates each.

```bash
export MOTAR_ASSETS=../assets/generation
export MOTAR_RESULTS=../results/generation
export HALTON_UPSTREAM_ROOT=../Halton-MaskGIT/source
export HALTON_OFFICIAL_CKPT=../assets/generation/ImageNet_256_base.pth
export TOP_P_CKPT=../assets/reconstruction/epoch_0006_step_00247482.pt
export TOP_P_SCORER_CKPT=../assets/generation/top_p_scorer/latest.pt
export TOP_P_GT2D_CACHE=../assets/generation/gt2d_top_p_train
export TOP_P_TEACHER_PROXY_CACHE=../assets/generation/teacher_proxy_top_p_train
export TOP_P_ROUTE_CACHE=../assets/generation/top_p_continuous_route_cache

bash scripts/launch_continuous_adapter20k.sh
export CONTINUOUS_ADAPTER_CKPT="$MOTAR_RESULTS/continuous_adapter20k_seed20260918/latest.pt"
bash scripts/launch_continuous_halton20k.sh

python experiments/continuous_halton_embedding_20260923/sweep_cfg.py \
  --checkpoint "$MOTAR_RESULTS/continuous_halton20k_seed0/latest.pt" \
  --adapter-checkpoint "$CONTINUOUS_ADAPTER_CKPT" \
  --scorer-checkpoint "$TOP_P_SCORER_CKPT" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/continuous_cfg_sweep"
```

The first stage trains the embedding adapter against the official Halton embedding table using the teacher-token cache. The second stage freezes that adapter and trains sparse Halton MaskGIT using the route-only cache. `sweep_cfg.py` selects CFG on a fixed 5k cohort, then evaluates the selected setting on 50k samples. Both training stages write a rolling `latest.pt`.

### Build the caches

[`cache_teacher_targets.py`](experiments/continuous_halton_embedding_20260923/cache_teacher_targets.py) builds the teacher-token cache by decoding each 1D base image and re-encoding it with the frozen top-p VQ model. [`cache_routes.py`](experiments/continuous_halton_embedding_20260923/cache_routes.py) applies the accepted scorer to the frozen 1D features and records only the selected top-p positions. Neither cache builder trains a generator.

Both cache builders use `prepare`, `extract`, and `finalize` modes. For example, after setting the variables above and supplying the paired GT cache, prepare the route cache and distribute extraction across eight GPUs:

```bash
python experiments/continuous_halton_embedding_20260923/cache_routes.py prepare \
  --assets-root "$MOTAR_ASSETS" --source-root "$TOP_P_GT2D_CACHE" \
  --top-p-ckpt "$TOP_P_CKPT" --scorer-ckpt "$TOP_P_SCORER_CKPT" \
  --output "$TOP_P_ROUTE_CACHE" --world-size 8

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$rank" python experiments/continuous_halton_embedding_20260923/cache_routes.py extract \
    --assets-root "$MOTAR_ASSETS" --source-root "$TOP_P_GT2D_CACHE" \
    --top-p-ckpt "$TOP_P_CKPT" --scorer-ckpt "$TOP_P_SCORER_CKPT" \
    --output "$TOP_P_ROUTE_CACHE" --world-size 8 --rank "$rank" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done

python experiments/continuous_halton_embedding_20260923/cache_routes.py finalize \
  --assets-root "$MOTAR_ASSETS" --source-root "$TOP_P_GT2D_CACHE" \
  --top-p-ckpt "$TOP_P_CKPT" --scorer-ckpt "$TOP_P_SCORER_CKPT" \
  --output "$TOP_P_ROUTE_CACHE" --world-size 8
```

The teacher cache builder takes the same `--source-root`, `--top-p-ckpt`, and `--output` paths, plus `--assets-root`; use its `--help` for rank and batch options. Cache identities and row alignment are validated again when training starts.

## Source layout

- [`experiments/continuous_halton_embedding_20260923/`](experiments/continuous_halton_embedding_20260923/): embedding adapter, continuous Halton model, cache builders, training, asynchronous evaluation, CFG sweep, and compact-checkpoint tools.
- [`scripts/launch_continuous_adapter20k.sh`](scripts/launch_continuous_adapter20k.sh) and [`scripts/launch_continuous_halton20k.sh`](scripts/launch_continuous_halton20k.sh): eight-GPU training launchers.
- [`scripts/extract_codes_top_p.py`](scripts/extract_codes_top_p.py): paired top-p code extraction.
- [`third_party/`](third_party/): vendored upstream components with their provenance and license files.
