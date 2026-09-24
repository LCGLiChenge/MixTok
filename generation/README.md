# MixTok Generation: Sparse 2D MaskGIT

This directory contains the **top-p sparse-2D MaskGIT** code for both the proxy-ID branch and the newer continuous-context branch. It includes their training, cache, and evaluation scripts plus the vendor adapters they use. Model weights, ImageNet, and generated caches are external assets.

## Generation model

The frozen official TiTok MaskGIT generates 32 one-dimensional IDs. The top-p tokenizer decodes their 16×16 feature map; an `f_1D`-only scorer chooses refinement positions and a codebook mapper predicts 256 proxy 2D IDs. One Halton MaskGIT then generates **only the selected** 2D IDs, and the top-p decoder directly replaces the selected features before rendering. The unselected features stay as the continuous 1D features; they are not replaced with proxy IDs at image decode time.

The generation code provides a proxy-ID context branch and a continuous-context branch. Both use the same selected 2D positions and generate discrete 2D IDs only at those positions. The continuous branch maps the 1D feature grid directly into Halton input embeddings; its training scripts and asset requirements are described below.

## Source layout

- `experiments/feature_to_token_20260916/train_halton_top_p.py`: **completed 20k** 8-GPU training entry point; official Halton initialization, 50% arccos / 50% Halton-prefix masks, selected-and-masked CE, budget-cosine LR, W&B, rolling `latest.pt`, and asynchronous 5k FID.
- `experiments/feature_to_token_20260916/data_top_p.py`, `halton_proxy.py`, `halton_official_completion.py`, `halton_mixed_completion.py`: model/data/objective implementation.
- `experiments/feature_to_token_20260916/cache_final_conditions.py`, `train_proxy_converter.py`, `top_p_scorer_original.py`: final condition cache and upstream mapper/scorer implementation.
- `experiments/feature_to_token_20260916/parallel_top_p_eval.py`, `eval_top_p_worker.py`, `sweep_top_p_cfg.py`: distributed evaluation and CFG scan.
- `scripts/extract_codes_top_p.py`: top-p 1D/2D code extraction.
- `scripts/launch_top_p_halton20k.sh`: path-configurable launcher for the proxy-ID experiment.
- `experiments/continuous_halton_embedding_20260923/`: continuous 768D context adapter, sparse Halton training, CFG sweep, and compact-checkpoint evaluation.
- `scripts/launch_continuous_adapter20k.sh`, `scripts/launch_continuous_halton20k.sh`: launchers for the new two-stage continuous-context variant.
- `third_party/`: vendored TiTok, LlamaGen, RandAR and E117 code with their provenance/license files.

## Required external assets

The repository is **code-only**. Before training, provide:

1. An H20-compatible Python 3.10/CUDA environment (`requirements-h20.txt` and `constraints-h20.txt` record the tested package versions) and eight H20 GPUs.
2. The official Halton-MaskGIT upstream source and `ImageNet_256_base.pth` from [`llvictorll/Halton-MaskGIT`](https://huggingface.co/llvictorll/Halton-MaskGIT). The trainer enforces checkpoint SHA-256 `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f`.
3. The frozen official TiTok-L32 tokenizer and the **top-p end-to-end tokenizer checkpoint at step 247482**, using its `model_ema`. The required TiTok and top-p hashes are enforced in `assets_top_p.py`. Older k-selector tokenizer/decoder weights are incompatible.
4. The accepted `f_1D`-only top-p scorer checkpoint, the separately trained codebook-8 mapper checkpoint, and their completed top-p condition cache. The trainer checks those checkpoint hashes against the cache metadata before starting.
5. The full top-p GT code cache for ImageNet train (both cached augmentations, 2,562,334 rows) and ADM evaluation assets. `configs/h20_assets.json` records asset paths and hashes; `bert2d/paths.py` describes the asset layout.

The trained checkpoints, generated caches, ImageNet data and ADM reference files are **not bundled here**. Supply non-public assets locally under `../assets/generation/` and verify their recorded hashes. Training and evaluation require those exact external assets. Do not substitute earlier k-selector checkpoints.

## Train the 20k version

Run the following from `generation/` after placing assets in the adjacent directories. `MOTAR_RESULTS` points to a sibling directory outside this component. The launcher refuses an existing output directory. The historical `MOTAR_*` environment variable names remain part of its command-line interface.

```bash
export MOTAR_ASSETS=../assets/generation
export MOTAR_RESULTS=../results/generation
export HALTON_UPSTREAM_ROOT=../Halton-MaskGIT/source
export HALTON_OFFICIAL_CKPT=../assets/generation/ImageNet_256_base.pth
export TOP_P_CKPT=../assets/reconstruction/epoch_0006_step_00247482.pt
export TOP_P_GT2D_CACHE=../assets/generation/gt2d_top_p_train
export TOP_P_CONDITION_CACHE=../assets/generation/final_conditions_top_p_train
export TOP_P_MAPPER_CKPT=../assets/generation/mapper20k/latest.pt
export TOP_P_SCORER_CKPT=../assets/generation/top_p_scorer/latest.pt
export TOP_P_OUTPUT="$MOTAR_RESULTS/halton20k_top_p_seed0"
bash scripts/launch_top_p_halton20k.sh
```

The run uses global batch 256 (8×32), seed 0, peak LR `2e-5`, 2,500-step warmup, hold through step 4,000, cosine decay to `2e-6` at step 20,000, and W&B project `mot-maskgit-review`. Training stops at **20,000 optimizer updates**. `latest.pt` is a rolling checkpoint; 5k FID is requested at 2k, 4k, 8k, 12k, 16k, 20k and completed epoch boundaries. The final 50k FID requires a separate CFG scan and evaluation using `sweep_top_p_cfg.py`.

## Continuous 768D context variant

The newer variant maps the 16 × 16 1D feature grid directly to 256 continuous, 768-dimensional Halton input embeddings. It uses the same top-p route policy, generates discrete 2D IDs only at selected locations, and keeps the 1D-derived continuous features at unselected decoder locations. Its embedding adapter starts from fresh parameters; it does not initialize from the proxy-ID mapper.

The implementation is in [`experiments/continuous_halton_embedding_20260923/`](experiments/continuous_halton_embedding_20260923/). `train_adapter.py` fits the continuous adapter against the official Halton embedding table; `train_halton.py` trains the sparse generator; `sweep_cfg.py` runs the CFG search and final evaluation. `export_top_p_inference.py` can produce a smaller inference-only tokenizer checkpoint, and `verify_top_p_inference.py` checks its tensor equality against the full checkpoint. Training requires the full checkpoint.

Besides the assets above, this variant needs a teacher-proxy cache for adapter targets and a completed **route-only cache** for Halton training. The latter contains `config.json`, `summary.json`, `route_indices.npy`, and `route_k.npy`. It must match the frozen tokenizer and scorer identities checked by [`data.py`](experiments/continuous_halton_embedding_20260923/data.py). The optional `prepare_h20.py` migrates route files from an existing completed proxy-ID baseline; pass both `--baseline-root` and a new `--root` directory. It does not download or rebuild the source caches.

From `generation/`, with the same external asset variables as above:

```bash
export TOP_P_TEACHER_PROXY_CACHE=../assets/generation/teacher_proxy_top_p_train
export TOP_P_ROUTE_CACHE=../assets/generation/top_p_continuous_route_cache
bash scripts/launch_continuous_adapter20k.sh

export CONTINUOUS_ADAPTER_CKPT="$MOTAR_RESULTS/continuous_adapter20k_seed20260918/latest.pt"
bash scripts/launch_continuous_halton20k.sh

python experiments/continuous_halton_embedding_20260923/sweep_cfg.py \
  --checkpoint "$MOTAR_RESULTS/continuous_halton20k_seed0/latest.pt" \
  --mapper-checkpoint "$CONTINUOUS_ADAPTER_CKPT" \
  --scorer-checkpoint "$TOP_P_SCORER_CKPT" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/continuous_cfg_sweep"
```

Both launchers use eight H20 GPUs, stop after 20,000 updates, and write a rolling `latest.pt`. The final sweep requires the ADM evaluation assets under `MOTAR_ASSETS`. Use new output directories for each run. If an existing completed proxy-ID baseline is available, the optional `run_pipeline.py` can orchestrate the stages using the same environment variables and a prepared route cache.
