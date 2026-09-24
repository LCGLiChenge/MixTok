# MoT top-p MaskGIT (20k release)

This is the source snapshot for the completed **top-p sparse-2D MaskGIT** experiment. It contains the 20,000-update Halton training code, condition-cache/mapper/evaluation code, and the small vendor adapters needed by that code. It does **not** contain the later 20k→25k continuation, model weights, ImageNet, or generated caches.

## Model and measured result

The frozen official TiTok MaskGIT generates 32 one-dimensional IDs. The top-p tokenizer decodes their 16×16 feature map; an `f_1D`-only scorer chooses refinement positions and a codebook mapper predicts 256 proxy 2D IDs. One Halton MaskGIT then generates **only the selected** 2D IDs, and the top-p decoder directly replaces the selected features before rendering. The unselected features stay as the continuous 1D features; they are not replaced with proxy IDs at image decode time.

The completed 20k checkpoint was evaluated with 8 TiTok + 8 Halton sampling steps. A fixed 5k CFG scan selected 2D CFG `w=2.5`; the subsequent 50k class-conditional ImageNet-256 evaluation gave gFID **1.9209705649433317**, with **98.95888** generated 2D IDs on average (32 + 98.95888 total). These numbers describe that exact checkpoint and protocol, not a fresh run from this repository.

## Source layout

- `experiments/feature_to_token_20260916/train_halton_top_p.py`: **completed 20k** 8-GPU training entry point; official Halton initialization, 50% arccos / 50% Halton-prefix masks, selected-and-masked CE, budget-cosine LR, W&B, rolling `latest.pt`, and asynchronous 5k FID.
- `experiments/feature_to_token_20260916/data_top_p.py`, `halton_proxy.py`, `halton_official_completion.py`, `halton_mixed_completion.py`: model/data/objective implementation.
- `experiments/feature_to_token_20260916/cache_final_conditions.py`, `train_proxy_converter.py`, `top_p_scorer_original.py`: final condition cache and upstream mapper/scorer implementation.
- `experiments/feature_to_token_20260916/parallel_top_p_eval.py`, `eval_top_p_worker.py`, `sweep_top_p_cfg.py`: distributed evaluation and CFG scan.
- `scripts/extract_codes_top_p.py`: top-p 1D/2D code extraction.
- `scripts/launch_top_p_halton20k.sh`: path-configurable launcher for the 20k experiment.
- `third_party/`: vendored TiTok, LlamaGen, RandAR and E117 code with their provenance/license files.

## Required external assets

The repository is **code-only**. Before training, provide:

1. An H20-compatible Python 3.10/CUDA environment (`requirements-h20.txt` and `constraints-h20.txt` record the tested package versions) and eight H20 GPUs.
2. The official Halton-MaskGIT upstream source and `ImageNet_256_base.pth` from [`llvictorll/Halton-MaskGIT`](https://huggingface.co/llvictorll/Halton-MaskGIT). The trainer enforces checkpoint SHA-256 `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f`.
3. The frozen official TiTok-L32 tokenizer and the **top-p end-to-end tokenizer checkpoint at step 247482**, using its `model_ema`. The required TiTok and top-p hashes are enforced in `assets_top_p.py`. Older k-selector tokenizer/decoder weights are incompatible.
4. The accepted `f_1D`-only top-p scorer checkpoint, the separately trained codebook-8 mapper checkpoint, and their completed top-p condition cache. The trainer checks those checkpoint hashes against the cache metadata before starting.
5. The full top-p GT code cache for ImageNet train (both cached augmentations, 2,562,334 rows) and ADM evaluation assets. `configs/h20_assets.json` records asset paths and hashes; `bert2d/paths.py` describes the asset layout.

The trained checkpoints, generated caches, ImageNet data and ADM reference files are **not bundled here**. Supply non-public assets locally under `../assets/generation/` and verify their recorded hashes. This code cannot reproduce the published number from a fresh clone without those exact assets. Do not substitute earlier k-selector checkpoints.

## Train the 20k version

Run the following from `generation/` after placing assets in the adjacent directories. `MOTAR_RESULTS` points to a sibling directory outside this component. The launcher refuses an existing output directory.

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

No 25k continuation script or result is included in this release.
