# MoT: reconstruction and sparse 2D generation

Code accompanying the anonymous submission. This repository contains two connected pipelines:

1. **Reconstruction:** a TiTok-based 1D representation is decoded into a 16 × 16 feature grid. A scorer assigns probability mass over the 256 grid positions; a top-p rule selects a different number of 2D refinement tokens for each image. The selected features are refined before image decoding.
2. **Generation:** a frozen TiTok MaskGIT produces 32 1D IDs. A scorer acting on the 1D features selects 2D positions, a mapper provides proxy 2D IDs, and a Halton MaskGIT generates IDs only at the selected positions. The decoder uses continuous 1D features at unselected positions.

The fixed **96/256** grid budget appears only in the reconstruction warm-start stage. The final reconstruction and generation pipelines use image-dependent top-p selection.

For a 16 × 16 grid, the router applies softmax to its 256 scores (after temperature scaling), sorts the resulting probabilities, and selects the shortest prefix whose cumulative mass reaches `p`. A straight-through surrogate supplies gradients during training. Thus `K` varies by image; the inference calibration targets roughly 96 selected positions **on average**, not 96 in every image. See `make_top_p_ste_mask` in the [reconstruction trainer](reconstruction/train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py).

## Contents

| Directory | Main purpose | Start here |
| --- | --- | --- |
| [`reconstruction/`](reconstruction/) | Adapter, fixed-budget warm-start, end-to-end top-p reconstruction, and ImageNet validation | [Reconstruction guide](reconstruction/README.md) |
| [`generation/`](generation/) | Top-p condition preparation, sparse 2D MaskGIT training, CFG sweep, and generation evaluation | [Generation guide](generation/README.md) |

Each component is run from **its own directory**. Configuration paths are relative to that working directory; for example, `../ImageNet/train` in a reconstruction YAML means `repo/ImageNet/train`. The two components have different software environments.

## Reported checkpoints and measurements

The following numbers describe evaluated checkpoints, not an expected outcome of simply running every current config from a fresh checkout.

| Pipeline | Evaluated protocol | Result | Checkpoint relationship |
| --- | --- | --- | --- |
| Reconstruction | ImageNet validation, 50,000 images at 256 × 256; EMA; hard top-p inference with `p=0.994897`, `T=1` | rFID **1.198612**, PSNR **19.5080**, LPIPS **0.222875**; mean **96.124** selected 2D tokens | Six-epoch top-p checkpoint at step **247482**, produced by the historical 5+1 continuation |
| Generation | Class-conditional ImageNet-256, 50,000 samples; 8 TiTok + 8 Halton steps; 2D CFG **2.5** selected by a separate fixed-5k sweep | gFID **1.9209706**; mean **98.95888** generated 2D IDs, in addition to 32 1D IDs | Completed **20,000-update** Halton checkpoint |

The checked-in `reconstruction/configs/top_p_train.yaml` specifies a **direct six-epoch run** from the fixed 20-epoch checkpoint. It is a distinct training lineage from the checkpoint used for the reported reconstruction row. Evaluate its output separately; do not assign the reference metrics to that run. The generation row likewise belongs to its completed 20k checkpoint, not a new run or a 25k continuation. See both component guides for the exact stage descriptions.

## Directory and asset layout

Keep large files out of Git. The root `.gitignore` excludes the directories below. You may use directories or symlinks; the paths consumed by the checked-in configs stay relative to this repository.

~~~text
repo/
├── reconstruction/           # source and YAML configs
├── generation/               # source, scripts, and vendored adapters
├── 1d-tokenizer/             # upstream TiTok source and public L32 weights
├── LlamaGen/                 # upstream LlamaGen source and public VQ weights
├── Halton-MaskGIT/source/    # upstream Halton source
├── ImageNet/
│   ├── train/
│   └── validation/
├── assets/
│   ├── reconstruction/       # evaluated top-p checkpoint when supplied
│   └── generation/           # Halton init, mapper, scorer, caches, FID assets
└── results/                  # generation runs and evaluations
~~~

The repository does **not** redistribute ImageNet, trained checkpoints, generated token/condition caches, or ADM FID reference files. The generation asset manifest at [`generation/configs/h20_assets.json`](generation/configs/h20_assets.json) retains local paths, sizes, and SHA-256 values. Assets without a public source must be supplied separately and verified against those values.

### Checkpoint handoff

| Artifact | Used by | Expected location or identity |
| --- | --- | --- |
| Public TiTok-L32 and LlamaGen VQ weights | Reconstruction training and evaluation | Under `1d-tokenizer/` and `LlamaGen/`; [public-weight helper](reconstruction/download_public_weights.py) |
| Fixed 20-epoch reconstruction checkpoint | Top-p reconstruction initialization | `reconstruction/weights/epoch_0020_step_00199440.pt` by default; SHA-256 `86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446` |
| Evaluated top-p checkpoint at step 247482 | Reconstruction validation and generation conditioning | Set the evaluation `ckpt` path and generation `TOP_P_CKPT` to the same file; SHA-256 `61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819` |
| Official Halton initialization | Generation training | `assets/generation/ImageNet_256_base.pth` in the example below; SHA-256 `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f` |
| Top-p scorer, codebook mapper, GT code cache, condition cache, ADM reference | Generation training and 50k evaluation | Supply under `assets/generation/` and set the paths in the generation command |

The generation code checks the TiTok and top-p checkpoint hashes in [`assets_top_p.py`](generation/experiments/feature_to_token_20260916/assets_top_p.py). Do not replace these with earlier fixed-budget or k-selector weights.

## Reconstruction

### Environment and public weights

From the repository root:

~~~bash
cd reconstruction
conda env create -n mot-reconstruction -f environment.yml
conda activate mot-reconstruction
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch
~~~

The source checkouts and ImageNet tree must occupy the paths shown above. If they are stored elsewhere, update the corresponding relative paths in the YAML files. [`reconstruction/README.md`](reconstruction/README.md) lists the expected upstream modules.

### Training stages

| Stage | Config | Output / handoff |
| --- | --- | --- |
| 1. Train the 1D-to-grid adapter for 2.5 epochs | `configs/adapter_only_2p5epoch.yaml` | `results/adapter_only_2p5epoch/latest.pt` |
| 2. Train a fixed top-96 scorer/refiner for 20 epochs | `configs/fixed96_from_adapter_20epoch.yaml` | Fixed-budget checkpoint; its adapter input is Stage 1 EMA |
| 3. Train end-to-end dynamic top-p for six epochs | `configs/top_p_train.yaml` | `results/top_p_from_fixed_6epoch/`; training uses `p=0.91`, `T=2` |

Run each stage from `reconstruction/` with eight visible GPUs:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_recon.py --config configs/adapter_only_2p5epoch.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/fixed96_from_adapter_20epoch.yaml

# First set "resume" in configs/top_p_train.yaml to the Stage 2 checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/top_p_train.yaml
~~~

Stage 3 defaults to `weights/epoch_0020_step_00199440.pt`. If Stage 2 produced a different filename, change `resume` before launching Stage 3. The clean Stage 1 → 2 → 3 recipe has not been established to reproduce the historical reference row above.

### Validation

Put the checkpoint to evaluate at the `ckpt` path in the selected YAML. Both evaluation configs use EMA weights, dynamic top-p with `p=0.994897` and `T=1`, and no fixed token clamp (`router_min_tokens=router_max_tokens=-1`).

~~~bash
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_5000.yaml

CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_50000.yaml
~~~

The 50k config writes `results/top_p_eval_50000.json`. Its `no_ssim: true` setting means that the reference run does not report SSIM.

## Generation

Generation needs the **same evaluated top-p reconstruction checkpoint** plus its own scorer, mapper, and completed caches. It is not runnable end to end from source files alone. The relevant preparation and model code is in `generation/experiments/feature_to_token_20260916/`; the [generation guide](generation/README.md) maps the entry points.

Prepare a separate Python 3.10/CUDA environment with PyTorch **2.10.0**, torchvision **0.25.0**, and the packages pinned in [`requirements-h20.txt`](generation/requirements-h20.txt) and [`constraints-h20.txt`](generation/constraints-h20.txt). The validated trainer targets eight H20 GPUs and requires W&B online login. From `generation/`, `python -m h20.preflight --environment-only` checks the pinned runtime versions before training.

After supplying the assets, run the 20k launcher from the repository root:

~~~bash
cd generation
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
~~~

The launcher refuses an existing output directory and uses eight GPUs, global batch 256, seed 0, and 20,000 optimizer updates. It writes a rolling `latest.pt`. The model generates 2D IDs **only at the selected top-p positions**; the mapper's unselected proxy IDs are not substituted into the final decoder features.

The final generation protocol first sweeps 2D CFG on a fixed 5k cohort, then evaluates the chosen setting on 50k samples. After a completed 20k run, the following command performs both steps and writes a `summary.json` under its output directory:

~~~bash
python experiments/feature_to_token_20260916/sweep_top_p_cfg.py \
  --checkpoint "$TOP_P_OUTPUT/latest.pt" \
  --mapper-checkpoint "$TOP_P_MAPPER_CKPT" \
  --scorer-checkpoint "$TOP_P_SCORER_CKPT" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/cfg_sweep_20k"
~~~

For the reported 20k checkpoint, this protocol selected CFG **2.5**. It does not evaluate a 25k continuation.

## Scope of verification

The source tree and configuration paths can be inspected without the external assets. Reproducing the numerical rows requires the exact checkpoints, caches, ImageNet split, and FID assets above. The two component guides distinguish measured checkpoint results from proposed clean-run recipes. Third-party code under `generation/third_party/` keeps its original provenance and license files.
