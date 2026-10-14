<div align="center">

# MixTok

**Adaptive Visual Tokenization with Global 1D Tokens and Sparse 2D Refinement**

[arXiv](https://arxiv.org/abs/2610.13181) | [Project Page](https://lcglichenge.github.io/MixTok-page/) | [🤗 Models](https://huggingface.co/Chloeeeeeeee123/MixTok)

Chenge Li · Yefei He · Xinyu Liu · Feng Chen · Hong Zhou · Bohan Zhuang

Zhejiang University · The Hong Kong University of Science and Technology · University of Adelaide

<img src="https://lcglichenge.github.io/MixTok-page/assets/images/generation-samples.webp" alt="MixTok generation: 1D base images, selected refinement locations, and final images" width="95%" />

*Content-adaptive generation: global structure from 1D tokens, local detail from sparse 2D refinement.*

</div>

MixTok combines **32 global 1D tokens** with an **image-dependent number of local 2D tokens**. A learned router selects where to refine and how many tokens to use. A shared decoder reconstructs images from the mixed feature grid; two-stage masked generation uses the same representation for class-conditional image synthesis.

On **ImageNet 256 × 256**, MixTok achieves **1.195 rFID** at approximately **128 tokens** and **1.9511 gFID** at approximately **131 tokens**. This repository provides the reconstruction and continuous-embedding generation pipelines, training configurations, and evaluation tools.

## Table of Contents

1. [Results](#1-results)
2. [Models & Weights](#2-models--weights)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Method](#5-method)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Repository Layout](#8-repository-layout)
9. [Citation](#9-citation)

## 1. Results

Results below follow the [project page](https://lcglichenge.github.io/MixTok-page/#results). Reconstruction uses all **50,000 ImageNet validation images**; MixTok's generation gFID is averaged over **four independent 50,000-image sampling runs**. Token counts include both global and refinement tokens.

### Image reconstruction

| Tokenizer | Avg. tokens | rFID ↓ | PSNR ↑ |
| --- | ---: | ---: | ---: |
| TiTok S-128 | 128 | 1.7222 | 17.72 |
| TiTok BL-128 VQ | 128 | 1.5063 | **19.83** |
| One-D-Piece L-256 | 128 | 1.4406 | 17.71 |
| FlexTok d18-d28 | 128 | 1.1422 | 16.76 |
| DetailFlow-16 | 128 | **1.1370** | 19.46 |
| **MixTok** | **≈128** | **1.1950** | **19.51** |

<p align="center">
  <img src="docs/recon_image.png" alt="Ground truth, 1D reconstruction, selected refinement locations, and MixTok reconstruction" width="95%" />
</p>

### Class-conditional generation and efficiency

| Generator | Params | Tokens | Steps | gFID ↓ | Latency ↓ | FLOPs ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdaGen-MaskGIT-L | 206M | 256 | 16 | 2.4883 | 372.0 ms | 3.623 T |
| TiTok-S-128 | 295M | 128 | 64 | 1.9994 | 1018.4 ms | 9.949 T |
| Halton-MaskGIT-L† | 480M | 576 | 32 | 2.5143 | 591.5 ms | 26.690 T |
| One-D-Piece-L-256 | 177M | 256 | 16 | 2.4144 | 350.1 ms | 3.526 T |
| LPD-L-256 | 337M | 256 | 20 | 2.3863 | **215.5 ms** | 0.879 T |
| LlamaGen-L | 343M | 256 | 256 | 3.6803 | 4951.8 ms | **0.599 T** |
| DetailFlow-32 | 326M | 256 | 39 | 2.6315 | 669.9 ms | 5.627 T |
| **MixTok** | **321M** | **130.66** | **8+8** | **1.9511** | **270.1 ms** | **1.452 T** |

Latency is median end-to-end generation time at batch size 1 on one **RTX 5090**, including RGB decoding. FLOPs are per image, counting multiply-add as two operations. †Halton-MaskGIT-L uses its native 384 × 384 output; all other rows use 256 × 256.

## 2. Models & Weights

Released model files are hosted on [🤗 Chloeeeeeeee123/MixTok](https://huggingface.co/Chloeeeeeeee123/MixTok/tree/main):

| Component | File |
| --- | --- |
| Reconstruction EMA model | [`recon/mixtok_recon.pt`](https://huggingface.co/Chloeeeeeeee123/MixTok/blob/main/recon/mixtok_recon.pt) |
| Inference-only generation bundle | [`gene/mixtok_gen.pt`](https://huggingface.co/Chloeeeeeeee123/MixTok/blob/main/gene/mixtok_gen.pt) |
| Generation bundle manifest | [`gene/mixtok_gen.pt.manifest.json`](https://huggingface.co/Chloeeeeeeee123/MixTok/blob/main/gene/mixtok_gen.pt.manifest.json) |

Download the release files from the repository root after installing the environment:

```bash
hf download Chloeeeeeeee123/MixTok --revision 1906bb6281940c152d1e25ea7049858c1a43f603 --local-dir assets/released
```

The model-only exports do not contain optimizer or scheduler state. The generation evaluator accepts the public generation bundle directly, together with the public reconstruction EMA weights. Separate Halton, adapter, scorer, and top-p training checkpoints remain supported. See the [generation guide](generation/README.md) for the export format and asset requirements.

### Training checkpoints and supporting assets

| Artifact | Used by | Expected location or identity |
| --- | --- | --- |
| Public TiTok-L32 and LlamaGen VQ weights | Reconstruction training and evaluation | Under `1d-tokenizer/` and `LlamaGen/`; [public-weight helper](reconstruction/download_public_weights.py) |
| Fixed 20-epoch reconstruction checkpoint | Top-p reconstruction initialization | `reconstruction/weights/epoch_0020_step_00199440.pt` by default; SHA-256 `86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446` |
| Evaluated top-p checkpoint at step 247482 | Reconstruction validation and generation conditioning | Set the evaluation `ckpt` path and generation `TOP_P_CKPT` to the same file; SHA-256 `61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819` |
| Official Halton initialization | Generation training | `assets/generation/ImageNet_256_base.pth` in the example below; SHA-256 `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f` |
| 1D-only generation scorer, teacher and route caches, GT codes, ADM reference | Embedding generation training and evaluation | Train the scorer from the Stage 3 EMA checkpoint using the [generation guide](generation/README.md); accepted 20k scorer SHA-256 `52fddf824047fa9c207c046152275fc38715beb2020e0613ed9e7e73f2893a35` |

The generation code checks the TiTok and top-p checkpoint hashes in [`assets_top_p.py`](generation/experiments/continuous_halton_embedding_20260923/assets_top_p.py). Do not replace these with earlier fixed-budget or k-selector weights.

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/LCGLiChenge/MixTok.git
cd MixTok
```

The reconstruction and generation pipelines use **separate environments**. Run component commands from their respective directories; configuration paths are relative to that working directory.

### Reconstruction

From the repository root, check out the pinned upstream sources and prepare public weights:

```bash
git clone https://github.com/bytedance/1d-tokenizer.git 1d-tokenizer
git -C 1d-tokenizer checkout 942a96fbdd873780179d1b78d5462911528bf8c8
git clone https://github.com/FoundationVision/LlamaGen.git LlamaGen
git -C LlamaGen checkout ce98ec41803a74a90ce68c40ababa9eaeffeb4ec

cd reconstruction
conda env create -n mixtok-reconstruction -f environment.yml
conda activate mixtok-reconstruction
python download_public_weights.py --project-root .. --torch-cache-root ../.cache/torch
```

Prepare ImageNet under `ImageNet/train/` and `ImageNet/validation/`, or update the YAML paths. See the [reconstruction guide](reconstruction/README.md) for the expected source modules and checkpoint setup.

### Generation

From the repository root:

```bash
cd generation
conda create -n mixtok-generation python=3.10 -y
conda activate mixtok-generation
python -m pip install -c constraints-h20.txt torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements-h20.txt -c constraints-h20.txt
python -m pip install tensorflow==2.20.0 -c constraints-h20.txt
python -m h20.preflight --environment-only

mkdir -p ../Halton-MaskGIT
git clone https://github.com/valeoai/Halton-MaskGIT.git ../Halton-MaskGIT/source
git -C ../Halton-MaskGIT/source checkout f61b0a1314717004dc7487531fd16a8bb71e1888
```

The paper training scripts target **eight H20 GPUs** and use W&B offline logging by default. Supply the official Halton initialization, public TiTok weights, reconstruction checkpoint, scorer, caches, and ADM FID assets described in the [generation guide](generation/README.md).

## 4. Quick Start

### Evaluate reconstruction

After completing the reconstruction setup, set `ckpt` in `reconstruction/configs/top_p_eval_5000.yaml` to your reconstruction checkpoint and confirm `data_path` points to ImageNet validation. From the repository root:

```bash
conda activate mixtok-reconstruction
cd reconstruction
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_5000.yaml
```

This screens the checkpoint on 5,000 images. The [full evaluation](#7-evaluation) uses 50,000 images.

### Generate an eight-image smoke run

After downloading the public release and preparing the [supporting TiTok and ADM assets](generation/README.md#evaluate-the-public-release), run from `generation/` in the generation environment:

```bash
export MOTAR_ASSETS=../assets/generation
export MOTAR_RESULTS=../results/generation
export HALTON_UPSTREAM_ROOT=../Halton-MaskGIT/source
export MIXTOK_GENERATOR=../assets/released/gene/mixtok_gen.pt
export TOP_P_CKPT=../assets/released/recon/mixtok_recon.pt

python experiments/continuous_halton_embedding_20260923/parallel_eval_compact.py \
  --checkpoint "$MIXTOK_GENERATOR" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/continuous_eval_smoke8" \
  --n 8 --workers 1 --allowed-gpus 0
```

Use a new output directory for each run. This checks generation and feature extraction; eight images do not produce a reported FID score. TiTok weights must be under `$MOTAR_ASSETS/weights/`, and ADM graph/reference files under `$MOTAR_ASSETS/fid/`.

The public bundle includes the Halton model, embedding adapter, and scorer, so no separate adapter or scorer paths are needed. The evaluator verifies the pinned public file hashes and strict-loads the inference tensors. The public reconstruction export omits its training step and source metadata; these are recorded as unknown in the evaluation manifest and do not affect sampling or FID computation. The ADM evaluator also requires a GPU-capable TensorFlow installation.

## 5. Method

1. **Global representation.** TiTok-L32 encodes 32 1D tokens. An adapter expands them into a continuous 16 × 16 feature grid.
2. **Adaptive refinement.** A scorer ranks the 256 grid locations. After temperature-scaled softmax, top-p selects the smallest ranked prefix whose cumulative probability reaches `p`. The selected count `K(x)` changes with each image.
3. **Mixed decoding.** Selected locations receive 2D codebook features; unselected locations keep their continuous 1D-derived features. A shared decoder renders the mixed grid.

<p align="center">
  <img src="docs/recon_main.png" width="94%" alt="MixTok reconstruction: global 1D features, a reconstruction grid scorer, top-p selection, sparse 2D features, and a shared decoder." />
</p>

*MixTok reconstruction overview. The scorer and top-p controller select spatial locations for 2D refinement.*

For **generation**, a TiTok MaskGIT first samples the 32 1D IDs. A scorer selects spatial locations using their 1D feature grid. An embedding adapter maps that grid into continuous Halton context embeddings, and sparse Halton MaskGIT generates 2D IDs only at the selected locations. Unselected decoder locations keep their continuous 1D-derived features.

<p align="center">
  <img src="docs/maskgit.png" width="88%" alt="MixTok generation pipeline with a 1D MaskGIT, top-p grid selection, continuous embedding adapter, sparse 2D MaskGIT, and shared decoder." />
</p>

*Class-conditional generation with continuous spatial context embeddings. The adapter maps 1D features into the 2D MaskGIT input space; only selected 2D tokens are generated.*

The fixed **96/256** allocation belongs to reconstruction warm-start training. Final reconstruction and generation use image-dependent top-p selection, so the number of selected 2D locations changes with each image.

## 6. Training

### Reconstruction

| Stage | Config | Output / handoff |
| --- | --- | --- |
| 1. Train the 1D-to-grid adapter for 2.5 epochs | `configs/adapter_only_2p5epoch.yaml` | `results/adapter_only_2p5epoch/latest.pt` |
| 2. Train a fixed top-96 scorer/refiner for 20 epochs | `configs/fixed96_from_adapter_20epoch.yaml` | Fixed-budget checkpoint; its adapter input is Stage 1 EMA |
| 3. Train end-to-end dynamic top-p for six epochs | `configs/top_p_train.yaml` | `results/top_p_from_fixed_6epoch/`; training uses `p=0.91`, `T=2` |

Run each stage from `reconstruction/` with eight visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_recon.py --config configs/adapter_only_2p5epoch.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/fixed96_from_adapter_20epoch.yaml

# First set "resume" in configs/top_p_train.yaml to the Stage 2 checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/top_p_train.yaml
```

Stage 3 runs for 6 epochs. It defaults to `weights/epoch_0020_step_00199440.pt`; if Stage 2 produced a different filename, change `resume` before launching.

### Generation

Complete scorer training and cache building in the [generation guide](generation/README.md), then run these commands from the repository root. The `MOTAR_*` environment variable names are part of the script interface:

```bash
cd generation
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
```

Each launcher runs for 20,000 optimizer updates and writes `latest.pt`. Train the scorer and build paired codes, teacher targets, and route caches first, following the [generation guide](generation/README.md).

## 7. Evaluation

### Reconstruction: ImageNet validation

Run from `reconstruction/` with its environment active and the evaluation YAML's `ckpt` pointing to the intended model:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_50000.yaml
```

The 50k configuration uses EMA weights, hard top-p with `p=0.994897` and `T=1`, and no fixed token clamp (`router_min_tokens=router_max_tokens=-1`). It writes `results/top_p_eval_50000.json`; `no_ssim: true` means this reference run does not report SSIM. Keep GPU type, precision, and evaluation batch size consistent when comparing runs.

### Generation: 50,000 samples

With the paths from [Quick Start](#4-quick-start), run from `generation/`:

```bash
python experiments/continuous_halton_embedding_20260923/parallel_eval_compact.py \
  --checkpoint "$MIXTOK_GENERATOR" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/continuous_eval_paper50k" \
  --n 50000 --batch-size 64 --workers 1 --allowed-gpus 0 \
  --stage1-guidance-scale 4.5 --cfg-w 3.0
```

The default sampling recipe uses **8 steps per stage**, Stage 1 guidance **4.5**, Stage 2 CFG **3.0**, scorer temperature **2**, and router top-p **0.75**. The generation batch size defaults to 8; the example uses 64. Changing it changes the RNG batches and sampled images, so record the actual batch size with each result. FP32 feature decoding and ADM feature extraction use chunks of at most eight images. A single run produces one 50k evaluation; the result table reports the project's four-run average. See the [generation guide](generation/README.md) for worker counts, cache identity checks, and checkpoint export details.

## 8. Repository Layout

```text
MixTok/
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
│   └── generation/           # Halton init, scorer, teacher/route caches, FID assets
└── results/                  # generation runs and evaluations
```

The upstream source checkouts, ImageNet, weights, caches, and run outputs shown above are external assets and excluded by `.gitignore`. The legacy [`generation/configs/h20_assets.json`](generation/configs/h20_assets.json) lists public TiTok weights but is not a complete manifest for the continuous-embedding experiment.

| Guide | Contents |
| --- | --- |
| [Reconstruction](reconstruction/README.md) | Environment, pinned source revisions, three-stage training, validation |
| [Generation](generation/README.md) | Scorer distillation, paired codes, teacher/route caches, training, evaluation, model export |

## 9. Citation

If you find MixTok useful, please cite our paper:

```bibtex
@misc{li2026mixtok,
  title = {MixTok: Adaptive Visual Tokenization with Global 1D Tokens and Sparse 2D Refinement},
  author = {Li, Chenge and He, Yefei and Liu, Xinyu and Chen, Feng and Zhou, Hong and Zhuang, Bohan},
  year = {2026},
  eprint = {2610.13181},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2610.13181}
}
```

### Acknowledgments

MixTok builds on [TiTok](https://github.com/bytedance/1d-tokenizer), [LlamaGen](https://github.com/FoundationVision/LlamaGen), and [Halton-MaskGIT](https://github.com/valeoai/Halton-MaskGIT). Vendored components under [`generation/third_party/`](generation/third_party/) retain their original provenance and license files.
