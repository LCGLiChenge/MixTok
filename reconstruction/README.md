# MixTok Reconstruction: Dynamic Top-p Token Allocation

This directory contains the MixTok reconstruction pipeline: an end-to-end Router selects a variable number of 2D refinement grids with a differentiable top-p mask, while the tokenizer and decoder are jointly optimized.

The fixed-budget warm-start is included below. Earlier exploratory search and AR experiments are outside this reconstruction directory.

## Fixed 0.375 pretraining

The fixed-budget pretraining pipeline is included in `configs/fixed96_from_adapter_20epoch.yaml`. It fixes the refinement budget to 96 of 256 grids, trains the Grid Scorer alone for 0.5 epoch, performs one additional epoch of joint reconstruction training without GAN loss, and enables GAN training from epoch 1.5 through epoch 20.

## Top-p inference settings

The dynamic top-p training config uses temperature `2.0` and `p=0.91`. The evaluation configs use EMA weights and hard top-p inference with temperature `1.0` and `p=0.994897`; the number of selected grids varies by image. Evaluation commands are below. Checkpoint weights and evaluation outputs are kept outside Git.

## Repository layout

```text
configs/
  adapter_only_2p5epoch.yaml          # Stage 1: frozen LlamaGen, train only the 1D adapter
  fixed96_from_adapter_20epoch.yaml   # Stage 2: fixed top-96 Grid Scorer training
  top_p_train.yaml          # 6-epoch recipe from the fixed checkpoint
  top_p_eval_5000.yaml      # quick checkpoint screening
  top_p_eval_50000.yaml     # final ImageNet validation protocol
models/
  titok_to_llamagen.py      # shared TiTok/LlamaGen representation module
train_titok_llamagen_recon.py
train_titok_llamagen_decoder_adapt_global_gain_texture.py
train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py
eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py
download_public_weights.py
environment.yml
```

`train_titok_llamagen_decoder_adapt_global_gain_texture.py` is retained because the evaluator imports shared spatial-score and feature-conversion utilities from it. The active allocation mode is `router_e2e_top_p`.

## Reconstruction initialization: adapter 2.5 + fixed 20 epochs

The reported reconstruction training starts from public TiTok-L32 and LlamaGen VQ checkpoints, trains the adapter for 2.5 epochs, then trains the fixed-budget model for 20 epochs. These are the first two stages of the evaluated model, before top-p continuation; the checked-in configs describe their stage structure.

Stage 1 trains only `TiTokToLlamaGenLatentDecoder` for 2.5 ImageNet epochs. The TiTok encoder/quantizer and the complete LlamaGen VQ model, including its decoder, stay frozen. The adapter maps 32 TiTok tokens to a `256 x 16 x 16` feature map that the frozen LlamaGen decoder reconstructs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_recon.py \
  --config configs/adapter_only_2p5epoch.yaml
```

The resulting adapter EMA is saved in:

```text
results/adapter_only_2p5epoch/latest.pt
```

Stage 2 initializes only the adapter from that checkpoint and trains fixed-budget MixTok for 20 epochs. Every image always receives exactly 96 refinement grids ranked by the Grid Scorer:

- epochs `0.0-0.5`: train only the Grid Scorer with gain-map supervision;
- epochs `0.5-1.5`: joint fixed-top-96 reconstruction training without GAN loss;
- epochs `1.5-20.0`: joint fixed-top-96 training with GAN, DINO, CLIP, and discriminator feature matching.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/fixed96_from_adapter_20epoch.yaml
```

This is the exact checkpoint handoff used by the configs:

```text
public TiTok/LlamaGen weights
  -> adapter_only_2p5epoch/latest.pt
  -> fixed96_from_adapter_20epoch
```

Stage 3 trains dynamic top-p for 6 epochs.

## Environment

```bash
conda env create -n mixtok-reconstruction -f environment.yml
conda activate mixtok-reconstruction
```

The default configs are run from `reconstruction/` and assume these sibling paths in the merged repository. Stage 2 and Stage 3 cache the scanned ImageNet file list under their output directories, so the ImageNet tree can be read-only. Set `--dataset-manifest` to a shared writable path if both stages should reuse one index:

```text
repo/
  reconstruction/
  1d-tokenizer/
    modeling/titok.py
    configs/infer/TiTok/titok_l32.yaml
    tokenizer_titok_l32.bin
  LlamaGen/
    tokenizer/tokenizer_image/vq_model.py
    pretrained_models/vq_ds16_c2i.pt
  ImageNet/
    train/
    validation/
  .cache/
    torch/hub/facebookresearch_dinov2_main/
    open_clip/
```

Public weights and feature backbones can be prepared with:

```bash
python download_public_weights.py \
  --project-root .. \
  --torch-cache-root ../.cache/torch
```

## Checkpoint

Training and evaluation checkpoints are not stored in Git. Place an exact-resume checkpoint at:

```text
weights/epoch_0020_step_00199440.pt
```

Provide the checkpoint locally and verify it with:

```bash
printf '%s  %s\n' 86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446 weights/epoch_0020_step_00199440.pt | sha256sum -c -
```

Expected size: `6,400,628,829` bytes.

With the default 8-GPU ImageNet-1K setup, the sixth checkpoint is written to:

```text
results/top_p_from_fixed_6epoch/epoch_0006_step_00247482.pt
```

Both evaluation configs point to this file by default. If the world size or dataset cardinality changes, update `ckpt` to the filename produced by the training run.

The training config starts from the fixed checkpoint EMA weights (`resume_use_ema_model: true`), retains the checkpoint discriminator, resets the optimizer and Router allocation state, and then runs six complete epochs. Override `resume`, data paths, or output paths in the YAML when using another layout.

## Train

The current recipe resumes from the fixed 20-epoch checkpoint and trains for six epochs with 8 GPUs, batch size 20 per GPU, full GAN loss from the first resumed step, top-p temperature 2 with `p=0.91`, and one checkpoint per epoch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py \
  --config configs/top_p_train.yaml
```

## Evaluate

Screen a checkpoint on 5,000 validation images:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_5000.yaml
```

Run the final 50,000-image evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py \
  --config configs/top_p_eval_50000.yaml
```

Both evaluation configs use EMA weights and the calibrated hard-inference setting (`T=1`, `p=0.994897`).
Record the GPU model and evaluation `batch_size` for each run. BF16 outputs can change with batch size, so keep both fixed when comparing repeated runs; the 5,000-image screen does not replace the 50,000-image result.
