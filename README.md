# MoT: anonymous review code

This repository combines the reconstruction and MaskGIT generation code for the paper. The source files are copied into a new repository without the original Git history. The package excludes the original authors' account links, machine-specific paths, datasets, model weights, and experiment logs. Third-party licenses and upstream attributions are retained.

## Layout

- `reconstruction/`: top-p reconstruction training and ImageNet evaluation.
- `generation/`: sparse-2D MaskGIT training, sampling and evaluation.

Run each component from its own directory. Paths in configs and commands are relative to that directory. External dependencies and datasets should be supplied as sibling directories at the root of this repository:

```text
repo/
  reconstruction/
  generation/
  1d-tokenizer/          # upstream TiTok source; local clone or symlink
  LlamaGen/              # upstream LlamaGen source; local clone or symlink
  Halton-MaskGIT/        # upstream generation source; local clone or symlink
  ImageNet/              # train/ and validation/; local symlink
  assets/                # checkpoints and cached conditions; ignored by Git
  results/               # training and evaluation outputs; ignored by Git
```

The two components have separate dependency sets; follow their own READMEs and environment files. Checkpoints and caches required to reproduce the reported numbers are not included. In particular, the generation pipeline needs the top-p reconstruction checkpoint and its separately trained mapper/scorer and condition cache. Hashes in the component documentation and manifest identify the expected artifacts.

## Reconstruction

```bash
cd reconstruction
conda env create -n mot-reconstruction -f environment.yml
conda activate mot-reconstruction
# Place upstream code and ImageNet in the sibling directories shown above.
# Place checkpoints under reconstruction/weights/ or update the relative config paths.
torchrun --standalone --nproc_per_node=8 train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic.py --config configs/top_p_train.yaml
python eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py --config configs/top_p_eval_50000.yaml
```

## Generation

```bash
cd generation
# Supply the files described in generation/README.md under ../assets/.
# The launcher reads paths from environment variables; examples use ../assets and ../results.
bash scripts/launch_top_p_halton20k.sh
```

See `reconstruction/README.md` and `generation/README.md` for the exact recipes, checkpoint lineage, evaluation protocols, and known asset requirements.
