# MixTok Generation: Continuous Embedding MaskGIT

This directory contains the **continuous embedding** generation path. The official TiTok MaskGIT samples 32 global 1D IDs. The frozen top-p tokenizer expands them into a 16 × 16 feature grid, and a scorer selects a variable number of refinement locations. A trained adapter maps the 1D feature grid to 256 continuous, 768-dimensional Halton input embeddings. Halton MaskGIT generates discrete 2D IDs only at the selected locations; the shared decoder keeps the 1D-derived features everywhere else.

![MixTok generation with continuous embedding context, top-p spatial selection, sparse 2D MaskGIT, and a shared decoder.](../docs/maskgit.png)

## Requirements

- Python 3.10, eight H20 GPUs for the paper training configuration, PyTorch 2.10.0, torchvision 0.25.0, and the packages in [`requirements-h20.txt`](requirements-h20.txt) and [`constraints-h20.txt`](constraints-h20.txt). Training logs to W&B offline by default; online logging is optional.
- Official Halton-MaskGIT source and `ImageNet_256_base.pth` from [`llvictorll/Halton-MaskGIT`](https://huggingface.co/llvictorll/Halton-MaskGIT). The required checkpoint SHA-256 is `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f`.
- Official TiTok-L32 weights and the evaluated top-p reconstruction checkpoint at step 247482, loaded from `model_ema`. The top-p checkpoint SHA-256 is `61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819`. The full checkpoint or its SHA-verified EMA inference export can supply the frozen tokenizer modules used by generation. [`export_top_p_inference.py`](experiments/continuous_halton_embedding_20260923/export_top_p_inference.py) creates the smaller export.
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

The first stage trains the embedding adapter against the official Halton embedding table using the teacher-token cache. The second stage freezes that adapter and trains sparse Halton MaskGIT using the route-only cache. `sweep_cfg.py` selects CFG on a fixed 5k cohort, then evaluates the selected setting on 50k samples. Both training stages write a rolling `latest.pt`. To run a diagnostic training pass on a different CUDA GPU, pass `--allow-non-h20` and lower `--min-free-gib` to the available memory. This changes the hardware used for the run and may change sampled outputs.

When evaluating a checkpoint trained on another machine, set `HALTON_UPSTREAM_ROOT` to the local Halton-MaskGIT source checkout. The evaluator checks that its six upstream source files match the hashes saved in the checkpoint before loading the model. `parallel_eval_compact.py`, the cache builders, and the training scripts accept the SHA-verified top-p EMA export. Training automatically uses the compact evaluator for that export. The paired-code extractor records both the artifact hash and its source checkpoint hash, so a copied cache can be verified without relying on its original absolute paths.

### Evaluate an existing checkpoint

Place the TiTok weights under `$MOTAR_ASSETS/weights/` and the ADM graph and reference under `$MOTAR_ASSETS/fid/`. Set the paths above to the matching Halton, adapter, scorer, and top-p checkpoints. An eight-image GPU smoke test is:

```bash
python experiments/continuous_halton_embedding_20260923/parallel_eval_compact.py \
  --checkpoint "$MOTAR_RESULTS/continuous_halton20k_seed0/latest.pt" \
  --adapter-checkpoint "$CONTINUOUS_ADAPTER_CKPT" \
  --scorer-checkpoint "$TOP_P_SCORER_CKPT" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/continuous_eval_smoke8" \
  --n 8 --workers 1 --allowed-gpus 0
```

Use `--n 5000` or `--n 50000` for ADM FID, with `--workers 1`, `2`, `4`, or `8` and a matching `--allowed-gpus` list. Each output directory must be new. The eight-image smoke confirms generation and feature extraction but does not report FID.

For the paper's generation setting, use 50,000 samples with TiTok guidance 4.75 and Halton CFG 3.375. The remaining sampling settings use the evaluator defaults (seed 20260914, eight steps per stage, router top-p 0.75):

```bash
python experiments/continuous_halton_embedding_20260923/parallel_eval_compact.py \
  --checkpoint "$MOTAR_RESULTS/continuous_halton20k_seed0/latest.pt" \
  --adapter-checkpoint "$CONTINUOUS_ADAPTER_CKPT" \
  --scorer-checkpoint "$TOP_P_SCORER_CKPT" \
  --top-p-checkpoint "$TOP_P_CKPT" \
  --output "$MOTAR_RESULTS/continuous_eval_paper50k" \
  --n 50000 --workers 4 --allowed-gpus 0,1,2,3 \
  --stage1-guidance-scale 4.75 --cfg-w 3.375
```

### Build the caches

Start with ImageNet train in the standard 1,000-class `ImageFolder` layout. Extract paired codes from the official TiTok tokenizer and the verified top-p EMA VQ weights. Match `--nproc_per_node` to the GPUs exposed by `CUDA_VISIBLE_DEVICES`; the paper run used eight GPUs.

```bash
export IMAGENET_TRAIN=../assets/ImageNet/train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  scripts/extract_codes_top_p.py \
  --data-path "$IMAGENET_TRAIN" --output-root "$TOP_P_GT2D_CACHE" \
  --mot-ckpt "$TOP_P_CKPT" --mot-state-key model_ema \
  --titok-root third_party/TiTok \
  --titok-config third_party/TiTok/configs/infer/TiTok/titok_l32.yaml \
  --titok-ckpt "$MOTAR_ASSETS/weights/tokenizer_titok_l32.bin" \
  --llamagen-root third_party/LlamaGen \
  --aug-mode adm --mixed-precision none --batch-size 32 --num-workers 4
```

[`cache_teacher_targets.py`](experiments/continuous_halton_embedding_20260923/cache_teacher_targets.py) builds the teacher-token cache by decoding each 1D base image and re-encoding it with the frozen top-p VQ model. [`cache_routes.py`](experiments/continuous_halton_embedding_20260923/cache_routes.py) applies the accepted scorer to the frozen 1D features and records only the selected top-p positions. Neither cache builder trains a generator. Both use `prepare`, `extract`, and `finalize` modes.

Build the teacher cache before training the adapter:

```bash
python experiments/continuous_halton_embedding_20260923/cache_teacher_targets.py prepare \
  --assets-root "$MOTAR_ASSETS" --source-root "$TOP_P_GT2D_CACHE" \
  --top-p-ckpt "$TOP_P_CKPT" --output "$TOP_P_TEACHER_PROXY_CACHE" --world-size 8

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$rank" python experiments/continuous_halton_embedding_20260923/cache_teacher_targets.py extract \
    --assets-root "$MOTAR_ASSETS" --source-root "$TOP_P_GT2D_CACHE" \
    --top-p-ckpt "$TOP_P_CKPT" --output "$TOP_P_TEACHER_PROXY_CACHE" \
    --world-size 8 --rank "$rank" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done

python experiments/continuous_halton_embedding_20260923/cache_teacher_targets.py finalize \
  --assets-root "$MOTAR_ASSETS" --source-root "$TOP_P_GT2D_CACHE" \
  --top-p-ckpt "$TOP_P_CKPT" --output "$TOP_P_TEACHER_PROXY_CACHE" --world-size 8
```

After the paired code cache is complete, build the selected-route cache:

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

Cache identities and row alignment are validated again when training starts. The route cache records SHA-256 hashes of the source metadata, 1D codes, 2D GT codes, and labels; older route caches require the paired source `manifest.sha256` for the 2D GT and label checks. The paired-code extractor records its batch size, GPU type, precision, and worker count in `run_config.json`; keep those settings when using `--resume`. Changing the batch size can change a few VQ IDs near codebook boundaries. The extractor writes code arrays before marking rows complete, so an interrupted run reprocesses only uncommitted rows. For a partial cache created before `run_config.json` existed, first verify the original command and add `--adopt-legacy-partial` to its initial resume. The teacher and route extractors also resume from written-row bitmaps and require the same `--batch` value used at `prepare` time.

## Source layout

- [`experiments/continuous_halton_embedding_20260923/`](experiments/continuous_halton_embedding_20260923/): embedding adapter, continuous Halton model, cache builders, training, asynchronous evaluation, CFG sweep, and compact-checkpoint tools.
- [`scripts/launch_continuous_adapter20k.sh`](scripts/launch_continuous_adapter20k.sh) and [`scripts/launch_continuous_halton20k.sh`](scripts/launch_continuous_halton20k.sh): eight-GPU training launchers.
- [`scripts/extract_codes_top_p.py`](scripts/extract_codes_top_p.py): paired top-p code extraction.
- [`third_party/`](third_party/): vendored upstream components with their provenance and license files.
