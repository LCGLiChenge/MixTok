#!/usr/bin/env bash
set -euo pipefail

: "${MOTAR_ASSETS:?Set the frozen asset directory}"
: "${MOTAR_RESULTS:?Set an output directory outside this repository}"
: "${HALTON_UPSTREAM_ROOT:?Set the official Halton source directory}"
: "${HALTON_OFFICIAL_CKPT:?Set the official Halton checkpoint}"
: "${TOP_P_CKPT:?Set the top-p tokenizer checkpoint}"
: "${TOP_P_GT2D_CACHE:?Set the completed top-p GT cache}"
: "${TOP_P_CONDITION_CACHE:?Set the completed top-p condition cache}"
: "${TOP_P_MAPPER_CKPT:?Set the 20k top-p mapper checkpoint}"
: "${TOP_P_SCORER_CKPT:?Set the accepted top-p scorer checkpoint}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${TOP_P_OUTPUT:-$MOTAR_RESULTS/halton20k_top_p_seed0}"
test -f "$TOP_P_GT2D_CACHE/manifest.sha256"
test -f "$TOP_P_CONDITION_CACHE/summary.json"
test -f "$TOP_P_MAPPER_CKPT"
test -f "$TOP_P_SCORER_CKPT"
test -f "$TOP_P_CKPT"
test -f "$HALTON_OFFICIAL_CKPT"
test ! -e "$output"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MOTAR_EVAL_WORKERS=8
export MOTAR_EVAL_ALLOWED_GPUS=0,1,2,3,4,5,6,7
cd "$repo_root"

exec torchrun --standalone --nproc_per_node=8 \
  experiments/feature_to_token_20260916/train_halton_top_p.py \
  --upstream-root "$HALTON_UPSTREAM_ROOT" \
  --pretrained "$HALTON_OFFICIAL_CKPT" \
  --source-root "$TOP_P_GT2D_CACHE" \
  --cache-root "$TOP_P_CONDITION_CACHE" \
  --top-p-ckpt "$TOP_P_CKPT" \
  --mapper-ckpt "$TOP_P_MAPPER_CKPT" \
  --scorer-ckpt "$TOP_P_SCORER_CKPT" \
  --output "$output" \
  --epochs 50 --stop-after 20000 --micro 32 --seed 0 \
  --lr 2e-5 --warmup 2500 --lr-schedule budget-cosine \
  --lr-hold-until 4000 --lr-decay-until 20000 --min-lr 2e-6 \
  --mask-objective mixed-prefix \
  --eval-cfg-w 2.75 --eval-stage2-steps 8 \
  --extra-eval-steps 2000,4000,8000,12000,16000,20000 \
  --eval-gpu auto --log-every 20 \
  --wandb-project mot-maskgit-review
