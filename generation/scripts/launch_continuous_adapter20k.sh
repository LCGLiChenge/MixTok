#!/usr/bin/env bash
set -euo pipefail

: "${MOTAR_ASSETS:?Set the frozen asset directory}"
: "${MOTAR_RESULTS:?Set an output directory outside this repository}"
: "${HALTON_OFFICIAL_CKPT:?Set the official Halton checkpoint}"
: "${TOP_P_CKPT:?Set the top-p tokenizer checkpoint}"
: "${TOP_P_GT2D_CACHE:?Set the completed paired top-p GT cache}"
: "${TOP_P_TEACHER_PROXY_CACHE:?Set the completed teacher-proxy cache}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${CONTINUOUS_ADAPTER_OUTPUT:-$MOTAR_RESULTS/continuous_adapter20k_seed20260918}"
test -f "$HALTON_OFFICIAL_CKPT"
test -f "$TOP_P_CKPT"
test -f "$TOP_P_GT2D_CACHE/titok_codes.npy"
test -f "$TOP_P_TEACHER_PROXY_CACHE/proxy_codes.npy"
test ! -e "$output"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
cd "$repo_root"

exec torchrun --standalone --nproc_per_node=8 \
  -m experiments.continuous_halton_embedding_20260923.train_adapter \
  --source-root "$TOP_P_GT2D_CACHE" \
  --teacher-root "$TOP_P_TEACHER_PROXY_CACHE" \
  --top-p-ckpt "$TOP_P_CKPT" --pretrained "$HALTON_OFFICIAL_CKPT" \
  --assets-root "$MOTAR_ASSETS" --output "$output" \
  --steps 20000 --stop-after 20000 \
  --batch 64 --eval-batch 64 --eval-every 500 --eval-sources 512 \
  --seed 20260918 --lr 5e-4 --warmup 500 --save-every 1000 \
  --wandb-project mixtok-continuous-halton
