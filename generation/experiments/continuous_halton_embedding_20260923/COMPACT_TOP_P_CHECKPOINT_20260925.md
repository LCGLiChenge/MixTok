# Inference-only top-p tokenizer checkpoint

The full top-p end-to-end training checkpoint is an external asset. Place it under `assets/reconstruction/` or pass its path with `--source`.

- Original: 6,400,703,289 bytes, SHA-256 `61e6e22acf5e4f7dcf44158518e4061a3fc9d6ca481ddc753e97f822546cb819`, step 247482.
- Inference export: `top_p_ema_inference_v1_step247482.pt`, 1,509,028,900 bytes, SHA-256 `e3806ee8011101a1f6bda002d18f2dd953f99b3dbfdf98d1be3126f1cb85547a`.
- The export retains the exact `model_ema` tensors under `latent_decoder.` (318 tensors, 305,315,328 elements) and `llamagen_vq.` (343 tensors, 71,883,403 elements). Total: 661 tensors, 377,198,731 elements. Dtypes and values are unchanged. Optimizers, discriminator, and unused model components are excluded.
- `export_top_p_inference.py` checks the original SHA-256 and step, then verifies every exported tensor after a save/load round trip.
- On H20, `verify_top_p_inference.py` strict-loaded both variants with the same official TiTok asset root and compared all 392 native TiTok state tensors and 661 decoder/VQ state tensors. Result: bitwise equal.
- `eval_worker_compact.py` and `parallel_eval_compact.py` are eval-only variants. They pin both artifact SHA-256 values and preserve the original training-checkpoint SHA-256 as model lineage, while recording the actual inference artifact SHA-256 separately. Sampling and decoder arithmetic are unchanged.

The compact file is for inference/evaluation only; it cannot resume tokenizer training. Tensor equality verifies the loaded weights, not a new FID measurement.
