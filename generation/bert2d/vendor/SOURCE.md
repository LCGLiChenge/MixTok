# ADM evaluator provenance

`adm_evaluator.py` is the local evaluation implementation used for this project's
reported BERT sparse-2D 5k measurements, derived from OpenAI guided-diffusion:
https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py

Copyright and license: see `LICENSE.guided-diffusion` (MIT).

At packaging on 2026-09-15, AST comparisons against upstream found identical
`Evaluator`, `FIDStatistics`, and `DistanceBlock` definitions. Local changes are
in the standalone CLI `main` (not invoked by bert2d) and `ManifoldEstimator`
(precision/recall, not used by this FID/IS pipeline). The exact local file was
retained for numerical-protocol continuity. Asset URLs, sizes and hashes are in
`bert2d/assets.py`; no reference images or graph weights are stored in Git.
