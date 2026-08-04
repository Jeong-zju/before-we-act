# Reproduce

1. Clone RoboFactory at `5868242322414a91454e22f1dd9641f613ba1bcf` and install its environment.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Run `python scripts/download_artifacts.py` (the dataset requires roughly 180 GB).
4. Verify checkpoint hashes against `artifacts/Stereo-CoRE/SHA256SUMS.json`.
5. Export the official DeFM checkpoint path as `DEFM_CHECKPOINT`; DINOv3 access follows its
   upstream Hugging Face license and authentication requirements.
6. Run `bash scripts/audit_data.sh`, then `bash scripts/train_stereo_core.sh` or
   `bash scripts/evaluate_frozen100.sh <checkpoint> <task>`.

The training entry point exactly reproduces the released main configuration: All-5 data,
120k optimizer updates, global batch 40, weighted item sampling, capability target every four
updates, and checkpoints at 60k/80k/100k/120k. Set `DATA_ROOT`, `MODEL_ROOT`, `OUTPUT`, and
`WORKERS` to override paths or loader count without changing the method. Training uses one GPU,
matching the released run; independent evaluations can safely occupy the remaining GPUs.

The exact train/held-out episode split and first-20 manifests are in `protocol/`. The formal main
metric is single-rollout frozen-seed SR@1. Recovery@3 is supplementary and is never substituted
for SR@1 in the raw results.
