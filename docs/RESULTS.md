# Results

All rows below use the same frozen unseen 100-seed manifests. Checkpoints, embedded normalization
statistics, configs, hashes and result directories are bound in `MODEL_REGISTRY.json`.

| Method | Lift | Camera | ThreeStack | LPD | TakePhoto | Mean |
|---|---:|---:|---:|---:|---:|---:|
| DINOv3-ACT (RGB) | 87 | 98 | 3 | 0 | 14 | 40.4 |
| Stereo-ACT + Cross-RelBias | 92 | 100 | 42 | 20 | 29 | 56.6 |
| Stereo-ACT + FFN-MoE (E=4) | 100 | 99 | 68 | 17 | 28 | 62.4 |
| Stereo-ACT + Local-ARCA | 96 | 100 | 81 | 15 | 29 | 64.2 |
| Stereo-CoRE (SR@1) | 99 | 100 | 99 | 94 | 29 | 84.2 |
| Stereo-CoRE (Recovery@3, supplementary) | 99 | 100 | 99 | 95 | 30 | 84.6 |

The primary formal metric is single-rollout SR@1. Recovery@3 reruns only the declared original
failure/disagreement seeds up to three times; it is marked with a dagger in paper tables and must
not be presented as single-rollout SR@1. Raw baseline JSON files are under
`results/all5_baselines/`; Stereo-CoRE raw and recovery summaries are under `results/stereo_core/`.
