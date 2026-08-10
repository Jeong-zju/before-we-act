# LaWAM source license map

The pinned upstream has no repository-root `LICENSE` file. Its
`pyproject.toml` and README declare MIT. `LICENSE-MIT` records that declared
license and the upstream author list; this does not erase file-level notices.

- `starVLA/model/framework/vlas/cross_attention_dit.py`: Apache-2.0,
  Copyright 2025 NVIDIA Corporation & Affiliates.
- `starVLA/model/framework/vlas/flowmatching_expert.py`: Apache-2.0,
  Copyright 2025 HuggingFace Inc. team.
- `latent_action_model/core/utils/{modules,pos_embs,vision_transformer}.py`:
  MIT, Copyright Meta Platforms, Inc. and affiliates.
- `starVLA/training/trainer_utils/overwatch.py`: MIT, derived from OpenVLA.
- Other receipted files: upstream repository MIT declaration.

The full per-file mapping and exact hashes are in `SOURCE_RECEIPT.json`.
Apache-licensed files retain their complete SPDX/copyright headers in the
verified read-only checkout; the full Apache-2.0 text is in
`LICENSE-APACHE-2.0`.
