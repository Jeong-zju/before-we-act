#!/usr/bin/env python3
"""End-to-end GauDP backend gate before committing to the formal run."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


OUTPUT = Path("/workspace/bwa_gau_dp_runs/audit/runtime_contract.json")
TEMP_CHECKPOINT = Path("/workspace/bwa_gau_dp_runs/audit/runtime_smoke.ckpt")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    sys.path.insert(0, "/workspace/repos/Policy-Lightning")
    sys.path.insert(0, "/workspace/repos/before-we-act")
    from bwa.robofactory_gaudp_dataset import RoboFactoryGauDPDataset
    from bwa.train_robofactory_gaudp import build_model
    from deployment.vla_baselines.policy_rpc_server import GauDPBackend

    dataset = RoboFactoryGauDPDataset()
    _cfg, model = build_model()
    model.set_normalizer(dataset.get_normalizer())
    TEMP_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, TEMP_CHECKPOINT)
    del model
    gc.collect()

    sample = dataset[0]
    image = np.rint(sample["obs"]["head_cam_0"][-1].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    state = sample["obs"]["state"][-1].astype(np.float32)
    backend = GauDPBackend(str(TEMP_CHECKPOINT), "/workspace/datasets/robofactory_multitask")
    backend.reset()
    observations = [
        {"agent": agent, "task": "long_pipeline_delivery", "image": image, "state": state}
        for agent in range(4)
    ]
    chunks = backend.infer(observations)
    if len(chunks) != 4:
        raise RuntimeError(f"expected four local action chunks, got {len(chunks)}")
    for agent, chunk in enumerate(chunks):
        if chunk.shape != (6, 8) or not np.isfinite(chunk).all():
            raise RuntimeError(f"agent {agent}: invalid action chunk {chunk.shape}")
    peak = int(torch.cuda.max_memory_allocated())
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    TEMP_CHECKPOINT.unlink(missing_ok=True)
    atomic_json(
        OUTPUT,
        {
            "schema": "bwa.gaudp.runtime_contract.v1",
            "status": "complete",
            "agents": 4,
            "action_chunk_shape": [6, 8],
            "finite_outputs": True,
            "online_gaussian": "batched_single_view_local_noposplat_self_mode",
            "observation_history": "per_agent_rgb_gaussian_qpos9_three_frames",
            "policy_contract": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
            "peak_gpu_memory_bytes": peak,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
