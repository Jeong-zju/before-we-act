#!/usr/bin/env python3
"""Audit the frozen DuoBench RDT-1B policy/training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_TRAINER_ARGUMENTS = {
    "adam_beta1", "adam_beta2", "adam_epsilon", "adam_weight_decay",
    "allow_tf32", "alpha", "cam_ext_mask_prob", "checkpointing_period",
    "checkpoints_total_limit", "cond_mask_prob", "config_path",
    "dataloader_num_workers", "dataset_type", "deepspeed",
    "gradient_accumulation_steps", "gradient_checkpointing", "hub_model_id",
    "hub_token", "image_aug", "learning_rate", "load_from_hdf5",
    "local_rank", "logging_dir", "lr_num_cycles", "lr_power",
    "lr_scheduler", "lr_warmup_steps", "max_grad_norm", "max_train_steps",
    "mixed_precision", "num_sample_batches", "num_train_epochs", "output_dir",
    "precomp_lang_embed", "pretrained_model_name_or_path",
    "pretrained_text_encoder_name_or_path",
    "pretrained_vision_encoder_name_or_path", "push_to_hub", "report_to",
    "resume_from_checkpoint", "sample_batch_size", "sample_period", "scale_lr",
    "seed", "set_grads_to_none", "state_noise_snr", "train_batch_size",
    "use_8bit_adam",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "contract",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "configs/rdt/duobench_rdt1b_full_data_v1.json",
    )
    args = parser.parse_args()
    raw = args.contract.read_bytes()
    cfg = json.loads(raw)

    assert cfg["schema"] == "before-we-act.rdt.duobench.full-data/1"
    assert cfg["status"] == "frozen_completed_run"
    actual_args = set(cfg["resolved_trainer_arguments"])
    assert actual_args == EXPECTED_TRAINER_ARGUMENTS, {
        "missing": sorted(EXPECTED_TRAINER_ARGUMENTS - actual_args),
        "unexpected": sorted(actual_args - EXPECTED_TRAINER_ARGUMENTS),
    }

    trainer = cfg["resolved_trainer_arguments"]
    assert trainer["seed"] is None
    assert trainer["max_train_steps"] == 215000
    assert trainer["train_batch_size"] == 4
    assert trainer["gradient_accumulation_steps"] == 1
    assert trainer["mixed_precision"] == "bf16"
    assert trainer["load_from_hdf5"] is True
    assert trainer["dataset_type"] == "finetune"
    assert trainer["lr_scheduler"] == "constant"

    data = cfg["data_configuration"]
    assert data["total_episodes"] == 550
    assert data["total_frames"] == 285988
    assert data["causal_pairs"] == 285438
    assert data["local_arm_streams"] == 1100
    assert data["train_test_split"].startswith("none")

    model = cfg["model_configuration"]
    assert model["total_parameters"] == 1228319872
    assert model["trainable_parameters"] == 1228319872
    assert model["all_model_parameters_trainable"] is True
    assert model["diffusion"]["num_train_timesteps"] == 1000
    assert model["diffusion"]["num_inference_timesteps"] == 5

    retention = cfg["checkpoint_and_supervisor"]["retention_contract"]
    assert retention["cli_checkpoints_total_limit"] == 2
    assert retention["cli_limit_effective"] is False
    assert retention["external_gc_keep"] == 2

    validation = cfg["validation20"]
    assert validation["total_episodes"] == 220
    assert validation["result"]["successes"] == 10
    assert validation["result"]["success_rate"] == 10 / 220

    print(json.dumps({
        "status": "ok",
        "contract": str(args.contract),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "trainer_argument_count": len(actual_args),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
