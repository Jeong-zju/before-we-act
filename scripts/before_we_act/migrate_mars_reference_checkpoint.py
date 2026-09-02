#!/usr/bin/env python3
"""Create a non-destructive action-contract wrapper for a legacy B-core/TUNE checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from before_we_act.care_training_data import sha256_file
from before_we_act.mars_action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract_hash,
    checkpoint_action_contract,
    normalization_stats_hash,
    validate_checkpoint_action_contract,
)


def migrate(source: Path, output: Path) -> None:
    if output.exists():
        value = torch.load(output, map_location="cpu", weights_only=False)
        validate_checkpoint_action_contract(value)
        if value.get("source_checkpoint_sha256") != sha256_file(source):
            raise RuntimeError("migrated reference source hash drifted")
        return
    value = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("reference checkpoint must be a mapping")
    if "stats" not in value or not isinstance(value["stats"], dict):
        raise ValueError("reference checkpoint is missing stats")
    stats = dict(value["stats"])
    stats.update(
        {
            "format_version": "before-we-act.mars.normalization-absolute/4-action-contract",
            "action_contract_version": ACTION_CONTRACT_VERSION,
        }
    )
    stats["action_contract"] = checkpoint_action_contract()
    normalization = normalization_stats_hash(stats)
    stats["normalization_sha256"] = normalization
    contract = checkpoint_action_contract(
        normalization_sha256=normalization,
        migrated_from_sha256=sha256_file(source),
    )
    value["stats"] = stats
    value["action_contract"] = contract
    config = dict(value.get("config", {}))
    config.update(
        {
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "action_contract_sha256": action_contract_hash(),
            "normalization_sha256": normalization,
        }
    )
    value["config"] = config
    value["source_checkpoint_sha256"] = sha256_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)
    validate_checkpoint_action_contract(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
