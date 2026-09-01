from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_PATHS = {
    "duobench_config.py": "deployment/duo_pi05/openpi_overlay/src/openpi/training/duobench_config.py",
    "train_stage.py": "deployment/duo_pi05/train_stage.py",
    "compute_norm.py": "deployment/duo_pi05/compute_norm.py",
    "audit_contract.py": "deployment/duo_pi05/audit_contract.py",
    "dataset_adapter.py": "deployment/duo_pi05/dataset_adapter.py",
    "duobench_dataset.py": "deployment/duo_pi05/openpi_overlay/src/openpi/training/duobench_dataset.py",
    "duobench_policy.py": "deployment/duo_pi05/openpi_overlay/src/openpi/policies/duobench_policy.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the released DuoBench pi0.5 reproduction contract")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--final-report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config or repo / "configs/pi05_duobench_lora_formal_v1.json"
    frozen = json.loads(config_path.read_text())
    compact = json.loads((repo / "deployment/duo_pi05/duobench_pi05_lora_v1.json").read_text())
    resolved = frozen["openpi_train_config"]
    checks: dict[str, bool] = {}

    for name, relative in SOURCE_PATHS.items():
        checks[f"source_sha256:{name}"] = sha256(repo / relative) == frozen["source"]["source_file_sha256"][name]

    checks.update(
        {
            "upstream_architecture": resolved["model"]["pi05"] is True
            and resolved["model"]["action_dim"] == 32
            and resolved["model"]["action_horizon"] == 16
            and resolved["model"]["paligemma_variant"] == "gemma_2b_lora"
            and resolved["model"]["action_expert_variant"] == "gemma_300m_lora",
            "fixed_seed": resolved["seed"] == compact["optimization"]["seed"] == 42,
            "fixed_weight_decay": resolved["optimizer"]["weight_decay"]
            == compact["optimization"]["weight_decay"]
            == 1e-10,
            "fixed_budget": resolved["num_train_steps"] == compact["optimization"]["updates"] == 25_000
            and resolved["batch_size"] == compact["optimization"]["global_batch_size"] == 128,
            "fixed_schedule": resolved["lr_schedule"]["warmup_steps"] == compact["optimization"]["warmup_steps"] == 500
            and resolved["lr_schedule"]["peak_lr"] == compact["optimization"]["peak_lr"] == 5e-5
            and resolved["lr_schedule"]["decay_lr"] == compact["optimization"]["decay_lr"] == 5e-6,
            "all_data_no_split": frozen["data"]["episodes"] == 550 and frozen["data"]["split"].startswith("none"),
            "decentralized_contract": frozen["policy_contract"]
            == "shared_weights_decentralized_head_rgb_local_wrist_rgb_own_state8_to_own_action8",
            "normalization_hash_bound": len(frozen["normalization"]["sha256"]) == 64,
            "checkpoint_hash_bound": len(frozen["checkpoint"]["tree_sha256"]) == 64,
        }
    )

    if args.final_report:
        report = json.loads(args.final_report.read_text())
        training = report["training"]
        checks.update(
            {
                "report_complete": report["status"] == "complete" and report["validation20"]["total_episodes"] == 220,
                "report_revisions": report["source_revisions"] == {
                    "openpi": frozen["source"]["openpi_commit"],
                    "duobench": frozen["source"]["duobench_commit"],
                    "rcs": frozen["source"]["rcs_commit"],
                    "dataset": frozen["source"]["dataset_revision"],
                },
                "report_training": training["updates"] == resolved["num_train_steps"]
                and training["global_batch_size"] == resolved["batch_size"]
                and training["devices"] == resolved["effective_launch"]["devices"],
                "report_checkpoint": report["checkpoint_tree_sha256"] == frozen["checkpoint"]["tree_sha256"],
                "report_policy_contract": report["policy_contract"] == frozen["policy_contract"],
            }
        )

    payload = {
        "schema": "bwa.pi05.duobench.release-audit.v1",
        "status": "complete" if all(checks.values()) else "failed",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
