#!/usr/bin/env python3
"""Build an identity-complete LPD gate summary from two rollout summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.m1_statistics import wilson_interval  # noqa: E402
from train.m2_checkpointing import m2_checkpoint_tree_sha256  # noqa: E402


FORMAT_VERSION = "wam.robofactory.lpd_fixed_seed_gate/2"
TASKS = ("lift_barrier", "long_pipeline_delivery")
POLICY_KINDS = {"wam", "static_act", "agent_flow"}
FILE_CHECKPOINT_POLICY_KINDS = {"static_act", "agent_flow"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "formal"), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--policy-kind", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--lift-summary", type=Path, required=True)
    parser.add_argument("--lpd-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = args.config.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    lift = _read_mapping(args.lift_summary.expanduser().resolve(strict=True))
    lpd = _read_mapping(args.lpd_summary.expanduser().resolve(strict=True))
    summary = build_gate_summary(
        mode=args.mode,
        experiment=args.experiment,
        policy_kind=args.policy_kind,
        config=config,
        checkpoint=checkpoint,
        source_commit=args.source_commit,
        seed_start=args.seed_start,
        episodes=args.episodes,
        lift=lift,
        lpd=lpd,
    )
    _atomic_json(args.output.expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2))
    return 0


def build_gate_summary(
    *,
    mode: str,
    experiment: str,
    policy_kind: str,
    config: Path,
    checkpoint: Path,
    source_commit: str,
    seed_start: int,
    episodes: int,
    lift: Mapping[str, Any],
    lpd: Mapping[str, Any],
) -> dict[str, Any]:
    if mode not in {"gate", "formal"}:
        raise ValueError("mode must be gate or formal")
    if not experiment:
        raise ValueError("experiment must be non-empty")
    if policy_kind not in POLICY_KINDS:
        raise ValueError("policy kind must be wam, static_act or agent_flow")
    if episodes <= 0 or seed_start < 0:
        raise ValueError("seed protocol is invalid")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source commit must be a lowercase 40-character SHA")
    validated = {
        "lift_barrier": _task_summary(
            lift,
            expected_episodes=episodes,
            seed_start=seed_start,
        ),
        "long_pipeline_delivery": _task_summary(
            lpd,
            expected_episodes=episodes,
            seed_start=seed_start,
        ),
    }
    lift_client = _mapping(lift, "client")
    lpd_client = _mapping(lpd, "client")
    if lift_client != lpd_client:
        raise ValueError("tasks used different inference client identities")
    lift_digest = _checkpoint_digest(lift_client)
    lpd_digest = _checkpoint_digest(lpd_client)
    if lift_digest != lpd_digest:
        raise ValueError("tasks used different checkpoint identities")
    if lift_client.get("checkpoint_format") != lpd_client.get("checkpoint_format"):
        raise ValueError("tasks used different checkpoint formats")
    actual_checkpoint_digest = _checkpoint_path_digest(
        checkpoint,
        policy_kind=policy_kind,
    )
    if lift_digest != actual_checkpoint_digest:
        raise ValueError("client checkpoint identity differs from gate checkpoint")
    client_checkpoint = lift_client.get("checkpoint")
    if (
        not isinstance(client_checkpoint, str)
        or Path(client_checkpoint).expanduser().resolve(strict=True) != checkpoint
    ):
        raise ValueError("client checkpoint path differs from gate checkpoint")
    config_digest = _sha256(config)
    client_config_digest = lift_client.get("config_sha256")
    if (
        client_config_digest is not None
        and client_config_digest != config_digest
    ):
        raise ValueError("client config identity differs from gate config")
    client_config = lift_client.get("config")
    if (
        client_config is not None
        and (
            not isinstance(client_config, str)
            or Path(client_config).expanduser().resolve(strict=True) != config
        )
    ):
        raise ValueError("client config path differs from gate config")
    passed = (
        all(task["success_rate"] >= 0.90 for task in validated.values())
        if mode == "formal"
        else all(task["successes"] >= 1 for task in validated.values())
    )
    return {
        "format_version": FORMAT_VERSION,
        "mode": mode,
        "experiment": experiment,
        "candidate": {
            "source_commit": source_commit,
            "policy_kind": policy_kind,
            "config": str(config),
            "config_sha256": config_digest,
            "checkpoint": str(checkpoint),
            "checkpoint_format": lift_client["checkpoint_format"],
            "checkpoint_sha256": lift_digest,
            "client": dict(lift_client),
        },
        "seed_protocol": {
            "seed_start": seed_start,
            "episodes_per_task": episodes,
            "identical_across_tasks": True,
        },
        **validated,
        "passed": passed,
    }


def _task_summary(
    summary: Mapping[str, Any],
    *,
    expected_episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    if (
        summary.get("completed") is not True
        or summary.get("fatal_error") is not None
        or int(summary.get("episodes_completed", -1)) != expected_episodes
        or float(summary.get("direct_model_action_coverage", -1.0)) != 1.0
    ):
        raise ValueError("rollout summary did not complete direct-model evaluation")
    raw_episodes = summary.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != expected_episodes:
        raise ValueError("rollout summary episode records are incomplete")
    compact: list[dict[str, Any]] = []
    for offset, value in enumerate(raw_episodes):
        episode = _mapping(value)
        seed = episode.get("seed")
        success = episode.get("success")
        if seed != seed_start + offset or not isinstance(success, bool):
            raise ValueError("rollout summary violates the fixed-seed schedule")
        compact.append(
            {
                key: episode[key]
                for key in (
                    "seed",
                    "success",
                    "stop_reason",
                    "steps",
                    "inference_latency_ms",
                )
                if key in episode
            }
        )
    successes = sum(bool(episode["success"]) for episode in compact)
    reported_successes = int(summary.get("successes", -1))
    if successes != reported_successes:
        raise ValueError("rollout aggregate disagrees with episode records")
    interval = wilson_interval(successes, expected_episodes)
    return {
        "successes": successes,
        "success_rate": successes / expected_episodes,
        "success_rate_wilson_95": [
            interval["lower"],
            interval["upper"],
        ],
        "episodes": compact,
    }


def _checkpoint_digest(client: Mapping[str, Any]) -> str:
    value = client.get("checkpoint_sha256", client.get("checkpoint_tree_sha256"))
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("client checkpoint SHA-256 is missing or invalid")
    return value


def _checkpoint_path_digest(path: Path, *, policy_kind: str) -> str:
    if policy_kind in FILE_CHECKPOINT_POLICY_KINDS:
        if not path.is_file():
            raise ValueError(f"{policy_kind} checkpoint must be a file")
        return _sha256(path)
    if not path.is_dir():
        raise ValueError("WAM checkpoint must be a directory")
    return m2_checkpoint_tree_sha256(path)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    return _mapping(value)


def _mapping(value: Any, key: str | None = None) -> Mapping[str, Any]:
    result = value if key is None else value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key or 'value'} must be a mapping")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
