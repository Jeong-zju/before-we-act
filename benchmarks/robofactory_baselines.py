"""Shared contract for the seven RoboFactory baseline adapters.

The contract intentionally separates upstream implementation availability from
the benchmark protocol.  A smoke run can therefore validate the data, process,
checkpoint and reporting plumbing without silently claiming that a surrogate
model is an upstream reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

SIX_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    display_name: str
    family: str
    source_url: str
    implementation_status: str
    upstream_train_entry: str | None
    upstream_eval_entry: str | None
    notes: str


BASELINES = (
    BaselineSpec(
        "act", "ACT", "CVAE action chunking", "https://github.com/tonyzhaozh/act",
        "native-local", "RoboFactory/robofactory/policy/ACT/train_act.py", None,
        "RoboFactory-local ACT implementation; multi-task adapter uses the shared contract.",
    ),
    BaselineSpec(
        "dp", "Diffusion Policy", "conditional diffusion", "https://github.com/real-stanford/diffusion_policy",
        "native-local", "RoboFactory/robofactory/policy/Diffusion-Policy/train.py", None,
        "RoboFactory-local DP implementation; multi-task adapter uses the shared contract.",
    ),
    BaselineSpec(
        "latent_tom", "LatentToM", "latent theory-of-mind", "https://github.com/robin-lab/LatentToM",
        "adapter-required", None, None,
        "No checked-in upstream trainer was found; keep this status until the upstream commit is pinned.",
    ),
    BaselineSpec(
        "gaudp", "GauDP", "Gaussian scene diffusion", "https://github.com/GAU-Scene/GauDP",
        "adapter-required", None, None,
        "No checked-in upstream trainer was found; keep this status until the upstream commit is pinned.",
    ),
    BaselineSpec(
        "maniflow", "ManiFlow", "flow matching", "https://github.com/Flow Matching/ManiFlow",
        "adapter-required", None, None,
        "No checked-in upstream trainer was found; keep this status until the upstream commit is pinned.",
    ),
    BaselineSpec(
        "rdt_1b", "RDT-1B", "robot diffusion transformer", "https://github.com/thu-ml/RoboticsDiffusionTransformer",
        "adapter-required", None, None,
        "RDT-1B weights are too large for an unverified smoke; pin the exact checkpoint before full training.",
    ),
    BaselineSpec(
        "openvla_oft", "OpenVLA-OFT", "VLA fine-tuning", "https://github.com/moojink/openvla-oft",
        "adapter-required", None, None,
        "OpenVLA-OFT requires a pinned VLA checkpoint and image-language preprocessing contract.",
    ),
)


def build_contract(*, data_root: str | Path, output_root: str | Path, seed: int = 20260819) -> dict[str, Any]:
    """Return the frozen, fair-comparison settings shared by every adapter."""
    return {
        "format_version": "before-we-act.robofactory-baseline-contract/1",
        "tasks": list(SIX_TASKS),
        "episodes_per_task": 20,
        "validation": "closed_loop",
        "train_split": "all six task training manifests, seed-disjoint",
        "seed": int(seed),
        "action_semantics": "raw commanded pd_joint_pos decoded through task codec",
        "observation_policy": "global RGB plus proprioception; no privileged state",
        "data_root": str(Path(data_root).expanduser()),
        "output_root": str(Path(output_root).expanduser()),
        "baselines": [asdict(spec) for spec in BASELINES],
    }


def validate_data_root(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    tasks: dict[str, Any] = {}
    missing: list[str] = []
    for task in SIX_TASKS:
        path = root / task
        manifest = path / "training_manifest.json"
        if not manifest.is_file():
            missing.append(task)
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            episodes = payload.get("episodes", payload.get("records", []))
            tasks[task] = {
                "manifest": str(manifest),
                "episodes": len(episodes) if isinstance(episodes, list) else None,
                "sha256_file": str(manifest) + ".sha256",
                "normalization": (path / "normalization.npz").is_file(),
            }
        except (OSError, json.JSONDecodeError) as exc:
            missing.append(f"{task}: {exc}")
    return {"root": str(root), "tasks": tasks, "missing": missing, "valid": not missing and len(tasks) == len(SIX_TASKS)}


def aggregate_validation20(results_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    """Aggregate one result JSON per baseline/task into macro and micro rates."""
    root = Path(results_root).expanduser().resolve()
    rows: dict[str, dict[str, Any]] = {}
    for spec in BASELINES:
        per_task: dict[str, dict[str, Any]] = {}
        for task in SIX_TASKS:
            candidates = (root / spec.key / task / "summary.json", root / spec.key / f"{task}.json")
            path = next((item for item in candidates if item.is_file()), None)
            if path is None:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            successes = int(payload.get("successes", payload.get("success_count", 0)))
            episodes = int(payload.get("episodes", payload.get("episodes_completed", 20)))
            per_task[task] = {"successes": successes, "episodes": episodes, "rate": successes / episodes if episodes else 0.0, "source": str(path)}
        total_successes = sum(row["successes"] for row in per_task.values())
        total_episodes = sum(row["episodes"] for row in per_task.values())
        rows[spec.key] = {
            "display_name": spec.display_name,
            "implementation_status": spec.implementation_status,
            "tasks": per_task,
            "tasks_completed": len(per_task),
            "macro_success_rate": (sum(row["rate"] for row in per_task.values()) / len(per_task)) if per_task else None,
            "micro_success_rate": total_successes / total_episodes if total_episodes else None,
            "episodes": total_episodes,
            "successes": total_successes,
        }
    report = {"format_version": "before-we-act.robofactory-validation20/1", "tasks": list(SIX_TASKS), "episodes_per_task": 20, "baselines": rows}
    if output is not None:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
