from __future__ import annotations

from importlib import import_module
from typing import Any


CANDIDATE_SPECS = {
    "p0": {
        "kind": "openpi_flow_action_expert",
        "module": "before_we_act.action_generator.openpi_flow",
        "official_repo": "https://github.com/Physical-Intelligence/openpi.git",
    },
    "p1": {
        "kind": "smolvla_flow_action_expert",
        "module": "before_we_act.action_generator.smolvla_flow",
        "official_repo": "https://github.com/huggingface/lerobot.git",
    },
    "p2": {
        "kind": "rdt_diffusion_transformer",
        "module": "before_we_act.action_generator.rdt_diffusion",
        "official_repo": "https://github.com/thu-ml/RoboticsDiffusionTransformer.git",
    },
    "p3": {
        "kind": "consistency_policy_ctm",
        "module": "before_we_act.action_generator.consistency_policy",
        "official_repo": "https://github.com/Aaditya-Prasad/Consistency-Policy.git",
    },
}


def build_action_core(candidate_id: str, config: dict[str, Any]):
    try:
        spec = CANDIDATE_SPECS[candidate_id]
    except KeyError as exc:
        raise ValueError(f"unknown R12 candidate {candidate_id!r}") from exc
    if config.get("kind") != spec["kind"]:
        raise ValueError("candidate ID and registered action kind differ")
    try:
        module = import_module(spec["module"])
    except ModuleNotFoundError as exc:
        if exc.name == spec["module"]:
            raise RuntimeError(
                f"{candidate_id} implementation is absent from this independent branch"
            ) from exc
        raise
    return module.build_core(config)
