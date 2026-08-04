from __future__ import annotations

from importlib import import_module
from typing import Any


CANDIDATE_SPECS = {
    "p0": {
        "kind": "vjepa2_predictor",
        "module": "before_we_act.team_belief.vjepa2_predictor",
        "official_repo": "https://github.com/facebookresearch/vjepa2.git",
    },
    "p1": {
        "kind": "lpwm_particle",
        "module": "before_we_act.team_belief.lpwm_particle",
        "official_repo": "https://github.com/taldatech/lpwm.git",
    },
    "p2": {
        "kind": "dino_wm_feature_dynamics",
        "module": "before_we_act.team_belief.dino_wm_feature_dynamics",
        "official_repo": "https://github.com/gaoyuezhou/dino_wm.git",
    },
    "p3": {
        "kind": "lerobot_vla_jepa",
        "module": "before_we_act.team_belief.lerobot_vla_jepa",
        "official_repo": "https://github.com/huggingface/lerobot.git",
    },
}


def build_candidate_encoder(candidate_id: str, config: dict[str, Any]):
    try:
        spec = CANDIDATE_SPECS[candidate_id]
    except KeyError as exc:
        raise ValueError(f"unknown R11 candidate {candidate_id!r}") from exc
    if config.get("kind") != spec["kind"]:
        raise ValueError("candidate ID and registered kind differ")
    try:
        module = import_module(spec["module"])
    except ModuleNotFoundError as exc:
        if exc.name == spec["module"]:
            raise RuntimeError(
                f"{candidate_id} implementation is absent from this independent branch"
            ) from exc
        raise
    return module.build_encoder(config)
